"""Diagnostic: distinguish "signal never existed" from "paraphrase destroyed it".

Reuses artifacts/generations.jsonl (the ORIGINAL, un-paraphrased completions).
No new generation. Answers two questions the main results.json can't:

  Q1. Pre-paraphrase probe AUC. If formality-vs-unsteered AUC on the original
      (un-paraphrased) activations is high (~0.95+), the steering signal exists
      and paraphrase is what kills it. If it's already ~0.7, steering at this
      layer/multiplier barely produced a separable signal -> fix generation,
      not robustness.

  Q2. Per-condition fluency (mean perplexity under the base model). If the
      random and/or formality conditions have much higher perplexity than
      unsteered, multiplier is too high and the probe may be detecting
      degradation artifacts rather than register. This would explain
      random >= formality.

Run:  python 05_diagnostic_preparaphrase.py
"""

import json

import numpy as np
import torch
from tqdm import tqdm

import config
from utils import activation_capture, batched, load_llama, masked_mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def extract_pooled_activations(model, tok, texts, layer_idx, batch_size=8, max_length=256):
    pooled = []
    for batch in tqdm(list(batched(texts, batch_size)), desc="re-encode"):
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model.device)
        store = {}
        with activation_capture(model, layer_idx, store):
            with torch.no_grad():
                model(**enc)
        pooled.append(masked_mean(store["acts"].float(), enc["attention_mask"]).cpu())
    return torch.cat(pooled, dim=0).numpy()


def probe_auc(X_pos, X_neg, seed):
    X = np.concatenate([X_pos, X_neg], 0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    p = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    return {
        "test_auc": float(roc_auc_score(yte, p.predict_proba(Xte)[:, 1])),
        "train_auc": float(roc_auc_score(ytr, p.predict_proba(Xtr)[:, 1])),
    }


def mean_perplexity(model, tok, texts, max_length=256, sample=120):
    """Mean per-token perplexity of texts under the base model (lower = more fluent)."""
    texts = texts[:sample]
    ppls = []
    for t in tqdm(texts, desc="ppl"):
        if len(t.strip()) < 5:
            continue
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        ppls.append(float(torch.exp(out.loss).item()))
    return float(np.mean(ppls)), float(np.median(ppls))


def main():
    with open(config.GEN_PATH) as f:
        records = [json.loads(line) for line in f]

    by_cond = {}
    for r in records:
        by_cond.setdefault(r["condition"], []).append(r["completion"])

    model, tok = load_llama(config.GEN_MODEL)

    # --- Q1: pre-paraphrase probe AUC ---
    print("\n[Q1] Extracting activations on ORIGINAL (un-paraphrased) completions...")
    acts = {}
    for cond in ("unsteered", "formality", "random"):
        acts[cond] = extract_pooled_activations(model, tok, by_cond[cond], config.STEER_LAYER)

    f_pre = probe_auc(acts["formality"], acts["unsteered"], config.SEED)
    r_pre = probe_auc(acts["random"], acts["unsteered"], config.SEED)

    # --- Q2: fluency per condition ---
    print("\n[Q2] Computing per-condition perplexity (fluency check)...")
    ppl = {}
    for cond in ("unsteered", "formality", "random", "kirchenbauer"):
        if cond in by_cond:
            ppl[cond] = mean_perplexity(model, tok, by_cond[cond])

    # --- load post-paraphrase numbers for side-by-side ---
    post = None
    if config.RESULTS_PATH.exists():
        with open(config.RESULTS_PATH) as f:
            post = json.load(f)

    print("\n" + "=" * 64)
    print("PRE-PARAPHRASE PROBE AUC  (the diagnostic number)")
    print("=" * 64)
    print(
        f"  formality vs unsteered : test={f_pre['test_auc']:.4f}  train={f_pre['train_auc']:.4f}"
    )
    print(
        f"  random    vs unsteered : test={r_pre['test_auc']:.4f}  train={r_pre['train_auc']:.4f}"
    )
    if post:
        fp = post["activation_probes"]["formality_vs_unsteered"]["test_auc"]
        rp = post["activation_probes"]["random_vs_unsteered"]["test_auc"]
        print("\n  pre -> post paraphrase:")
        print(
            f"    formality : {f_pre['test_auc']:.4f} -> {fp:.4f}   (drop {f_pre['test_auc'] - fp:+.4f})"
        )
        print(
            f"    random    : {r_pre['test_auc']:.4f} -> {rp:.4f}   (drop {r_pre['test_auc'] - rp:+.4f})"
        )

    print("\n" + "=" * 64)
    print("FLUENCY (perplexity under base model; lower = more fluent)")
    print("=" * 64)
    base = ppl.get("unsteered", (float("nan"), float("nan")))[0]
    for cond, (m, med) in ppl.items():
        ratio = m / base if base and not np.isnan(base) else float("nan")
        print(f"  {cond:13s}: mean={m:8.2f}  median={med:8.2f}  ({ratio:.2f}x unsteered)")

    print("\n" + "=" * 64)
    print("READING THE RESULT")
    print("=" * 64)
    fpre = f_pre["test_auc"]
    if fpre > 0.9:
        print("  Formality signal EXISTS pre-paraphrase (AUC>0.9).")
        print("  -> Paraphrase is what destroys it. The robustness thesis is")
        print("     the thing that failed, not the setup. This is a genuine")
        print("     (negative) finding. Decide if it's enough of a result or pivot.")
    elif fpre > 0.75:
        print("  Formality signal is WEAK even pre-paraphrase (0.75-0.9).")
        print("  -> Steering at this layer/multiplier is underpowered. Sweep")
        print("     multiplier (try 2.0, your known-good value) and layer BEFORE")
        print("     concluding anything about paraphrase robustness.")
    else:
        print("  Formality signal BARELY EXISTS pre-paraphrase (<0.75).")
        print("  -> Generation/steering is broken or mis-tuned. The post-paraphrase")
        print("     comparison is meaningless. Fix generation first: check the")
        print("     sanity samples, drop multiplier to 2.0, verify steering hook fires.")

    r_ppl = ppl.get("random", (float("nan"),))[0]
    if base and r_ppl / base > 1.5:
        print("\n  WARNING: random-direction perplexity >1.5x unsteered. multiplier=4")
        print("  is degrading fluency; the probe may detect distortion, not register.")
        print("  This likely explains random >= formality. Lower the multiplier.")

    out = {"pre_paraphrase": {"formality": f_pre, "random": r_pre}, "perplexity": ppl}
    with open(config.ART / "diagnostic.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {config.ART / 'diagnostic.json'}")


if __name__ == "__main__":
    main()
