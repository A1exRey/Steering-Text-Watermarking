"""Localize the Qwen probe-inversion confound.

Symptom (Qwen3-1.7B sweep): every steering condition has raw post-paraphrase
AUC ~0.13-0.20 (i.e. ~0.80-0.86 INVERTED), nearly flat across method and
multiplier, present even at m1 where steering barely perturbs the text, and
identical for the random control. KGW (no steering hook) behaves normally.

That pattern = the probe is detecting "a steering hook ran during generation"
rather than the steering direction. This script finds the non-semantic
difference between steered and unsteered generations that the probe keys on.

Tests (reuse artifacts/<model>/sweep/*.jsonl; NO regeneration):
  T1. Length. Mean token/char length of steered vs unsteered completions. A
      systematic length shift fully explains a pooled-activation contrast.
  T2. Degenerate/empty/repetition rate. Hook side effects sometimes cause early
      EOS or repetition that the probe trivially detects.
  T3. Surface-only probe (no model) steered-vs-unsteered: if it already hits
      ~0.85, the contrast is shallow and not about activations at all.
  T4. Length-matched activation probe: if AUC collapses to ~0.5 once lengths are
      matched, length was the confound.
  T5. First-token / prefix check: print a few steered vs unsteered completions
      to eyeball any systematic prefix (e.g. a leaked template token, a stray
      <think> remnant, a leading space) that distinguishes the runs.

Run:  AOS_MODEL=qwen3-1.7b python 09_qwen_confound_diagnostic.py
"""

import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from utils import activation_capture, batched, load_llama, masked_mean

SWEEP_DIR = config.ART / "sweep"
SEEDS = [0, 1, 2]
CHECK = ["formality_m1.0", "random_m1.0", "sentiment_m3.0"]  # weak + control + strong


def load_all(tag, field):
    out = []
    for s in SEEDS:
        with open(SWEEP_DIR / f"para_{tag}_s{s}.jsonl") as f:
            out += [json.loads(l)[field] for l in f]
    return out


def cv_auc(Xp, Xn, seed=42):
    X = np.concatenate([Xp, Xn], 0)
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5))
    oof = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, oof))
    return auc, max(auc, 1 - auc)


def surface(tok, texts):
    f = []
    for t in texts:
        toks = tok(t).input_ids
        w = t.split()
        nc = max(len(t), 1)
        f.append(
            [
                len(toks),
                nc,
                np.mean([len(x) for x in w]) if w else 0,
                len(set(x.lower() for x in w)) / max(len(w), 1),
                t.count(".") / nc,
                t.count(",") / nc,
            ]
        )
    return np.array(f, float)


def pooled(model, tok, texts, layer, bs=8, ml=256):
    texts = [t if (t and t.strip()) else "." for t in texts]  # guard zero-len (Qwen crash)
    out = []
    for b in batched(texts, bs):
        enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=ml).to(
            model.device
        )
        st = {}
        with activation_capture(model, layer, st):
            with torch.no_grad():
                model(**enc)
        out.append(masked_mean(st["acts"].float(), enc["attention_mask"]).cpu())
    return torch.cat(out, 0).numpy()


def lenmatch(a, b, tok, rng):
    la = np.array([len(tok(t).input_ids) for t in a])
    lb = np.array([len(tok(t).input_ids) for t in b])
    bins = np.linspace(0, max(la.max(), lb.max()) + 1, 11)
    ia, ib = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        ai = np.where((la >= lo) & (la < hi))[0]
        bi = np.where((lb >= lo) & (lb < hi))[0]
        k = min(len(ai), len(bi))
        if k:
            ia += list(rng.choice(ai, k, replace=False))
            ib += list(rng.choice(bi, k, replace=False))
    return np.array(ia), np.array(ib)


def degen_rate(texts):
    """Fraction of completions that are empty, very short, or highly repetitive."""
    bad = 0
    for t in texts:
        w = t.split()
        if len(t.strip()) < 5:
            bad += 1
        elif len(w) >= 8 and len(set(w)) / len(w) < 0.4:  # low type-token = repetition
            bad += 1
    return bad / max(len(texts), 1)


def main():
    rng = np.random.default_rng(0)
    model, tok = load_llama(config.GEN_MODEL)
    print(
        f"model={config.GEN_MODEL} steer_layer={config.STEER_LAYER} "
        f"pad_token={tok.pad_token!r} pad_id={tok.pad_token_id} pad_side={tok.padding_side}\n"
    )

    uns = load_all("unsteered", "paraphrased")
    uns_orig = load_all("unsteered", "completion")

    for tag in CHECK:
        st = load_all(tag, "paraphrased")
        st_orig = load_all(tag, "completion")

        # T1 length (on ORIGINAL completions, where the hook acted)
        lu = np.array([len(tok(t).input_ids) for t in uns_orig])
        ls = np.array([len(tok(t).input_ids) for t in st_orig])
        # T2 degeneracy
        du, ds = degen_rate(uns_orig), degen_rate(st_orig)
        # T3 surface probe (paraphrased)
        _, surf_sym = cv_auc(surface(tok, st), surface(tok, uns))
        # T4 length-matched activation probe (paraphrased)
        ia, ib = lenmatch(st, uns, tok, rng)
        if len(ia) > 30:
            Xs = pooled(model, tok, [st[i] for i in ia], config.STEER_LAYER)
            Xu = pooled(model, tok, [uns[i] for i in ib], config.STEER_LAYER)
            _, lc_sym = cv_auc(Xs, Xu)
        else:
            lc_sym = float("nan")
        # full activation probe for reference
        _, full_sym = cv_auc(
            pooled(model, tok, st, config.STEER_LAYER), pooled(model, tok, uns, config.STEER_LAYER)
        )

        print(f"=== {tag} ===")
        print(
            f"  T1 length(tok): steered {ls.mean():.1f}±{ls.std():.1f}  "
            f"unsteered {lu.mean():.1f}±{lu.std():.1f}  diff {ls.mean() - lu.mean():+.1f}"
        )
        print(f"  T2 degenerate rate: steered {ds:.2%}  unsteered {du:.2%}")
        print(f"  T3 surface-only probe AUC (sym): {surf_sym:.3f}")
        print(f"  T4 length-matched activation AUC (sym): {lc_sym:.3f}  (n={len(ia)})")
        print(f"     full activation AUC (sym):           {full_sym:.3f}")
        # T5 eyeball
        print("  T5 sample completions (steered | unsteered):")
        for i in range(2):
            print(f"     S: {st_orig[i][:90]!r}")
            print(f"     U: {uns_orig[i][:90]!r}")
        print()

    print("=" * 64)
    print("HOW TO READ:")
    print("  * If steered length differs a lot from unsteered AND length-matched")
    print("    AUC (T4) collapses toward 0.5 -> LENGTH is the confound. Fix: match")
    print("    max_new_tokens / generation length across conditions, or length-")
    print("    normalize before pooling.")
    print("  * If degenerate rate (T2) is high for steered -> hook is breaking")
    print("    generation on Qwen (early EOS/repetition). Fix the hook/multiplier.")
    print("  * If surface-only (T3) ~= full activation AUC -> probe reads surface")
    print("    stats, not semantics; same artifact as before but stronger on Qwen.")
    print("  * If T5 shows a systematic prefix/format diff -> tokenizer/template")
    print("    confound; inspect the chat-template output for Qwen.")


if __name__ == "__main__":
    main()
