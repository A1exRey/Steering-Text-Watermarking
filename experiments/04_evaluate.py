"""Stage 4: evaluate paraphrase robustness per condition.

Three evaluations:

1. Formality probe: train logistic regression on re-encoded paraphrased
   activations at STEER_LAYER, formality-steered vs unsteered. Report AUC.
2. Random probe: same, but random-steered vs unsteered. If random-direction
   AUC is similar to formality AUC, the whole thesis is broken - it means
   any residual-stream perturbation survives paraphrase, so there's nothing
   special about semantic directions.
3. Kirchenbauer detector AUC on paraphrased text, same positive/negative
   set. Expect this to be substantially weaker than the activation probes
   if the thesis holds.

Output: artifacts/results.json with all AUCs + sample sizes + config echo.
"""

import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import WatermarkDetector, WatermarkingConfig

import config
from utils import activation_capture, batched, load_llama, masked_mean


def extract_pooled_activations(model, tok, texts, layer_idx, batch_size=8, max_length=256):
    """Re-encode texts through model, return mean-pooled activations at layer_idx."""
    pooled = []
    for batch in tqdm(list(batched(texts, batch_size)), desc="re-encode"):
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        store = {}
        with activation_capture(model, layer_idx, store):
            with torch.no_grad():
                model(**enc)
        p = masked_mean(store["acts"].float(), enc["attention_mask"])
        pooled.append(p.cpu())
    return torch.cat(pooled, dim=0).numpy()


def probe_auc(X_pos, X_neg, seed):
    """Train logistic regression on (pos vs neg), return held-out AUC + other stats."""
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    probe = LogisticRegression(max_iter=2000, C=1.0).fit(X_tr, y_tr)
    scores = probe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, scores)
    train_auc = roc_auc_score(y_tr, probe.predict_proba(X_tr)[:, 1])
    return {"test_auc": float(auc), "train_auc": float(train_auc), "n_test": int(len(y_te))}


def evaluate_activation_probes(records, model, tok):
    """Extract activations once for each condition, then run two probes."""
    by_cond = {c: [] for c in ("unsteered", "formality", "random")}
    for r in records:
        if r["condition"] in by_cond:
            by_cond[r["condition"]].append(r)

    print("Re-encoding paraphrased text through base model...")
    acts = {}
    for cond, recs in by_cond.items():
        texts = [r["paraphrased"] for r in recs]
        acts[cond] = extract_pooled_activations(model, tok, texts, config.STEER_LAYER)
        print(f"  {cond}: {acts[cond].shape}")

    formality_result = probe_auc(acts["formality"], acts["unsteered"], seed=config.SEED)
    random_result = probe_auc(acts["random"], acts["unsteered"], seed=config.SEED)
    return {"formality_vs_unsteered": formality_result, "random_vs_unsteered": random_result}


def evaluate_kgw(records, model, tok):
    """Run HF WatermarkDetector on paraphrased text. AUC on z-scores."""
    kgw_cfg = WatermarkingConfig(
        greenlist_ratio=config.KGW_GAMMA,
        bias=config.KGW_BIAS,
        hashing_key=config.KGW_SEED,
        seeding_scheme=config.KGW_SCHEME,
        context_width=1,
    )
    detector = WatermarkDetector(
        model_config=model.config,
        device=model.device,
        watermarking_config=kgw_cfg,
    )

    pos = [r for r in records if r["condition"] == "kirchenbauer"]
    neg = [r for r in records if r["condition"] == "unsteered"]

    def get_zscores(recs):
        zs = []
        for r in tqdm(recs, desc="kgw detect"):
            ids = tok(
                r["paraphrased"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).input_ids.to(model.device)
            # Detector needs at least a couple of tokens
            if ids.shape[1] < 4:
                zs.append(0.0)
                continue
            with torch.no_grad():
                result = detector(ids, return_dict=True)
            # z_score is a numpy array of shape (batch,); we have batch=1
            z = float(np.asarray(result.z_score).reshape(-1)[0])
            zs.append(z)
        return zs

    pos_z = get_zscores(pos)
    neg_z = get_zscores(neg)

    y = np.concatenate([np.ones(len(pos_z)), np.zeros(len(neg_z))])
    s = np.concatenate([pos_z, neg_z])
    auc = roc_auc_score(y, s)
    return {
        "test_auc": float(auc),
        "n_pos": len(pos_z),
        "n_neg": len(neg_z),
        "pos_z_mean": float(np.mean(pos_z)),
        "neg_z_mean": float(np.mean(neg_z)),
    }


def main():
    with open(config.PARA_PATH) as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} paraphrased records.")

    model, tok = load_llama(config.GEN_MODEL)

    probe_results = evaluate_activation_probes(records, model, tok)
    kgw_result = evaluate_kgw(records, model, tok)

    results = {
        "config": {
            "gen_model": config.GEN_MODEL,
            "steer_layer": config.STEER_LAYER,
            "steer_multiplier": config.STEER_MULTIPLIER,
            "kgw_gamma": config.KGW_GAMMA,
            "kgw_bias": config.KGW_BIAS,
            "n_gen_prompts": config.N_GEN_PROMPTS,
        },
        "activation_probes": probe_results,
        "kirchenbauer": kgw_result,
    }

    with open(config.RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Pretty-print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (post-paraphrase)")
    print("=" * 60)
    f_auc = probe_results["formality_vs_unsteered"]["test_auc"]
    r_auc = probe_results["random_vs_unsteered"]["test_auc"]
    k_auc = kgw_result["test_auc"]
    print(f"  formality-steering probe AUC : {f_auc:.4f}")
    print(f"  random-direction probe AUC   : {r_auc:.4f}")
    print(f"  Kirchenbauer detector AUC    : {k_auc:.4f}")
    print()
    print(f"  H1 (formality survives)   : {'PASS' if f_auc > 0.75 else 'FAIL'}")
    print(f"  H2 (random does NOT)      : {'PASS' if r_auc < 0.65 else 'FAIL'}  (lower=better)")
    print(f"  H3 (KGW degrades)         : {'PASS' if k_auc < 0.75 else 'N/A'}")
    print()
    print(
        f"  Architectural win over KGW: {'YES' if (f_auc - k_auc) > 0.1 else 'NO'}  (gap={f_auc - k_auc:+.4f})"
    )
    print(
        f"  Direction choice matters  : {'YES' if (f_auc - r_auc) > 0.1 else 'NO'}  (gap={f_auc - r_auc:+.4f})"
    )
    print("=" * 60)
    print(f"Full results written to {config.RESULTS_PATH}")


if __name__ == "__main__":
    main()
