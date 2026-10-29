"""Diagnostic: explain the non-monotone AUC and test whether the high
detectability at LOW multiplier (m1 ~0.86 for every method, incl. random) is a
shallow surface artifact rather than a semantic watermark signal.

Context / why this matters
--------------------------
The corrected sweep showed post-paraphrase symmetric AUC is NON-monotone in the
steering multiplier, and that m1 (barely-steered) is the MOST detectable point
for all four methods including the random control. If detection tracked the
intended semantic perturbation, weak steering would be the *least* detectable,
not the most. Before writing the paper we must know what the probe is keying on.

Note on mechanism: the sweep's probe re-encodes PARAPHRASED text through a CLEAN
model (no hook). So a literal additive hook offset cannot be the cause -- the
hook isn't active at probe time. The surviving signal must be a property of the
*text*. This script asks: is that property shallow surface statistics (length,
token-frequency, punctuation) that happen to survive paraphrase, or genuine
content the steering induced?

Tests (all reuse existing artifacts/sweep/*.jsonl -- NO new generation)
-----------------------------------------------------------------------
For each method at m1 and at m3 (a degraded point), compare steered-vs-unsteered:

  T1. Layer transfer. Probe pooled activations at a DIFFERENT layer than the
      steered one (config.STEER_LAYER). If separability is just as high at an
      unrelated layer, the signal is a global text property, not something tied
      to the steered representation.

  T2. Surface-feature probe. Train a logistic probe on hand-built SURFACE
      features only (token count, char count, mean word length, type-token
      ratio, punctuation rates, uppercase rate). If this matches the activation
      probe's AUC, the activation probe is just reading surface statistics.

  T3. Length-control. Re-run the activation probe after matching steered and
      unsteered sets on completion length (truncate to common length dist). If
      AUC collapses, length was the confound.

Outputs: prints a table; writes artifacts/sweep/artifact_diagnostic.json.
Run:  python 07_artifact_diagnostic.py
"""

import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from utils import activation_capture, batched, load_llama, masked_mean

SWEEP_DIR = config.ART / "sweep"
SEEDS = [0, 1, 2]

# Methods to inspect and the two multipliers of interest:
#   m1.0 = the anomalous "most detectable" weak-steering point
#   m3.0 = a mid/degraded point for contrast
METHODS = ["formality", "random", "sentiment", "complexity"]
# Norm-relative fractions matching the current sweep grid (see config). Use a
# low (barely-steered) and a mid point; tags are e.g. "formality_m0.1".
PROBE_MULTS = ["0.1", "0.4"]

# Layers to compare for the transfer test. Densified: probe EVERY OTHER layer
# across the model's depth, not just three points, so "layer-invariant" is
# defensible. STEER_LAYER is included and marked in the output.
_n_layers = getattr(config, "N_LAYERS", None)
if _n_layers is None:
    # infer from model config without loading weights if possible; fall back to
    # a span around the steered layer if unavailable.
    try:
        from transformers import AutoConfig

        _cfg = AutoConfig.from_pretrained(config.GEN_MODEL)
        _n_layers = _cfg.num_hidden_layers
    except Exception:
        _n_layers = config.STEER_LAYER * 2
ALT_LAYERS = sorted(set(range(0, _n_layers, 2)) | {config.STEER_LAYER})


def load_para(tag, seed, field="paraphrased"):
    p = SWEEP_DIR / f"para_{tag}_s{seed}.jsonl"
    with open(p) as f:
        return [json.loads(l)[field] for l in f]


def cv_auc(Xpos, Xneg, groups_pos, groups_neg, seed=42):
    """Symmetric cross-validated AUC, grouped by prompt.

    Grouping matters here for the same reason it does in 06_pareto_sweep.probe_auc:
    each prompt contributes one row per generation seed to BOTH classes, and at low
    steering those rows are near-duplicates. Ungrouped folds let the probe key on
    prompt identity, which inflates exactly the low-multiplier points this script
    exists to explain. StratifiedGroupKFold keeps a prompt wholly inside one fold.
    """
    X = np.concatenate([Xpos, Xneg], 0)
    y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))])
    groups = np.concatenate([np.asarray(groups_pos), np.asarray(groups_neg)])
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5))
    oof = cross_val_predict(clf, X, y, groups=groups, cv=skf, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, oof))
    return max(auc, 1.0 - auc)  # symmetric, matching the sweep


def pooled_acts(model, tok, texts, layer, bs=8, max_length=256):
    texts = [t if (t and t.strip()) else "." for t in texts]  # guard zero-len (Qwen crash)
    out = []
    for batch in batched(texts, bs):
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model.device)
        store = {}
        with activation_capture(model, layer, store):
            with torch.no_grad():
                model(**enc)
        out.append(masked_mean(store["acts"].float(), enc["attention_mask"]).cpu())
    return torch.cat(out, 0).numpy()


def pooled_acts_all_layers(model, tok, texts, layers, bs=8, max_length=256):
    """Pooled activations for EVERY requested layer in ONE forward pass per batch.

    The per-layer version above re-encodes the same texts once per probed layer,
    so cost scaled with len(ALT_LAYERS) -- 8x on Llama, 14x on Qwen -- for
    activations that a single forward pass already computes.

    Equivalence: activation_capture hooks the forward OUTPUT of
    model.model.layers[L], and HF's hidden_states[i] is the INPUT to layer i, so
    layer L's output is hidden_states[L+1]. That identity holds for every layer
    except the last, where HF appends the final-norm output instead; ALT_LAYERS
    steps by 2 from 0 and so never reaches it. The assert below enforces that.
    """
    n_layers = model.config.num_hidden_layers
    assert max(layers) < n_layers - 1, (
        f"layer {max(layers)} is the last layer; hidden_states[{n_layers}] is "
        "post-final-norm and does not match the hook. Use pooled_acts for it."
    )
    texts = [t if (t and t.strip()) else "." for t in texts]
    out = {L: [] for L in layers}
    for batch in batched(texts, bs):
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model.device)
        with torch.no_grad():
            res = model(**enc, output_hidden_states=True)
        for L in layers:
            h = res.hidden_states[L + 1].float()
            out[L].append(masked_mean(h, enc["attention_mask"]).cpu())
    return {L: torch.cat(v, 0).numpy() for L, v in out.items()}


def surface_features(tok, texts):
    """Hand-built shallow features that survive paraphrase. No model needed."""
    feats = []
    for t in texts:
        toks = tok(t).input_ids
        words = t.split()
        n_char = max(len(t), 1)
        n_word = max(len(words), 1)
        uniq = len(set(w.lower() for w in words))
        feats.append(
            [
                len(toks),  # token count
                n_char,  # char count
                np.mean([len(w) for w in words]) if words else 0,  # mean word len
                uniq / n_word,  # type-token ratio
                t.count(".") / n_char,  # period rate
                t.count(",") / n_char,  # comma rate
                sum(c.isupper() for c in t) / n_char,  # uppercase rate
                t.count("\n") / n_char,  # newline rate
            ]
        )
    return np.array(feats, dtype=float)


def length_matched_indices(texts_a, texts_b, tok, rng):
    """Return index subsets of a and b with approximately matched token-length
    distributions, by binning and downsampling to the per-bin minimum."""
    la = np.array([len(tok(t).input_ids) for t in texts_a])
    lb = np.array([len(tok(t).input_ids) for t in texts_b])
    bins = np.linspace(0, max(la.max(), lb.max()) + 1, 11)
    ia, ib = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        ai = np.where((la >= lo) & (la < hi))[0]
        bi = np.where((lb >= lo) & (lb < hi))[0]
        k = min(len(ai), len(bi))
        if k == 0:
            continue
        ia.extend(rng.choice(ai, k, replace=False))
        ib.extend(rng.choice(bi, k, replace=False))
    return np.array(ia), np.array(ib)


def gather(method, mult):
    """Concatenate paraphrased steered + unsteered across seeds for one (method,mult).

    Also returns the per-row prompt index, which is the CV group (see cv_auc).
    """
    steered, unsteered, g_steered, g_unsteered = [], [], [], []
    for s in SEEDS:
        steered += load_para(f"{method}_m{mult}", s)
        unsteered += load_para("unsteered", s)
        g_steered += load_para(f"{method}_m{mult}", s, field="idx")
        g_unsteered += load_para("unsteered", s, field="idx")
    return steered, unsteered, g_steered, g_unsteered


def main():
    rng = np.random.default_rng(0)
    model, tok = load_llama(config.GEN_MODEL)
    results = {}

    print(f"STEER_LAYER={config.STEER_LAYER}; alt layers tested: {ALT_LAYERS}\n")
    header = (
        f"{'method':11s} {'mult':4s} {'act@steer':9s} "
        + " ".join(f"act@L{L:<2d}" for L in ALT_LAYERS)
        + f" {'surface':8s} {'len-ctrl':8s}"
    )
    print(header)
    print("-" * len(header))

    for method in METHODS:
        for mult in PROBE_MULTS:
            steered, unsteered, g_s, g_u = gather(method, mult)

            # activation probe at each layer (one forward pass covers all layers)
            acts_s = pooled_acts_all_layers(model, tok, steered, ALT_LAYERS)
            acts_u = pooled_acts_all_layers(model, tok, unsteered, ALT_LAYERS)
            layer_aucs = {L: cv_auc(acts_s[L], acts_u[L], g_s, g_u) for L in ALT_LAYERS}

            # surface-feature probe
            Fs = surface_features(tok, steered)
            Fu = surface_features(tok, unsteered)
            surf_auc = cv_auc(Fs, Fu, g_s, g_u)

            # length-controlled activation probe at the steered layer
            ia, ib = length_matched_indices(steered, unsteered, tok, rng)
            if len(ia) > 30:
                Xs_lc = pooled_acts(model, tok, [steered[i] for i in ia], config.STEER_LAYER)
                Xu_lc = pooled_acts(model, tok, [unsteered[i] for i in ib], config.STEER_LAYER)
                lc_auc = cv_auc(Xs_lc, Xu_lc, [g_s[i] for i in ia], [g_u[i] for i in ib])
            else:
                lc_auc = float("nan")

            steer_auc = layer_aucs[config.STEER_LAYER]
            row = (
                f"{method:11s} {mult:4s} {steer_auc:9.3f} "
                + " ".join(f"{layer_aucs[L]:7.3f}" for L in ALT_LAYERS)
                + f" {surf_auc:8.3f} {lc_auc:8.3f}"
            )
            print(row)

            results[f"{method}_m{mult}"] = {
                "act_by_layer": layer_aucs,
                "surface_auc": surf_auc,
                "length_controlled_auc": lc_auc,
                "n_length_matched": int(len(ia)),
            }

    # ---- interpretation ----
    print("\n" + "=" * 64)
    print("INTERPRETATION")
    print("=" * 64)
    low_mult = PROBE_MULTS[0]  # the barely-steered point
    print(f"\nFor the low-steering points (m{low_mult}):")
    layer_inv_stats = {}
    for method in METHODS:
        r = results.get(f"{method}_m{low_mult}")
        if not r:
            continue
        layer_aucs = r["act_by_layer"]
        steer = layer_aucs[config.STEER_LAYER]
        surf = r["surface_auc"]
        lc = r["length_controlled_auc"]
        vals = list(layer_aucs.values())
        rng = max(vals) - min(vals)
        std = float(np.std(vals))
        # is the steered layer the peak?
        peak_layer = max(layer_aucs, key=lambda k: layer_aucs[k])
        steered_is_peak = peak_layer == config.STEER_LAYER
        layer_inv_stats[method] = {
            "range": rng,
            "std": std,
            "steered_is_peak": steered_is_peak,
            "peak_layer": peak_layer,
        }
        flags = []
        if surf >= steer - 0.05:
            flags.append("SURFACE-EXPLAINED")
        if not np.isnan(lc) and lc < steer - 0.10:
            flags.append("LENGTH-CONFOUNDED")
        if rng < 0.07 and not steered_is_peak:
            flags.append("LAYER-INVARIANT (flat across depth, steered layer not the peak)")
        verdict = "; ".join(flags) if flags else "no shallow artifact -> may be genuine"
        print(f"  {method:11s}: act@steer={steer:.3f} surface={surf:.3f} len-ctrl={lc:.3f}")
        print(
            f"               layers: range={rng:.3f} std={std:.3f} "
            f"peak@L{peak_layer} steered@L{config.STEER_LAYER} "
            f"({'steered=peak' if steered_is_peak else 'steered NOT peak'})"
        )
        print(f"               -> {verdict}")

    # citable summary for the paper
    print("\n--- LAYER-INVARIANCE SUMMARY (cite these) ---")
    ranges = [s["range"] for s in layer_inv_stats.values()]
    n_steered_peak = sum(s["steered_is_peak"] for s in layer_inv_stats.values())
    print(f"  layers probed: {len(ALT_LAYERS)} ({ALT_LAYERS[0]}..{ALT_LAYERS[-1]})")
    print(f"  max AUC range across layers (any method): {max(ranges):.3f}")
    print(f"  steered layer is the peak in {n_steered_peak}/{len(layer_inv_stats)} methods")
    print("  -> small range + steered layer rarely the peak = global, not localized")

    results["_layer_invariance_summary"] = {
        "n_layers_probed": len(ALT_LAYERS),
        "layers": list(ALT_LAYERS),
        "max_range": float(max(ranges)),
        "n_methods_steered_is_peak": int(n_steered_peak),
        "n_methods": len(layer_inv_stats),
        "per_method": layer_inv_stats,
    }
    with open(SWEEP_DIR / "artifact_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {SWEEP_DIR / 'artifact_diagnostic.json'}")


if __name__ == "__main__":
    main()
