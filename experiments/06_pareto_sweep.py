"""Pareto sweep: post-paraphrase detectability vs generation-time fluency cost.

This is the core experiment for the negative-results paper. It answers the
objection "you only tried one operating point" by sweeping the watermark
strength of every method and plotting the whole detection-vs-fluency frontier.

Methods compared at matched fluency:
  - one curve per SEMANTIC direction in directions.pt (e.g. formality)
  - one curve for the RANDOM orthogonal control direction
  - one curve for Kirchenbauer (KGW), sweeping the logit bias

Axes of the headline figure:
  x = pre-paraphrase perplexity ratio vs unsteered  (the watermark's fluency tax)
  y = post-paraphrase probe/detector AUC            (robustness to paraphrase)

The thesis "no advantage at matched fluency" is credible only if the curves
overlap across the whole frontier, not at one cherry-picked point.

Mechanistic claim ("detects perturbation, not semantics"): pool all
direction-based operating points (every semantic direction AND random, across
all multipliers) and regress post-paraphrase AUC on perplexity-delta. If AUC is
well-predicted by perplexity-delta alone and the semantic and random points lie
on the SAME line, detection is a function of distortion, not of what the
direction means.

Phases (each resumable; re-running skips completed operating points):
  1. generate    -> artifacts/sweep/gen_{tag}.jsonl
  2. paraphrase  -> artifacts/sweep/para_{tag}.jsonl
  3. evaluate    -> artifacts/sweep/sweep_results.csv + two PDFs + correlation

Run:  python 06_pareto_sweep.py
Configurable grid + sample count near the top of this file.
"""

import csv
import json
import math

import numpy as np
import torch
from tqdm import tqdm
from transformers import WatermarkingConfig, WatermarkDetector

import config
from utils import (
    activation_capture,
    batch_chat_generate,
    batched,
    load_llama,
    masked_mean,
    steering_hook,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------- sweep grid -----------------------------------
# Fewer prompts than the main run: 200 gives AUC standard error ~0.03, which is
# plenty to resolve curve *shapes*. Bump to 400 for camera-ready numbers.
N_SWEEP = 200
GEN_BATCH = 16

# Multiple generation seeds. Within a seed, the RNG is reset to an IDENTICAL
# state before every operating point, so the only thing varying across
# interventions at a given seed is the intervention itself (this is what fixes
# the non-monotone perplexity: cross-point sampling noise is removed). Across
# seeds we get error bars on both axes. 3 is a reasonable rigor/cost tradeoff;
# drop to 2 if wall-clock is tight, raise to 5 for camera-ready.
SEEDS = [0, 1, 2]

# Steering multipliers swept for EVERY direction (semantic + random control).
# Norm-relative fractions, set PER-MODEL in config.MODEL_REGISTRY (each model's
# degradation point differs). Legitimacy: figures plot AUC against measured
# perplexity, not the fraction, so per-model grids remain comparable in fluency
# space. Confirm/adjust a model's grid with 10_steer_calibration.
MULTIPLIERS = config.STEER_FRACTIONS

# KGW logit-bias values swept (gamma fixed at config.KGW_GAMMA).
KGW_BIASES = [1.0, 2.0, 3.0, 4.0]

# Which paraphrase attack to evaluate against. None = the default Pegasus files;
# set by --attack to score an alternative attacker (e.g. DIPPER) without
# touching any existing result. Output paths below are re-derived in main().
ATTACK = None

SWEEP_DIR = config.ART / "sweep"
SWEEP_DIR.mkdir(exist_ok=True)
# Per-sample detector scores, kept so 12_gap_test.py can bootstrap a CI on the
# KGW-vs-steering difference without re-encoding anything.
SCORES_DIR = SWEEP_DIR / "scores"
SCORES_DIR.mkdir(exist_ok=True)
CSV_PATH = SWEEP_DIR / "sweep_results.csv"
PARETO_PDF = SWEEP_DIR / "pareto_detection_vs_fluency.pdf"
DISTORT_PDF = SWEEP_DIR / "distortion_correlation.pdf"
# ----------------------------------------------------------------------------


def load_prompts(n):
    # These are the exact prompts recovered from both models' saved runs.
    # Loading them avoids depending on a mutable remote dataset ordering.
    with open(config.PROMPTS_PATH, encoding="utf-8") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise ValueError("The prompts file must contain a JSON list of strings")
    if len(prompts) < n:
        raise ValueError(f"requested {n} prompts, but only {len(prompts)} are available")
    return prompts[:n]


def gen_kwargs():
    return dict(
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=True,
        temperature=config.GEN_TEMPERATURE,
        top_p=config.GEN_TOP_P,
        pad_token_id=None,  # filled in after tok load
    )


def reseed(seed):
    """Reset all RNGs to an identical state. Called before EVERY operating point
    within a seed so that sampling randomness is matched across interventions —
    the difference between unsteered and steered generations at the same seed is
    then attributable to the intervention, not to different sampling draws."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- operating points -----------------------------
def build_operating_points(directions):
    """Return list of dicts describing every operating point to generate.

    Each point: {tag, method, kind, param, direction(optional)}
      kind in {"unsteered", "semantic", "random", "kgw"}
    """
    points = [{"tag": "unsteered", "method": "unsteered", "kind": "unsteered", "param": 0.0}]

    for name, vec in directions.items():
        kind = "random" if name == "random_ortho" else "semantic"
        label = "random" if name == "random_ortho" else name
        for m in MULTIPLIERS:
            points.append(
                {
                    "tag": f"{label}_m{m}",
                    "method": label,
                    "kind": kind,
                    "param": m,
                    "direction": vec,
                }
            )

    for b in KGW_BIASES:
        points.append({"tag": f"kgw_b{b}", "method": "kgw", "kind": "kgw", "param": b})
    return points


# ----------------------------- phase 1: generate ----------------------------
def phase_generate(points, model, tok, prompts):
    base_kwargs = gen_kwargs()
    base_kwargs["pad_token_id"] = tok.eos_token_id

    for seed in SEEDS:
        for pt in points:
            out_path = SWEEP_DIR / f"gen_{pt['tag']}_s{seed}.jsonl"
            if out_path.exists():
                print(f"[gen] skip {pt['tag']} seed={seed} (exists)")
                continue
            print(f"[gen] {pt['tag']} seed={seed} (kind={pt['kind']}, param={pt['param']})")

            kwargs = dict(base_kwargs)
            if pt["kind"] == "kgw":
                kwargs["watermarking_config"] = WatermarkingConfig(
                    greenlist_ratio=config.KGW_GAMMA,
                    bias=pt["param"],
                    hashing_key=config.KGW_SEED,
                    seeding_scheme=config.KGW_SCHEME,
                    context_width=1,
                )

            # Identical RNG state before every point at this seed -> matched
            # sampling noise across interventions.
            reseed(seed)
            if pt["kind"] in ("semantic", "random"):
                with steering_hook(model, config.STEER_LAYER, pt["direction"], pt["param"]):
                    comps = batch_chat_generate(model, tok, prompts, kwargs, GEN_BATCH)
            else:
                comps = batch_chat_generate(model, tok, prompts, kwargs, GEN_BATCH)

            with open(out_path, "w") as f:
                for i, c in enumerate(comps):
                    f.write(json.dumps({"idx": i, "prompt": prompts[i], "completion": c}) + "\n")


# ----------------------------- phase 2: paraphrase --------------------------
def phase_paraphrase(points):
    import re
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer

    # one (point, seed) unit per generated file
    units = [(pt, s) for s in SEEDS for pt in points]
    todo = [
        (pt, s) for (pt, s) in units if not (SWEEP_DIR / f"para_{pt['tag']}_s{s}.jsonl").exists()
    ]
    if not todo:
        print("[para] all (point, seed) units already paraphrased; skipping load")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[para] loading {config.PARAPHRASE_MODEL}")
    ptok = PegasusTokenizer.from_pretrained(config.PARAPHRASE_MODEL)
    pmodel = PegasusForConditionalGeneration.from_pretrained(config.PARAPHRASE_MODEL).to(device)
    pmodel.eval()
    sent_split = re.compile(r"(?<=[.!?])\s+")

    def para_text(text, ptok, pmodel):
        sents = [s.strip() for s in sent_split.split(text.strip()) if s.strip()]
        if not sents:
            return text
        out = []
        for i in range(0, len(sents), 16):
            chunk = sents[i : i + 16]
            enc = ptok(
                chunk, truncation=True, padding="longest", max_length=60, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                g = pmodel.generate(
                    **enc, max_length=60, num_beams=5, num_return_sequences=1, temperature=1.5
                )
            out.extend(ptok.batch_decode(g, skip_special_tokens=True))
        return " ".join(out)

    for pt, s in todo:
        gen_path = SWEEP_DIR / f"gen_{pt['tag']}_s{s}.jsonl"
        out_path = SWEEP_DIR / f"para_{pt['tag']}_s{s}.jsonl"
        with open(gen_path) as f:
            recs = [json.loads(l) for l in f]
        print(f"[para] {pt['tag']} seed={s} ({len(recs)} texts)")
        with open(out_path, "w") as f:
            for r in tqdm(recs, desc=pt["tag"]):
                f.write(
                    json.dumps({**r, "paraphrased": para_text(r["completion"], ptok, pmodel)})
                    + "\n"
                )

    del pmodel, ptok
    torch.cuda.empty_cache()


# ----------------------------- phase 3: evaluate ----------------------------
def _safe(texts):
    """Replace empty / whitespace-only strings with a single placeholder token.

    Qwen3's attention reshape crashes on a zero-length sequence (the
    'cannot reshape tensor of 0 elements into [B,0,-1,128]' error). Empty
    generations can occur at high steering fractions (text degrades to nothing
    after skip_special_tokens) or from the paraphraser. We must NOT drop them --
    that would desync steered/unsteered row counts and the probe labels -- so we
    map them to a constant benign token. They become a tiny constant signal in
    BOTH classes and wash out of the contrast.
    """
    return [t if (t and t.strip()) else "." for t in texts]


def pooled_acts(model, tok, texts, layer, bs=8, max_length=256):
    texts = _safe(texts)
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


def mean_ppl(model, tok, texts, max_length=256, sample=150):
    vals = []
    for t in texts[:sample]:
        if len(t.strip()) < 5:
            continue
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        vals.append(float(torch.exp(out.loss).item()))
    return float(np.mean(vals))


def cluster_bootstrap_std(y, scores, groups, seed, n_boot=200):
    """Std of the AUC under a bootstrap that resamples PROMPTS, not rows.

    Rows are not independent: each prompt contributes one row per generation seed
    in each class. Resampling rows treats those as independent draws and
    understates the error. Resampling whole prompts respects the design.

    Used for the probe AND for KGW so `auc_post_std` means the same thing on
    every row of the CSV (it feeds the error bars in the frontier figures).
    For the probe these are out-of-fold scores, so this captures sampling
    variability but not variability from refitting the probe.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return 0.0
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], scores[idx]))
    return float(np.std(boots)) if boots else 0.0


def probe_auc(Xpos, Xneg, seed, groups_pos=None, groups_neg=None):
    """Pooled cross-validated AUC for steered (pos) vs unsteered (neg).

    All seeds' samples are pooled into a single CV here (see phase_evaluate),
    which is the correct estimator: the earlier per-seed AUC-then-average
    overfit on ~200 samples per seed and inflated the low-multiplier points
    (the spurious m1 ~0.86 spike).

    GROUPING (added after the leakage audit -- this is the important part).
    Pooling alone did not fix that spike, because the real cause is not sample
    size but the repeated-prompt design. The pooled set contains the SAME prompt
    three times per class (once per generation seed), and `reseed` matches the
    sampling RNG across interventions, so at low steering a prompt's steered and
    unsteered completions are near-duplicates. Plain StratifiedKFold scattered
    those near-duplicates across train and test, letting the probe key on prompt
    identity rather than on the perturbation. That inflates precisely the
    low-multiplier points and is what produced the non-monotone AUC-vs-alpha
    curve (more steering appearing LESS detectable, which is not physical).

    StratifiedGroupKFold with the prompt index as the group keeps every row of a
    prompt -- both classes, all seeds -- inside one fold, so nothing the probe
    saw in training reappears at test time. KGW needs no such fix: its detector
    is an unfitted z-score test and cannot leak.

    Standardized + regularized (C=0.5), 5-fold out-of-fold scoring.

    Returns (auc, auc_symmetric, boot_std, detail) where `detail` carries the
    per-sample out-of-fold scores for 12_gap_test.py.
    """
    X = np.concatenate([Xpos, Xneg], 0)
    y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5))

    if groups_pos is None or groups_neg is None:
        groups = np.full(len(y), -1)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_groups = None
    else:
        groups = np.concatenate([np.asarray(groups_pos), np.asarray(groups_neg)])
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_groups = groups

    oof = cross_val_predict(clf, X, y, groups=cv_groups, cv=skf, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, oof))
    boot_std = cluster_bootstrap_std(y, oof, groups, seed)
    detail = {"score": oof, "y": y, "group": groups}
    return auc, max(auc, 1.0 - auc), boot_std, detail


def kgw_auc(pos_texts, neg_texts, model, tok, bias, seed, groups_pos=None, groups_neg=None):
    cfg = WatermarkingConfig(
        greenlist_ratio=config.KGW_GAMMA,
        bias=bias,
        hashing_key=config.KGW_SEED,
        seeding_scheme=config.KGW_SCHEME,
        context_width=1,
    )
    det = WatermarkDetector(model_config=model.config, device=model.device, watermarking_config=cfg)

    def zs(texts):
        out = []
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids.to(
                model.device
            )
            if ids.shape[1] < 4:
                out.append(0.0)
                continue
            with torch.no_grad():
                r = det(ids, return_dict=True)
            out.append(float(np.asarray(r.z_score).reshape(-1)[0]))
        return out

    pz, nz = zs(pos_texts), zs(neg_texts)
    scores = np.concatenate([pz, nz])
    y = np.concatenate([np.ones(len(pz)), np.zeros(len(nz))])
    auc = float(roc_auc_score(y, scores))

    if groups_pos is None or groups_neg is None:
        groups = np.full(len(y), -1)
    else:
        groups = np.concatenate([np.asarray(groups_pos), np.asarray(groups_neg)])
    # Same cluster bootstrap as the probe, so the two error bars are comparable.
    boot_std = cluster_bootstrap_std(y, scores, groups, seed)
    detail = {"score": scores, "y": y, "group": groups}
    return auc, max(auc, 1.0 - auc), boot_std, detail


def para_path(tag, seed):
    """Paraphrase file for the active attack.

    ATTACK is None for the default Pegasus run (para_<tag>_s<seed>.jsonl, the
    layout every earlier result used) and a name like "dipper" otherwise
    (para_dipper_<tag>_s<seed>.jsonl, written by 15_dipper_attack.py). Results
    for each attack go to their own CSV and scores directory, so adding an
    attack never overwrites an existing one.
    """
    stem = f"para_{tag}_s{seed}" if ATTACK is None else f"para_{ATTACK}_{tag}_s{seed}"
    return SWEEP_DIR / f"{stem}.jsonl"


def load_field(tag, seed, field):
    with open(para_path(tag, seed)) as f:
        return [json.loads(l)[field] for l in f]


def load_all_seeds(tag, field):
    """Concatenate one field across all seeds (used for pooled-CV AUC)."""
    out = []
    for s in SEEDS:
        out += load_field(tag, s, field)
    return out


def phase_evaluate(points, model, tok):
    """Two estimators, deliberately different:

    - PERPLEXITY (x-axis): computed per seed (steered_s / unsteered_s) and
      aggregated to mean +/- std. Within-seed matching is the right fluency
      reference and yields honest x error bars.

    - AUC (y-axis): all seeds POOLED into a single cross-validation. Averaging
      per-seed AUCs (the previous approach) overfit on ~200 samples/seed and
      inflated the low-multiplier points. Pooling (~600/class) is the correct
      low-variance estimator; per-fold dispersion gives the y error bar.
    """
    # --- per-seed unsteered perplexity references (for the x-axis) ---
    print("Precomputing per-seed unsteered perplexity...")
    uns_ppl = {s: mean_ppl(model, tok, load_field("unsteered", s, "completion")) for s in SEEDS}

    # --- pooled unsteered references (for the y-axis AUC) ---
    print("Precomputing pooled unsteered activations/texts...")
    uns_para_all = load_all_seeds("unsteered", "paraphrased")
    uns_orig_all = load_all_seeds("unsteered", "completion")
    # Prompt index per row -> the CV group. Both classes are built from the same
    # 200 prompts in the same seed order, so a prompt's steered and unsteered
    # rows share a group and can never straddle a fold boundary.
    uns_groups = load_all_seeds("unsteered", "idx")
    uns_acts_para = pooled_acts(model, tok, uns_para_all, config.STEER_LAYER)
    uns_acts_orig = pooled_acts(model, tok, uns_orig_all, config.STEER_LAYER)

    rows = []
    for pt in points:
        if pt["kind"] == "unsteered":
            rows.append(
                {
                    "tag": pt["tag"],
                    "method": "unsteered",
                    "kind": "unsteered",
                    "param": 0.0,
                    "n_seeds": len(SEEDS),
                    "ppl_ratio_mean": 1.0,
                    "ppl_ratio_std": 0.0,
                    "auc_pre": 0.5,
                    "auc_pre_sym": 0.5,
                    "auc_pre_std": 0.0,
                    "auc_post": 0.5,
                    "auc_post_sym": 0.5,
                    "auc_post_std": 0.0,
                }
            )
            continue

        # x-axis: per-seed perplexity ratio
        ppl_ratios = []
        for s in SEEDS:
            ppl_s = mean_ppl(model, tok, load_field(pt["tag"], s, "completion"))
            ppl_ratios.append(ppl_s / uns_ppl[s])
        ppl_mean, ppl_std = float(np.mean(ppl_ratios)), float(np.std(ppl_ratios))

        # y-axis: pooled-across-seeds AUC
        para_all = load_all_seeds(pt["tag"], "paraphrased")
        orig_all = load_all_seeds(pt["tag"], "completion")
        grp_all = load_all_seeds(pt["tag"], "idx")

        if pt["kind"] == "kgw":
            auc_post, auc_post_sym, post_std, det_post = kgw_auc(
                para_all, uns_para_all, model, tok, pt["param"], config.SEED, grp_all, uns_groups
            )
            auc_pre, auc_pre_sym, _, _ = kgw_auc(
                orig_all, uns_orig_all, model, tok, pt["param"], config.SEED, grp_all, uns_groups
            )
        else:
            acts_para = pooled_acts(model, tok, para_all, config.STEER_LAYER)
            auc_post, auc_post_sym, post_std, det_post = probe_auc(
                acts_para, uns_acts_para, config.SEED, grp_all, uns_groups
            )
            acts_orig = pooled_acts(model, tok, orig_all, config.STEER_LAYER)
            auc_pre, auc_pre_sym, _, _ = probe_auc(
                acts_orig, uns_acts_orig, config.SEED, grp_all, uns_groups
            )

        np.savez(SCORES_DIR / f"scores_{pt['tag']}.npz", **det_post)

        rows.append(
            {
                "tag": pt["tag"],
                "method": pt["method"],
                "kind": pt["kind"],
                "param": pt["param"],
                "n_seeds": len(SEEDS),
                "ppl_ratio_mean": ppl_mean,
                "ppl_ratio_std": ppl_std,
                "auc_pre": auc_pre,
                "auc_pre_sym": auc_pre_sym,
                "auc_pre_std": 0.0,
                "auc_post": auc_post,
                "auc_post_sym": auc_post_sym,
                "auc_post_std": post_std,
            }
        )
        print(
            f"  {pt['tag']:16s} ppl={ppl_mean:.2f}±{ppl_std:.2f}x "
            f"auc_post={auc_post:.3f}±{post_std:.3f} (sym {auc_post_sym:.3f})"
        )

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {CSV_PATH}")
    return rows


# ----------------------------- plotting + stats -----------------------------
def make_plots(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = {}
    for r in rows:
        if r["kind"] == "unsteered":
            continue
        methods.setdefault(r["method"], []).append(r)

    # ---- Figure 1: Pareto frontier with error bars on both axes ----
    # Steering perplexity blows up at high fractions (can exceed 80x), which
    # crushes the informative low-ppl region on a linear axis. Use log-x so the
    # whole frontier is legible, and mark the "deployable" region (ppl <= 1.5x).
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    markers = {"semantic": "o", "random": "s", "kgw": "^"}
    for method, pts in sorted(methods.items()):
        pts = sorted(pts, key=lambda r: r["ppl_ratio_mean"])
        x = [p["ppl_ratio_mean"] for p in pts]
        y = [p["auc_post"] for p in pts]
        xerr = [p["ppl_ratio_std"] for p in pts]
        yerr = [p["auc_post_std"] for p in pts]
        kind = pts[0]["kind"]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            marker=markers.get(kind, "o"),
            label=method,
            linewidth=1.5,
            markersize=6,
            capsize=2,
            elinewidth=0.8,
        )
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.axvspan(1.0, 1.5, color="green", alpha=0.06, label="deployable (≤1.5× ppl)")
    ax.set_xscale("log")
    ax.set_xlabel("Generation-time perplexity ratio (vs unsteered, log scale)")
    ax.set_ylabel("Post-paraphrase detection AUC")
    ax.set_title(
        "Detection vs fluency frontier\n(steering only matches KGW by inflating perplexity)"
    )
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(PARETO_PDF)
    fig.savefig(str(PARETO_PDF).replace(".pdf", ".png"), dpi=150)
    print(f"Wrote {PARETO_PDF}")

    # ---- Figure 1b: distortion-based frontier (robust to ppl<1) ----
    # Some models (e.g. Qwen) produce steered text scoring MORE fluent than
    # unsteered (ppl_ratio<1) via homogenization, which makes a raw-ppl x-axis
    # misleading ("free" detection at ppl<1). Plot two-sided distortion
    # d = |log(ppl_ratio)| so displacement in EITHER direction counts as moving
    # away from the base distribution. NaN/degenerate points are skipped.
    figd, axd = plt.subplots(figsize=(5.6, 4.0))
    for method, pts in sorted(methods.items()):
        pp = [
            (p, p["ppl_ratio_mean"])
            for p in pts
            if p["ppl_ratio_mean"] == p["ppl_ratio_mean"] and p["ppl_ratio_mean"] > 0
        ]
        pp = sorted(pp, key=lambda z: abs(math.log(z[1])))
        x = [abs(math.log(v)) for _, v in pp]
        y = [p["auc_post"] for p, _ in pp]
        yerr = [p["auc_post_std"] for p, _ in pp]
        kind = pp[0][0]["kind"] if pp else "semantic"
        axd.errorbar(
            x,
            y,
            yerr=yerr,
            marker=markers.get(kind, "o"),
            label=method,
            linewidth=1.5,
            markersize=6,
            capsize=2,
            elinewidth=0.8,
        )
    axd.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    axd.axvspan(0.0, 0.10, color="green", alpha=0.06, label="low distortion (|log ppl|≤0.1)")
    axd.set_xlabel("Distortion  |log(perplexity ratio)|")
    axd.set_ylabel("Post-paraphrase detection AUC")
    axd.set_title("Detection vs distortion\n(two-sided; robust to ppl<1)")
    axd.legend(fontsize=7.5, loc="lower right")
    axd.grid(alpha=0.3)
    figd.tight_layout()
    dist_pdf = str(PARETO_PDF).replace("pareto_detection_vs_fluency", "pareto_distortion")
    figd.savefig(dist_pdf)
    figd.savefig(dist_pdf.replace(".pdf", ".png"), dpi=150)
    print(f"Wrote {dist_pdf}")

    # ---- Figure 2: distortion correlation (mechanistic claim) ----
    # Pool all direction-based points (semantic + random); regress detectability
    # on DISTORTION. Use SYMMETRIC AUC here: the claim is about magnitude of
    # separability vs distortion, and a verifier knows the watermark sign.
    #
    # Scale matters, and the earlier version got it wrong. Regressing on the raw
    # delta (ppl_ratio - 1) put points spanning 1x to ~87x on one linear axis, so
    # a couple of degenerate blowups dominated the fit and R^2 collapsed to ~0.15
    # -- an artifact of the scale, not evidence against the distortion account.
    # We now fit on d = |log(ppl_ratio)|, the same two-sided axis the frontier
    # figures use, over non-degenerate points only (the ppl > PPL_MAX rows are
    # destroyed text, not operating points, and are excluded from every figure).
    PPL_MAX = 10.0
    dir_rows = [
        r
        for r in rows
        if r["kind"] in ("semantic", "random")
        and r["ppl_ratio_mean"] == r["ppl_ratio_mean"]  # not NaN
        and 0 < r["ppl_ratio_mean"] <= PPL_MAX
    ]
    n_excluded = sum(1 for r in rows if r["kind"] in ("semantic", "random")) - len(dir_rows)
    dist = np.array([abs(math.log(r["ppl_ratio_mean"])) for r in dir_rows])
    auc = np.array([r["auc_post_sym"] for r in dir_rows])

    fig2, ax2 = plt.subplots(figsize=(5.2, 4.0))
    for kind, color in [("semantic", "C0"), ("random", "C1")]:
        xs = [abs(math.log(r["ppl_ratio_mean"])) for r in dir_rows if r["kind"] == kind]
        ys = [r["auc_post_sym"] for r in dir_rows if r["kind"] == kind]
        ax2.scatter(xs, ys, label=kind, c=color, s=40)

    r2 = None
    if len(dist) >= 3 and np.ptp(dist) > 0:
        b1, b0 = np.polyfit(dist, auc, 1)
        xfit = np.linspace(dist.min(), dist.max(), 50)
        ax2.plot(xfit, b1 * xfit + b0, "k--", lw=1.2, label=f"fit: AUC={b1:.2f}·d+{b0:.2f}")
        pred = b1 * dist + b0
        ss_res = np.sum((auc - pred) ** 2)
        ss_tot = np.sum((auc - auc.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ax2.set_xlabel("Distortion  |log(perplexity ratio)|")
    ax2.set_ylabel("Post-paraphrase detection AUC (symmetric)")
    if n_excluded:
        print(
            f"  [distortion fit] excluded {n_excluded} degenerate point(s) "
            f"(NaN or ppl > {PPL_MAX}x)"
        )
    title = "Detection is a function of distortion, not semantics"
    if r2 is not None:
        title += f"  (R²={r2:.2f})"
    ax2.set_title(title, fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(DISTORT_PDF)
    fig2.savefig(str(DISTORT_PDF).replace(".pdf", ".png"), dpi=150)
    print(f"Wrote {DISTORT_PDF}")
    return r2


def interpret(rows, r2):
    print("\n" + "=" * 64)
    print("PARETO SWEEP — INTERPRETATION")
    print("=" * 64)

    # Compare best semantic vs random vs kgw at matched DISTORTION.
    #
    # The earlier version selected each method's point nearest to 1.2x raw
    # perplexity. That is not a matched comparison on any model whose ppl ratio
    # dips below 1: on Qwen the ratios run 0.72-1.46, so "nearest to 1.2x" pulled
    # a d=0.18 semantic point against a d=0.03 random one and then declared the
    # semantic direction ahead. Selecting on d = |log(ppl_ratio)| compares points
    # that are actually displaced from the base distribution by the same amount.
    PPL_MAX = 10.0
    TARGET_D = 0.18  # ~ 1.2x perplexity in either direction

    def usable(r):
        p = r["ppl_ratio_mean"]
        return r["kind"] != "unsteered" and p == p and 0 < p <= PPL_MAX

    def auc_at(method_kind, target=TARGET_D):
        cand = [r for r in rows if r["kind"] == method_kind and usable(r)]
        if not cand:
            return None
        return min(cand, key=lambda r: abs(abs(math.log(r["ppl_ratio_mean"])) - target))

    sem = auc_at("semantic")
    rnd = auc_at("random")
    kgw = auc_at("kgw")
    print(f"\nNearest operating point to distortion d={TARGET_D} (symmetric AUC):")
    for tag, r in [("semantic", sem), ("random", rnd), ("kgw", kgw)]:
        if r:
            d = abs(math.log(r["ppl_ratio_mean"]))
            print(
                f"  {tag:9s}: ppl={r['ppl_ratio_mean']:.2f}±{r['ppl_ratio_std']:.2f}x  "
                f"d={d:.3f}  auc={r['auc_post_sym']:.3f}±{r['auc_post_std']:.3f}  "
                f"({r['tag']})"
            )

    if r2 is not None:
        print(
            f"\nDistortion-correlation R² = {r2:.3f}  "
            f"(fit on d=|log ppl|, degenerate points excluded)"
        )
        if r2 > 0.6:
            print("  -> Detectability is largely a function of distortion; semantic")
            print("     and random points share the trend. Supports 'detects perturbation,")
            print("     not semantics'.")
        else:
            print("  -> AUC is NOT a clean function of distortion. The simple")
            print("     'distortion detector' framing is not supported; the layer-invariance")
            print("     result (07_artifact_diagnostic) is the stronger mechanistic claim.")

    if sem and rnd and kgw:
        s, rr, k = sem["auc_post_sym"], rnd["auc_post_sym"], kgw["auc_post_sym"]
        ds = [abs(math.log(r["ppl_ratio_mean"])) for r in (sem, rnd, kgw)]
        print(
            f"\n  semantic={s:.3f}  random={rr:.3f}  kgw={k:.3f}  "
            f"(d = {ds[0]:.2f}/{ds[1]:.2f}/{ds[2]:.2f})"
        )
        if max(ds) - min(ds) > 0.10:
            print("  NOTE: these three points are not well matched in distortion")
            print("  (spread > 0.10); read the frontier figure instead of this line.")
        if s <= rr + 0.03 and s <= k + 0.03:
            print("  VERDICT: at matched distortion the semantic watermark does NOT beat")
            print("  random or KGW. Negative result holds.")
        else:
            print("  VERDICT: semantic shows an edge here — inspect the full curve.")
    print("=" * 64)


def parse_args():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--attack",
        default=None,
        help="score against para_<attack>_<tag>_s<seed>.jsonl instead "
        "of the default Pegasus files (e.g. 'dipper', written by "
        "15_dipper_attack.py). Results go to their own CSV and "
        "scores directory; existing results are never overwritten.",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="skip generation and paraphrasing; only re-run phase 3. "
        "Implied by --attack, whose paraphrases are produced "
        "elsewhere.",
    )
    return ap.parse_args()


def apply_attack(attack):
    """Point the module's output paths at this attack's namespace."""
    global ATTACK, CSV_PATH, SCORES_DIR, PARETO_PDF, DISTORT_PDF
    ATTACK = attack
    suffix = "" if attack is None else f"_{attack}"
    CSV_PATH = SWEEP_DIR / f"sweep_results{suffix}.csv"
    SCORES_DIR = SWEEP_DIR / f"scores{suffix}"
    SCORES_DIR.mkdir(exist_ok=True)
    PARETO_PDF = SWEEP_DIR / f"pareto_detection_vs_fluency{suffix}.pdf"
    DISTORT_PDF = SWEEP_DIR / f"distortion_correlation{suffix}.pdf"


def main():
    args = parse_args()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    apply_attack(args.attack)
    eval_only = args.eval_only or args.attack is not None

    directions = torch.load(config.DIR_PATH)
    directions = {k: v for k, v in directions.items() if torch.is_tensor(v)}
    print(f"Directions found: {list(directions.keys())}")
    if args.attack:
        print(f"Attack: {args.attack}  ->  {CSV_PATH.name}")

    prompts = load_prompts(N_SWEEP)
    model, tok = load_llama(config.GEN_MODEL)
    points = build_operating_points(directions)
    print(f"{len(points)} operating points to sweep.")

    if eval_only:
        missing = [
            pt["tag"] for pt in points for s in SEEDS if not para_path(pt["tag"], s).exists()
        ]
        if missing:
            raise SystemExit(
                f"missing paraphrases for {len(missing)} (tag, seed) units, e.g. "
                f"{para_path(missing[0], SEEDS[0]).name}. Run the attack script "
                f"first, or restrict the grid to the tags it covered."
            )
    else:
        phase_generate(points, model, tok, prompts)
        phase_paraphrase(points)

    rows = phase_evaluate(points, model, tok)
    r2 = make_plots(rows)
    interpret(rows, r2)


if __name__ == "__main__":
    main()
