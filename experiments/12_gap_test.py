"""Confidence interval on the headline KGW-vs-steering gap.

Answers Reviewer 3, Question 1 ("The headline gap is statistically reliable...
Did the authors run a statistical test?"). Previously: no. This is that test.

Why a plain error-bar comparison is not enough
----------------------------------------------
KGW and the activation probe score the SAME unsteered negative set, so their
AUCs are positively correlated. Comparing two independent per-method error bars
overstates the uncertainty of the DIFFERENCE. The fix is to resample once and
recompute both AUCs on the same replicate, which preserves the correlation.

Design
------
Cluster bootstrap over PROMPTS. Each prompt contributes one row per generation
seed to both classes, so rows are not independent; resampling whole prompts
respects that. For each replicate we draw len(prompts) prompts with replacement,
gather every row belonging to them, and recompute both AUCs.

Symmetric AUC (the paper's stated metric) is max(AUC, 1-AUC). Taking that max
inside each replicate would bias the estimate upward, so orientation is fixed
ONCE on the full sample and the oriented AUC is used throughout.

Caveat to state in the paper: the probe's scores are out-of-fold predictions, so
this captures sampling variability but not variability from refitting the probe.
It is a CI on the gap given the fitted detector, which is the relevant quantity
for "does KGW beat steering on this data".

Reads artifacts/<model>/sweep/scores/scores_<tag>.npz, written by
06_pareto_sweep.py phase 3. CPU-only, runs in seconds, re-encodes nothing.

Usage:
  # auto-pick the best point per method inside the deployable band
  python 12_gap_test.py --sweep artifacts/llama-1b/sweep

  # or name the two operating points explicitly
  python 12_gap_test.py --sweep artifacts/llama-1b/sweep --a kgw_b3.0 --b sentiment_m0.1
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOT = 10000
DEPLOY_PPL = 1.15  # the paper's "deployable regime" threshold


def suffix_for(attack):
    return "" if attack is None else f"_{attack}"


def load_scores(sweep_dir, tag, attack=None):
    p = Path(sweep_dir) / f"scores{suffix_for(attack)}" / f"scores_{tag}.npz"
    if not p.exists():
        raise SystemExit(
            f"missing {p}\nRun 06_pareto_sweep.py (phase 3) after the "
            "StratifiedGroupKFold patch to write per-sample scores"
            + (f", with --attack {attack}." if attack else ".")
        )
    d = np.load(p)
    return d["y"], d["score"], d["group"]


def oriented(y, score):
    """Return scores flipped so AUC >= 0.5, plus the orientation actually used.

    This is the symmetric-AUC convention of the paper, applied once rather than
    per bootstrap replicate (see module docstring).
    """
    auc = roc_auc_score(y, score)
    return (score, 1.0) if auc >= 0.5 else (-score, -1.0)


def load_flagged(sweep_dir):
    """Tags rejected by the degeneracy criterion (written by 14_degeneracy.py).

    Repetition collapse is LOW perplexity by construction, so a collapsed point
    can sit inside the deployable band and top the AUC ranking while being
    destroyed text rather than a watermark. The perplexity filter cannot catch
    it -- on Qwen, formality alpha=0.6 lands at 1.15x ppl with 0.96 AUC purely
    because the text repeats itself. The paper excludes these points, so the
    gap test must too.
    """
    p = Path(sweep_dir) / "degeneracy.csv"
    if not p.exists():
        print(
            f"WARNING: no {p.name} in {sweep_dir}; collapsed operating points "
            "cannot be excluded. Run 14_degeneracy.py first.\n"
        )
        return set()
    return {
        r["tag"]
        for r in csv.DictReader(open(p))
        if str(r.get("flagged", "0")).strip() in ("1", "True", "true")
    }


def pick_best(sweep_dir, deploy_ppl, attack=None):
    """Best KGW point and best steering point inside the deployable band.

    Selection is by auc_post_sym, the metric the paper says it reports. Doing
    this in code rather than by eye is the point: hand-picking rows from the CSV
    is what put a mixed raw/symmetric table in the submitted version.

    Degeneracy-flagged points are dropped before ranking, for the reason given
    in load_flagged.
    """
    csv_path = Path(sweep_dir) / f"sweep_results{suffix_for(attack)}.csv"
    rows = list(csv.DictReader(open(csv_path)))
    flagged = load_flagged(sweep_dir)
    d_max = abs(math.log(deploy_ppl))
    skipped = []
    band = []
    for r in rows:
        if r["kind"] == "unsteered":
            continue
        if r["tag"] in flagged:
            skipped.append(r["tag"])
            continue
        try:
            ppl = float(r["ppl_ratio_mean"])
            sym = float(r["auc_post_sym"])
        except (ValueError, TypeError):
            continue
        # Two-sided, matching the paper's distortion metric. A one-sided
        # `ppl <= deploy_ppl` test is wrong on Qwen, where several directions
        # HOMOGENIZE: formality alpha=0.4 sits at 0.72x ppl, so it passes a
        # one-sided filter while carrying d=0.33 -- more distortion than any
        # point the band is meant to contain.
        if not math.isfinite(ppl) or ppl <= 0 or abs(math.log(ppl)) > d_max:
            continue
        band.append((r["tag"], r["kind"], ppl, sym))
    if not band:
        raise SystemExit(f"no operating points with d <= {d_max:.3f}")

    kgw = [b for b in band if b[1] == "kgw"]
    steer = [b for b in band if b[1] in ("semantic", "random")]
    if not kgw or not steer:
        raise SystemExit("need both a KGW and a steering point inside the band")

    best_kgw = max(kgw, key=lambda b: b[3])
    best_steer = max(steer, key=lambda b: b[3])
    if skipped:
        print(
            f"Excluded {len(skipped)} degeneracy-flagged point(s): {', '.join(sorted(skipped))}\n"
        )
    print(
        f"Deployable band (d <= {d_max:.3f}, i.e. "
        f"{1 / deploy_ppl:.2f}x-{deploy_ppl:.2f}x ppl), ranked by auc_post_sym:"
    )
    for tag, kind, ppl, sym in sorted(band, key=lambda b: -b[3]):
        mark = " <-" if tag in (best_kgw[0], best_steer[0]) else ""
        print(
            f"  {tag:18s} {kind:9s} ppl={ppl:.3f}  d={abs(math.log(ppl)):.3f}  sym={sym:.3f}{mark}"
        )
    print()
    return best_kgw[0], best_steer[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="artifacts/<model>/sweep")
    ap.add_argument("--a", help="tag A (default: best KGW point in band)")
    ap.add_argument("--b", help="tag B (default: best steering point in band)")
    ap.add_argument("--deploy-ppl", type=float, default=DEPLOY_PPL)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--attack",
        default=None,
        help="read sweep_results_<attack>.csv and scores_<attack>/ "
        "instead of the default Pegasus results",
    )
    args = ap.parse_args()

    if args.a and args.b:
        tag_a, tag_b = args.a, args.b
    else:
        tag_a, tag_b = pick_best(args.sweep, args.deploy_ppl, args.attack)

    ya, sa, ga = load_scores(args.sweep, tag_a, args.attack)
    yb, sb, gb = load_scores(args.sweep, tag_b, args.attack)

    if not (len(ya) == len(yb) and np.array_equal(ya, yb) and np.array_equal(ga, gb)):
        raise SystemExit(
            "the two operating points do not share a row layout; they must be "
            "evaluated against the same unsteered negative set in the same order"
        )
    if np.all(ga == -1):
        raise SystemExit("scores carry no prompt groups; re-run the patched 06")

    sa, ora = oriented(ya, sa)
    sb, orb = oriented(yb, sb)
    auc_a = roc_auc_score(ya, sa)
    auc_b = roc_auc_score(yb, sb)

    uniq = np.unique(ga)
    by_group = {g: np.where(ga == g)[0] for g in uniq}
    rng = np.random.default_rng(args.seed)

    diffs, a_boot, b_boot = [], [], []
    for _ in range(args.n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        if len(np.unique(ya[idx])) < 2:
            continue
        fa = roc_auc_score(ya[idx], sa[idx])
        fb = roc_auc_score(yb[idx], sb[idx])
        a_boot.append(fa)
        b_boot.append(fb)
        diffs.append(fa - fb)

    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_one_sided = float(np.mean(diffs <= 0))

    def ci(v):
        return np.percentile(np.asarray(v), [2.5, 97.5])

    a_lo, a_hi = ci(a_boot)
    b_lo, b_hi = ci(b_boot)

    print(f"prompts (clusters): {len(uniq)}   rows: {len(ya)}   replicates: {len(diffs)}")
    print(
        f"orientation: {tag_a} {'as-is' if ora > 0 else 'FLIPPED'}, "
        f"{tag_b} {'as-is' if orb > 0 else 'FLIPPED'}"
    )
    print()
    print(f"  {tag_a:18s} AUC = {auc_a:.3f}  [{a_lo:.3f}, {a_hi:.3f}]")
    print(f"  {tag_b:18s} AUC = {auc_b:.3f}  [{b_lo:.3f}, {b_hi:.3f}]")
    print(f"  {'difference':18s}     = {auc_a - auc_b:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print()
    print(f"  bootstrap P(difference <= 0) = {p_one_sided:.4f}")
    # test both signs: a difference that is reliably NEGATIVE also excludes
    # zero, and reporting that as "not resolved" would understate the finding
    if lo > 0:
        print(f"\n  -> 95% CI excludes zero; {tag_a} > {tag_b} on this data.")
    elif hi < 0:
        print(
            f"\n  -> 95% CI excludes zero; {tag_b} > {tag_a} on this data "
            f"(the difference is reliably negative)."
        )
    else:
        print(
            f"\n  -> 95% CI includes zero; the gap is NOT resolved at n="
            f"{len(uniq)} prompts. Do not call it statistically reliable."
        )

    print("\nSentence for the paper (fill in the model name):")
    print(
        f"  In the deployable band, {tag_a} exceeds {tag_b} by "
        f"{auc_a - auc_b:.2f} AUC (95% CI [{lo:.2f}, {hi:.2f}], cluster "
        f"bootstrap over {len(uniq)} prompts, {len(diffs)} replicates)."
    )


if __name__ == "__main__":
    main()
