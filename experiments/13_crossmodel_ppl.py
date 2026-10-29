"""Re-score every operating point's fluency under an INDEPENDENT judge model.

Why this exists
---------------
The frontier x-axis is the perplexity of steered text under the same base model
that was steered. That is circular, and on Qwen3-1.7B it breaks: steering toward
"formal" or "complex" homogenizes output toward the model's own high-probability
phrasing, so the ratio falls BELOW 1 (0.72-0.90) before degeneration drives it
back up. The ratio is therefore non-monotone in the steering fraction, and points
in the crossing region get flattering scores. Concretely, Qwen formality at
alpha=0.6 lands at ppl ratio 1.15 -- inside the "deployable" band -- with
post-paraphrase AUC 0.98, while alpha=0.4 sits at 0.72 (distortion 0.33). More
steering reads as LESS distortion, which is the metric failing, not the text
improving.

A judge that was never steered has no reason to prefer the steered model's
homogenized register, so it breaks the circularity while keeping the metric,
the units and the axis definition identical. Nothing in the paper's framing has
to change; only the model computing the number does.

Method
------
Identical to phase_evaluate's x-axis in 06_pareto_sweep.py, with the judge
swapped in: per seed, mean perplexity of the steered completions divided by the
mean perplexity of that seed's unsteered completions, then mean +/- std across
seeds. Only the RATIO is meaningful across models -- absolute perplexity under a
different tokenizer is not comparable to the self-scored numbers.

Reads the cached artifacts/<model>/sweep/gen_*.jsonl. Generates nothing, touches
no existing file unless --merge is passed.

Usage:
  # score Qwen's generations with Llama as the judge, and vice versa
  python 13_crossmodel_ppl.py --sweep artifacts/qwen3-1.7b/sweep \
      --judge meta-llama/Llama-3.2-1B-Instruct
  python 13_crossmodel_ppl.py --sweep artifacts/llama-1b/sweep \
      --judge Qwen/Qwen3-1.7B

  # also write ppl_judge_* columns into sweep_results.csv
  python 13_crossmodel_ppl.py --sweep artifacts/qwen3-1.7b/sweep \
      --judge meta-llama/Llama-3.2-1B-Instruct --merge
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils import load_llama

SEEDS = [0, 1, 2]
PPL_SAMPLE = 150  # matches mean_ppl in 06_pareto_sweep.py
MAX_LENGTH = 256


def mean_ppl(model, tok, texts, max_length=MAX_LENGTH, sample=PPL_SAMPLE, desc=""):
    """Mean per-text perplexity. Mirrors 06_pareto_sweep.mean_ppl exactly so the
    judge numbers are constructed the same way as the self-scored ones."""
    vals = []
    for t in tqdm(texts[:sample], desc=desc, leave=False):
        if len(t.strip()) < 5:
            continue
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        if enc["input_ids"].shape[1] < 2:
            continue  # loss undefined for a single token
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        loss = float(out.loss.item())
        if math.isfinite(loss):
            vals.append(math.exp(min(loss, 50.0)))  # guard overflow on degenerate text
    return float(np.mean(vals)) if vals else float("nan")


def load_completions(sweep_dir, tag, seed):
    p = sweep_dir / f"gen_{tag}_s{seed}.jsonl"
    with open(p) as f:
        return [json.loads(line)["completion"] for line in f]


def discover_tags(sweep_dir):
    """Every tag with a gen_<tag>_s0.jsonl, unsteered first."""
    tags = set()
    for p in sweep_dir.glob("gen_*_s0.jsonl"):
        tags.add(re.sub(r"^gen_(.*)_s0\.jsonl$", r"\1", p.name))
    if "unsteered" not in tags:
        raise SystemExit(f"no gen_unsteered_s0.jsonl in {sweep_dir}")
    return ["unsteered"] + sorted(t for t in tags if t != "unsteered")


def alpha_of(tag):
    m = re.search(r"_m([0-9.]+)$", tag)
    return float(m.group(1)) if m else None


def method_of(tag):
    return re.sub(r"_(m|b)[0-9.]+$", "", tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="artifacts/<model>/sweep")
    ap.add_argument(
        "--judge", required=True, help="HF id of a model that was NOT steered in this sweep"
    )
    ap.add_argument(
        "--merge", action="store_true", help="write ppl_judge_mean/std into sweep_results.csv"
    )
    args = ap.parse_args()

    sweep_dir = Path(args.sweep)
    tags = discover_tags(sweep_dir)
    print(f"{len(tags)} operating points in {sweep_dir}")
    print(f"judge: {args.judge}\n")

    model, tok = load_llama(args.judge)

    # per-seed unsteered reference under the judge
    uns = {}
    for s in SEEDS:
        uns[s] = mean_ppl(
            model, tok, load_completions(sweep_dir, "unsteered", s), desc=f"unsteered s{s}"
        )
    print("judge unsteered ppl per seed: " + ", ".join(f"{uns[s]:.2f}" for s in SEEDS) + "\n")

    results = {}
    for tag in tags:
        if tag == "unsteered":
            results[tag] = (1.0, 0.0)
            print(f"  {tag:18s} ratio=1.00 (reference)")
            continue
        ratios = []
        for s in SEEDS:
            p = mean_ppl(model, tok, load_completions(sweep_dir, tag, s), desc=f"{tag} s{s}")
            ratios.append(p / uns[s])
        mean, std = float(np.nanmean(ratios)), float(np.nanstd(ratios))
        results[tag] = (mean, std)
        d = abs(math.log(mean)) if (mean == mean and mean > 0) else float("nan")
        print(f"  {tag:18s} ratio={mean:8.2f}±{std:7.2f}x  d={d:.3f}")

    out_csv = sweep_dir / "crossmodel_ppl.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "judge", "ppl_judge_mean", "ppl_judge_std", "distortion_judge"])
        for tag, (mean, std) in results.items():
            d = abs(math.log(mean)) if (mean == mean and mean > 0) else float("nan")
            w.writerow([tag, args.judge, mean, std, d])
    print(f"\nWrote {out_csv}")

    # ---- the question this script exists to answer ----
    print("\n" + "=" * 64)
    print("MONOTONICITY IN STEERING STRENGTH")
    print("=" * 64)
    print("A fluency metric should not improve as steering increases. Any 'no'")
    print("below means the axis reorders operating points on that model.\n")

    by_method = {}
    for tag in tags:
        a = alpha_of(tag)
        if a is None:
            continue
        by_method.setdefault(method_of(tag), []).append((a, tag))

    any_break = False
    for m, pts in sorted(by_method.items()):
        pts.sort()
        seq = [results[t][0] for _, t in pts]
        finite = [v for v in seq if v == v]
        mono = all(b >= a - 1e-9 for a, b in zip(finite, finite[1:]))
        any_break |= not mono
        shown = "  ".join(
            f"{a:.1f}:{v:.2f}" if v == v else f"{a:.1f}:nan" for (a, _), v in zip(pts, seq)
        )
        print(f"  {m:12s} monotone={'yes' if mono else 'NO ':3s}  {shown}")

    print()
    if any_break:
        print("At least one direction is still non-monotone under the judge.")
        print("The reordering is a property of the text, not of the scoring model:")
        print("report those points as an explicit exception rather than relying on")
        print("the perplexity axis to rank them.")
    else:
        print("All directions are monotone under the independent judge. The")
        print("non-monotonicity was an artifact of scoring steered text with the")
        print("model that was steered; cite the judge ratios on the frontier axis.")
    print("=" * 64)

    if args.merge:
        csv_path = sweep_dir / "sweep_results.csv"
        rows = list(csv.DictReader(open(csv_path)))
        fields = list(rows[0].keys())
        for c in ("ppl_judge_mean", "ppl_judge_std"):
            if c not in fields:
                fields.append(c)
        for r in rows:
            mean, std = results.get(r["tag"], (float("nan"), float("nan")))
            r["ppl_judge_mean"], r["ppl_judge_std"] = mean, std
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nMerged judge columns into {csv_path}")


if __name__ == "__main__":
    main()
