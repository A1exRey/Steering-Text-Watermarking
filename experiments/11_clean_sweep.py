"""Clean a sweep_results CSV before it feeds figures/tables.

Two data-quality issues seen on the Qwen3-1.7B run (and possibly others at high
steering fractions):

  1. NaN or blown-up perplexity (e.g. ppl_ratio = 5834 or NaN). Cause: at high
     steering fractions the model degenerates into empty strings or single
     repeated tokens, which the base model scores as astronomically perplexed
     (or, if all-empty, averages to NaN). These points are not meaningful
     operating points -- they are 'the watermark destroyed the text' -- and a
     5000x outlier makes the Pareto axis unreadable.

  2. Perplexity ratio < 1.0 (steered scores MORE fluent than unsteered). This is
     real, not a bug: steering toward formal/complex registers can homogenize
     text toward higher-probability phrasing. It is kept (it is honest data) but
     flagged, because it complicates the 'fluency cost' x-axis.

This script writes <input>_clean.csv with:
  - degenerate rows (NaN ppl, or ppl_ratio > PPL_MAX) DROPPED
  - a 'note' column annotating ppl<1 and near-degenerate points
and prints a summary of what was dropped and why.

Usage:
  python 11_clean_sweep.py artifacts/qwen3-1.7b/sweep/sweep_results.csv
"""

import csv
import math
import sys
from pathlib import Path

PPL_MAX = 10.0  # ppl_ratio above this = degenerate text, not an operating point


def fnum(s):
    try:
        v = float(s)
        return v
    except (ValueError, TypeError):
        return float("nan")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    rows = list(csv.DictReader(open(path)))

    kept, dropped = [], []
    for r in rows:
        ppl = fnum(r.get("ppl_ratio_mean"))
        note = ""
        if math.isnan(ppl):
            dropped.append((r["tag"], "NaN perplexity (degenerate/empty generation)"))
            continue
        if ppl > PPL_MAX:
            dropped.append((r["tag"], f"perplexity blowup ({ppl:.0f}x); text destroyed"))
            continue
        if ppl < 1.0 and r["kind"] != "unsteered":
            note = "ppl<1: steered scored more fluent than unsteered (homogenization)"
        r["note"] = note
        kept.append(r)

    out = path.with_name(path.stem + "_clean.csv")
    fields = list(rows[0].keys())
    if "note" not in fields:
        fields = fields + ["note"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in kept:
            r.setdefault("note", "")
            w.writerow(r)

    print(f"Input : {path}")
    print(f"Output: {out}")
    print(f"Kept {len(kept)} / {len(rows)} rows.\n")
    if dropped:
        print("DROPPED (degenerate -- exclude from figures, mention in text):")
        for tag, why in dropped:
            print(f"  {tag:18s} {why}")
    flagged = [r for r in kept if r.get("note")]
    if flagged:
        print(
            f"\nFLAGGED ppl<1 (kept, but note in caption): {', '.join(r['tag'] for r in flagged)}"
        )
    print("\nReminder: report the dropped points in the text as 'steering at the")
    print("highest fractions degenerated (empty/repetitive output), which we")
    print("exclude from the frontier' -- this is itself evidence that steering's")
    print("only path to high detectability is text destruction.")


if __name__ == "__main__":
    main()
