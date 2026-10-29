"""Degeneracy audit: find operating points whose text collapsed, regardless of perplexity.

Why perplexity is not enough
----------------------------
The frontier excludes points whose perplexity ratio exceeds 10x, on the grounds
that they are destroyed text rather than watermarks. That rule misses a whole
failure mode, because the most common form of collapse is REPETITION, and
repeated tokens are highly predictable -- degenerate loops have LOW perplexity by
construction. A point can therefore be visibly destroyed and still score inside
the "deployable" band.

Qwen3-1.7B formality at alpha=0.6 is exactly this case. Perplexity ratio 1.15
self-scored and 1.13 under an independent Llama-3.1-8B judge, i.e. inside the
band, with post-paraphrase AUC 0.98 -- yet the generations include completions
like "The singular and singularly singular singular singular and singularly
singular and singularly singular and ...". Swapping the scoring model does not
help; no perplexity-based axis can rank this text correctly. Detecting it needs a
criterion that looks at repetition directly.

Metrics (pure stdlib, no model, no GPU)
---------------------------------------
Per completion:
  empty        -- fewer than MIN_CHARS non-space characters
  distinct-4   -- unique 4-grams / total 4-grams over whitespace tokens.
                  Fluent prose sits near 1.0; a tight loop approaches 0.
A completion counts as degenerate if it is empty or its distinct-4 falls below
DISTINCT4_MIN. An operating point is flagged when the degenerate fraction exceeds
DEGEN_RATE_MAX.

Outputs a table plus <sweep>/degeneracy.csv. Nothing is modified.

Usage:
  python 14_degeneracy.py --sweep artifacts/qwen3-1.7b/sweep
  python 14_degeneracy.py --sweep artifacts/llama-1b/sweep --distinct4-min 0.6
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

SEEDS = [0, 1, 2]
MIN_CHARS = 5  # matches the guard in mean_ppl
DISTINCT4_MIN = 0.50  # below this, over half the 4-grams are repeats
NGRAM = 4
PPL_MAX = 10.0  # the frontier's existing exclusion rule (do_pareto.py)

# Flag a point when more than this fraction of its completions collapsed.
#
# Any cutoff is a choice, so here is the evidence for this one. Mean distinct-4
# is sharply bimodal across all 50 operating points of both models: every healthy
# point sits at 0.96-1.00, and the collapsed ones at 0.21-0.90, with a clean gap
# and nothing in between. Ranking by collapse RATE reproduces that split at 5%:
# it selects Qwen formality_m0.6 (10.0%, distinct-4 0.885) and complexity_m0.8
# (14.2%, 0.862) and Llama random_m0.6/m0.8 (11.3%/18.3%), and selects nothing
# whose distinct-4 is in the healthy cluster. 10% would have been knife-edge --
# Qwen formality_m0.6 lands on exactly 0.100 -- which is a bad place for a
# threshold to sit. Report the full table (see degeneracy.csv) so readers can see
# the split does not depend on the exact number.
DEGEN_RATE_MAX = 0.05


def distinct_n(text, n=NGRAM):
    """Unique n-grams / total n-grams over whitespace tokens. 1.0 if too short."""
    toks = text.split()
    if len(toks) < n + 1:
        return 1.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def load_completions(sweep_dir, tag, seed):
    with open(sweep_dir / f"gen_{tag}_s{seed}.jsonl") as f:
        return [json.loads(line)["completion"] for line in f]


def discover_tags(sweep_dir):
    tags = {
        re.sub(r"^gen_(.*)_s0\.jsonl$", r"\1", p.name) for p in sweep_dir.glob("gen_*_s0.jsonl")
    }
    if "unsteered" not in tags:
        raise SystemExit(f"no gen_unsteered_s0.jsonl in {sweep_dir}")
    return ["unsteered"] + sorted(t for t in tags if t != "unsteered")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="artifacts/<model>/sweep")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional output CSV, leaving the input directory unchanged",
    )
    ap.add_argument("--distinct4-min", type=float, default=DISTINCT4_MIN)
    ap.add_argument("--degen-rate-max", type=float, default=DEGEN_RATE_MAX)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep)
    tags = discover_tags(sweep_dir)

    # perplexity ratios, so the table shows what the ppl rule would have decided
    ppl = {}
    csv_path = sweep_dir / "sweep_results.csv"
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path)):
            try:
                ppl[r["tag"]] = float(r["ppl_ratio_mean"])
            except (ValueError, TypeError):
                ppl[r["tag"]] = float("nan")

    print(
        f"{sweep_dir}   empty: <{MIN_CHARS} chars   "
        f"degenerate: empty or distinct-{NGRAM} < {args.distinct4_min}\n"
    )
    header = f"{'tag':18s} {'ppl':>8s} {'empty%':>7s} {'degen%':>7s} {'distinct-4':>11s}   flag"
    print(header)
    print("-" * len(header))

    rows, flagged = [], []
    for tag in tags:
        texts = []
        for s in SEEDS:
            texts += load_completions(sweep_dir, tag, s)
        n = len(texts)
        empties = sum(1 for t in texts if len(t.strip()) < MIN_CHARS)
        d4 = [distinct_n(t) for t in texts if len(t.strip()) >= MIN_CHARS]
        degen = empties + sum(1 for v in d4 if v < args.distinct4_min)
        empty_rate, degen_rate = empties / n, degen / n
        mean_d4 = statistics.fmean(d4) if d4 else 0.0

        p = ppl.get(tag, float("nan"))
        is_flagged = degen_rate > args.degen_rate_max
        # A flagged point is MISSED by the existing rule when its perplexity is
        # finite and under the blow-up cut -- that is the whole point of this
        # script, since repetition keeps perplexity low. (An earlier version
        # tested p <= 1.15, the deployable band, and so reported Qwen
        # complexity_m0.8 at 2.37x as already handled when it was not.)
        missed = is_flagged and p == p and p <= PPL_MAX
        deployable = missed and p <= 1.15
        mark = (
            "MISSED BY PPL (deployable!)"
            if deployable
            else "MISSED BY PPL"
            if missed
            else "degenerate (ppl rule catches it)"
            if is_flagged
            else ""
        )
        if is_flagged:
            flagged.append((tag, degen_rate, p, missed, deployable))

        print(
            f"{tag:18s} {p:8.2f} {empty_rate * 100:6.1f}% {degen_rate * 100:6.1f}% "
            f"{mean_d4:11.3f}   {mark}"
        )
        rows.append(
            {
                "tag": tag,
                "n": n,
                "ppl_ratio_mean": p,
                "empty_rate": empty_rate,
                "degenerate_rate": degen_rate,
                "mean_distinct4": mean_d4,
                "flagged": int(is_flagged),
                "missed_by_ppl_rule": int(missed),
                "in_deployable_band": int(deployable),
            }
        )

    out = args.out or sweep_dir / "degeneracy.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")

    print("\n" + "=" * 72)
    missed = [f for f in flagged if f[3]]
    if missed:
        print("COLLAPSED TEXT THAT THE PERPLEXITY RULE KEEPS")
        print("=" * 72)
        for tag, rate, p, _, deployable in missed:
            note = "  <-- inside the deployable band" if deployable else ""
            print(
                f"  {tag:18s} {rate * 100:.1f}% of completions collapsed, "
                f"ppl ratio only {p:.2f}x{note}"
            )
        print("\nThese points must be excluded from the frontier or annotated on it.")
        print("They are destroyed text that perplexity scores as acceptable, because")
        print("repetition is high-probability. Pass the tags to do_pareto.py, or let")
        print("it read degeneracy.csv, so the figures and the text agree.")
    elif flagged:
        print(
            f"Every flagged point is also caught by the perplexity rule (ppl > {PPL_MAX}x or NaN);"
        )
        print("no extra exclusions needed on this model.")
    else:
        print("No operating point exceeds the degeneracy threshold.")
    print("=" * 72)


if __name__ == "__main__":
    main()
