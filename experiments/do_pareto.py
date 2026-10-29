"""Plot-only: regenerate all sweep figures from an existing CSV. No GPU/model.

The sweep (06) recomputes the CSV from scratch every run (it re-encodes all
paraphrased text through the model to get probe AUCs). That is wasteful when you
only want to redraw or restyle a figure. This script reads a finished
sweep_results.csv and produces every figure in seconds, CPU-only.

It also folds in the cleaning logic (drop NaN / perplexity-blowup operating
points, flag ppl<1) so figures and tables come from ONE place.

Figures written next to the input CSV:
  - pareto_detection_vs_fluency.{pdf,png}  (raw perplexity x-axis, log scale)
  - pareto_distortion.{pdf,png}            (two-sided |log ppl| x-axis; robust to ppl<1)
  - frontier_table.tex                     (LaTeX booktabs table, deployable + high-AUC rows)

Usage:
  python experiments/do_pareto.py data/llama-1b/sweep/sweep_results.csv --out-dir outputs/llama
"""

import argparse
import csv
import math
from pathlib import Path

PPL_MAX = 10.0  # ppl_ratio above this = degenerate (text destroyed)
DEPLOY_PPL = 1.15  # "deployable" raw-perplexity threshold
DEPLOY_DIST = 0.10  # "low distortion" |log ppl| threshold
MARKERS = {"semantic": "o", "random": "s", "kgw": "^"}


def fnum(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def load_degenerate(path):
    """Tags flagged by 14_degeneracy.py, if it has been run for this sweep.

    Perplexity cannot see repetition collapse -- a looping completion is highly
    predictable and therefore LOW perplexity -- so the ppl rule alone leaves
    destroyed text on the frontier. On Qwen it keeps formality_m0.6 (10% of
    completions collapsed, ppl 1.15x, inside the deployable band) and
    complexity_m0.8 (14.2%, ppl 2.37x). Both are excluded here.
    """
    dpath = Path(path).parent / "degeneracy.csv"
    if not dpath.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(dpath)):
        if r.get("flagged") == "1":
            out[r["tag"]] = fnum(r.get("degenerate_rate"))
    return out


def load_clean(path):
    """Return (kept_rows, dropped) where kept rows are valid, non-degenerate."""
    rows = list(csv.DictReader(open(path)))
    degen = load_degenerate(path)
    kept, dropped = [], []
    for r in rows:
        r["_ppl"] = fnum(r.get("ppl_ratio_mean"))
        r["_pplstd"] = fnum(r.get("ppl_ratio_std"))
        r["_auc"] = fnum(r.get("auc_post"))
        r["_aucsym"] = fnum(r.get("auc_post_sym"))
        r["_aucstd"] = fnum(r.get("auc_post_std"))
        if r["kind"] == "unsteered":
            kept.append(r)
            continue
        if math.isnan(r["_ppl"]):
            dropped.append((r["tag"], "NaN perplexity (degenerate generation)"))
        elif r["_ppl"] > PPL_MAX:
            dropped.append((r["tag"], f"perplexity blowup ({r['_ppl']:.0f}x)"))
        elif r["tag"] in degen:
            dropped.append(
                (
                    r["tag"],
                    f"repetition collapse "
                    f"({degen[r['tag']] * 100:.1f}% of completions; "
                    f"ppl only {r['_ppl']:.2f}x)",
                )
            )
        else:
            kept.append(r)
    return kept, dropped


def by_method(rows):
    m = {}
    for r in rows:
        if r["kind"] == "unsteered":
            continue
        m.setdefault(r["method"], []).append(r)
    return m


def plot_perplexity(rows, outdir, model_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = by_method(rows)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for method, pts in sorted(methods.items()):
        pts = sorted(pts, key=lambda r: r["_ppl"])
        x = [p["_ppl"] for p in pts]
        y = [p["_auc"] for p in pts]
        yerr = [p["_aucstd"] for p in pts]
        xerr = [p["_pplstd"] for p in pts]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            marker=MARKERS.get(pts[0]["kind"], "o"),
            label=method,
            linewidth=1.5,
            markersize=6,
            capsize=2,
            elinewidth=0.8,
        )
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.axvspan(1.0, 1.5, color="green", alpha=0.06, label="deployable (≤1.5×)")
    ax.set_xscale("log")
    ax.set_xlabel("Generation-time perplexity ratio (log scale)")
    ax.set_ylabel("Post-paraphrase detection AUC")
    ax.set_title(f"{model_name}: detection vs fluency")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = outdir / "pareto_detection_vs_fluency"
    fig.savefig(f"{p}.pdf")
    fig.savefig(f"{p}.png", dpi=150)
    print(f"Wrote {p}.pdf")


def plot_distortion(rows, outdir, model_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = by_method(rows)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for method, pts in sorted(methods.items()):
        pp = [(p, p["_ppl"]) for p in pts if not math.isnan(p["_ppl"]) and p["_ppl"] > 0]
        pp = sorted(pp, key=lambda z: abs(math.log(z[1])))
        x = [abs(math.log(v)) for _, v in pp]
        y = [p["_auc"] for p, _ in pp]
        yerr = [p["_aucstd"] for p, _ in pp]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=MARKERS.get(pp[0][0]["kind"], "o"),
            label=method,
            linewidth=1.5,
            markersize=6,
            capsize=2,
            elinewidth=0.8,
        )
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.axvspan(
        0.0, DEPLOY_DIST, color="green", alpha=0.06, label=f"low distortion (≤{DEPLOY_DIST})"
    )
    ax.set_xlabel("Distortion  |log(perplexity ratio)|")
    ax.set_ylabel("Post-paraphrase detection AUC")
    ax.set_title(f"{model_name}: detection vs distortion")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "pareto_distortion"
    fig.savefig(f"{p}.pdf")
    fig.savefig(f"{p}.png", dpi=150)
    print(f"Wrote {p}.pdf")


def write_table(rows, outdir, model_name):
    """LaTeX table: best per-method point in the low-distortion band + the
    method's highest-AUC point overall (which shows the distortion it costs)."""
    methods = by_method(rows)
    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Ppl. & Distortion & AUC$_{\text{post}}$ \\")
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{4}{l}{\emph{Best in low-distortion band ($|\log\,\text{ppl}|\le "
        + f"{DEPLOY_DIST}"
        + r"$)}} \\"
    )
    order = ["kgw", "sentiment", "formality", "complexity", "random"]
    for m in order:
        if m not in methods:
            continue
        low = [p for p in methods[m] if abs(math.log(p["_ppl"])) <= DEPLOY_DIST and p["_ppl"] > 0]
        if low:
            b = max(low, key=lambda p: p["_aucsym"])
            lines.append(
                f"{m} & {b['_ppl']:.2f} & {abs(math.log(b['_ppl'])):.3f} & {b['_aucsym']:.3f} \\\\"
            )
        else:
            lines.append(f"{m} & -- & -- & (none) \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\emph{Highest-AUC point per method (any distortion)}} \\")
    for m in order:
        if m not in methods:
            continue
        b = max(methods[m], key=lambda p: p["_aucsym"])
        lines.append(
            f"{m} & {b['_ppl']:.2f} & {abs(math.log(b['_ppl'])):.3f} & {b['_aucsym']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out = outdir / "frontier_table.tex"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model-name", default=None)
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="directory for generated figures and tables"
    )
    args = ap.parse_args()

    path = Path(args.csv)
    model_name = args.model_name or path.parent.parent.name  # artifacts/<model>/sweep/
    kept, dropped = load_clean(path)
    outdir = args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {path}  ({len(kept)} usable rows)")
    if dropped:
        print("Dropped degenerate operating points (report in text):")
        for tag, why in dropped:
            print(f"  {tag:18s} {why}")
    flagged = [
        r
        for r in kept
        if r["kind"] != "unsteered" and not math.isnan(r["_ppl"]) and r["_ppl"] < 1.0
    ]
    if flagged:
        print(
            f"ppl<1 (homogenization; distortion axis handles these): "
            f"{', '.join(r['tag'] for r in flagged)}"
        )
    print()

    plot_perplexity(kept, outdir, model_name)
    plot_distortion(kept, outdir, model_name)
    write_table(kept, outdir, model_name)


if __name__ == "__main__":
    main()
