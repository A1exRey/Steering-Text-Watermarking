"""Pegasus-vs-DIPPER frontier overlay, for the appendix.

DIPPER currently reaches the paper only as two table columns. That hides the
shape of the result. The claim is not "every AUC drops under the stronger
attack" -- it is "KGW's curve drops and steering's barely moves", and a table of
25 pairs per model makes the reader do that subtraction by hand, twice.

Perplexity is a property of the GENERATION, not of the attack, so a tag sits at
the same x under both attacks. The two curves are therefore sampled at
identical points and the vertical distance between them is exactly the damage
the stronger attacker does. That is the entire figure; the connectors draw it.

Cleaning matches do_pareto.py so figures and tables drop the same points: NaN or
>10x perplexity ratio (destroyed text), plus anything degeneracy.csv flags.
Repetition collapse is LOW perplexity by construction -- a looping completion is
highly predictable -- so the perplexity rule alone leaves Qwen formality
alpha=0.6 on the frontier at 0.96 AUC under DIPPER, which would be the top of
the deployable band and is destroyed text.

Two conventions are inherited deliberately:

  colour   assigned by sorted() method name, which is the implicit order
           matplotlib gives do_pareto.py, so a method keeps its colour across
           every figure in the paper. The sort is over the union of both
           panels, so a method missing from one model does not shift the rest.

  x-limits per panel, NOT shared. Qwen's surviving points span d <= 0.38 while
           Llama's reach 1.63; a shared axis would squeeze the Qwen panel into
           its left fifth, which is the only region anyone reads.

y is auc_post_sym, the symmetric AUC the tables report, rather than the raw
auc_post the older figures use. They differ on one point in fifty and only by
0.011, but this figure sits next to the per-point tables and should not
disagree with them.

CPU-only, no model, no re-encoding; runs in about a second.

Usage:
  python 17_attack_frontier_plot.py \
      --sweep artifacts/llama-1b/sweep    --name "Llama-3.2-1B" \
      --sweep artifacts/qwen3-1.7b/sweep  --name "Qwen3-1.7B" \
      --out attack_frontier.pdf
"""

import argparse
import csv
import math
from pathlib import Path

PPL_MAX = 10.0  # ppl ratio above this = destroyed text (do_pareto.py rule)
DEPLOY_DIST = 0.10  # shaded low-distortion band, matching Figures 2 and 3
MARKERS = {"semantic": "o", "random": "s", "kgw": "^"}


def fnum(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def flagged_tags(sweep):
    """Tags rejected by 14_degeneracy.py. See module docstring for why."""
    p = Path(sweep) / "degeneracy.csv"
    if not p.exists():
        print(
            f"WARNING: no degeneracy.csv in {sweep}; collapsed operating "
            "points cannot be excluded. Run 14_degeneracy.py first."
        )
        return set()
    return {
        r["tag"]
        for r in csv.DictReader(open(p))
        if str(r.get("flagged", "0")).strip() in ("1", "true", "True")
    }


def load(sweep, attack, flagged):
    """{tag: point} for one attack, already cleaned."""
    fname = "sweep_results.csv" if attack is None else f"sweep_results_{attack}.csv"
    p = Path(sweep) / fname
    if not p.exists():
        raise SystemExit(
            f"missing {p}\nRun 06_pareto_sweep.py"
            + (f" --attack {attack} --eval-only first." if attack else " first.")
        )
    out = {}
    for r in csv.DictReader(open(p)):
        if r["kind"] == "unsteered" or r["tag"] in flagged:
            continue
        ppl = fnum(r.get("ppl_ratio_mean"))
        if not math.isfinite(ppl) or ppl <= 0 or ppl > PPL_MAX:
            continue
        out[r["tag"]] = {
            "method": r["method"],
            "kind": r["kind"],
            "d": abs(math.log(ppl)),
            "auc": fnum(r.get("auc_post_sym")),
            "err": fnum(r.get("auc_post_std")),
        }
    return out


def panel(ax, peg, dip, colours, title, xmax=None):
    """Draw one model. Returns nothing; legends are built by the caller."""
    shared = sorted(set(peg) & set(dip))
    for tag in shared:
        # integrity check: same generation, so the same x under both attacks.
        # a mismatch means the two CSVs were built from different generations
        # and the overlay would be comparing different text.
        if abs(peg[tag]["d"] - dip[tag]["d"]) > 1e-6:
            print(
                f"WARNING: {tag} has d={peg[tag]['d']:.4f} under Pegasus but "
                f"{dip[tag]['d']:.4f} under DIPPER; the CSVs disagree about "
                "the generation, not just the attack"
            )

    methods = {}
    for tag in shared:
        methods.setdefault(peg[tag]["method"], []).append(tag)

    for method, tags in sorted(methods.items()):
        tags = sorted(tags, key=lambda t: peg[t]["d"])
        c = colours[method]
        m = MARKERS.get(peg[tags[0]]["kind"], "o")
        x = [peg[t]["d"] for t in tags]

        # the vertical drop per operating point: the figure's actual content
        for t in tags:
            ax.plot(
                [peg[t]["d"]] * 2,
                [peg[t]["auc"], dip[t]["auc"]],
                color=c,
                lw=0.8,
                alpha=0.35,
                zorder=1,
            )

        ax.errorbar(
            x,
            [peg[t]["auc"] for t in tags],
            yerr=[peg[t]["err"] for t in tags],
            color=c,
            marker=m,
            ls="-",
            lw=1.4,
            ms=5,
            capsize=2,
            elinewidth=0.7,
            zorder=3,
        )
        ax.errorbar(
            x,
            [dip[t]["auc"] for t in tags],
            yerr=[dip[t]["err"] for t in tags],
            color=c,
            marker=m,
            ls="--",
            lw=1.4,
            ms=5,
            mfc="white",
            capsize=2,
            elinewidth=0.7,
            zorder=3,
        )

    ax.axhline(0.5, color="gray", ls=":", lw=1, zorder=0)
    ax.axvspan(0.0, DEPLOY_DIST, color="green", alpha=0.06, zorder=0)
    ax.set_xlim(0.0, xmax)
    # same wording as do_pareto.py, so the appendix figure reads like Figs 2-3
    ax.set_xlabel("Distortion  |log(perplexity ratio)|", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep",
        action="append",
        required=True,
        help="artifacts/<model>/sweep; repeat for a second panel",
    )
    ap.add_argument(
        "--name", action="append", default=None, help="panel title; repeat once per --sweep"
    )
    ap.add_argument(
        "--attack", default="dipper", help="reads sweep_results_<attack>.csv as the strong attack"
    )
    ap.add_argument(
        "--attack-label",
        default=None,
        help="legend label for --attack (default: its upper-case name)",
    )
    ap.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="clip the distortion axis. Llama's surviving points "
        "reach d=1.6 while the deployable band ends at 0.14, "
        "so autoscale leaves the region of interest in the "
        "left tenth of the panel; 0.75 is a readable zoom. "
        "Anything beyond is then off-scale -- say so in the "
        "caption if you use it",
    )
    ap.add_argument(
        "--out", default="attack_frontier.pdf", help="written next to the FIRST --sweep"
    )
    args = ap.parse_args()

    names = args.name or []
    if len(names) < len(args.sweep):
        names += [Path(s).parent.name for s in args.sweep[len(names) :]]
    label = args.attack_label or args.attack.upper()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = []
    for sweep in args.sweep:
        flagged = flagged_tags(sweep)
        peg = load(sweep, None, flagged)
        dip = load(sweep, args.attack, flagged)
        shared = set(peg) & set(dip)
        if not shared:
            raise SystemExit(f"{sweep}: no operating point survives cleaning under both attacks")
        panels.append((peg, dip))
        drops = [peg[t]["auc"] - dip[t]["auc"] for t in shared]
        print(
            f"{sweep}: {len(shared)} points survive cleaning; "
            f"mean AUC lost to {label} = {sum(drops) / len(drops):+.3f}"
        )

    # one colour per method across BOTH panels, in do_pareto.py's implicit order
    kinds = {p["method"]: p["kind"] for peg, _ in panels for p in peg.values()}
    every = sorted(kinds)
    colours = {m: f"C{i}" for i, m in enumerate(every)}

    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.1), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (peg, dip), title in zip(axes, panels, names):
        panel(ax, peg, dip, colours, title, args.xmax)
    axes[0].set_ylabel("Post-attack detection AUC", fontsize=9)

    method_keys = [
        Line2D([], [], color=colours[m], marker=MARKERS.get(kinds[m], "o"), ls="-", ms=5, label=m)
        for m in every
    ]
    attack_keys = [
        Line2D([], [], color="black", ls="-", marker="o", ms=5, label="Pegasus"),
        Line2D([], [], color="black", ls="--", marker="o", ms=5, mfc="white", label=label),
    ]
    if len(axes) == 1:
        # one panel: a second legend() call on the same axes would replace the
        # first, so merge them
        axes[0].legend(
            handles=method_keys + attack_keys, fontsize=7, loc="lower right", framealpha=0.9
        )
    else:
        axes[0].legend(handles=method_keys, fontsize=7, loc="lower right", framealpha=0.9)
        axes[-1].legend(handles=attack_keys, fontsize=7, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    out = Path(args.sweep[0]) / args.out
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
