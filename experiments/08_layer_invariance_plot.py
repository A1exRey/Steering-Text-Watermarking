"""Layer-invariance figure: post-paraphrase detection AUC vs probe layer.

This is the paper's MECHANISM figure. A genuine watermark localized to the
steered representation would peak at the steered layer and fall off elsewhere.
The diagnostic shows the opposite: detectability is flat across depth for every
method, including the random control -- evidence that steering does not create a
localized semantic mark but a global, content-independent shift.

Reads artifacts/<model>/sweep/artifact_diagnostic.json (produced by 07).
Writes artifacts/<model>/sweep/layer_invariance.{pdf,png}.

NOTE ON SERIES DISCOVERY. A previous version hardcoded the series keys
("formality_m1.0", "sentiment_m3.0") from the old absolute-multiplier
convention. After 07 moved to norm-relative fractions the keys became
"formality_m0.1" / "sentiment_m0.4", every series was silently skipped, and the
script wrote an EMPTY figure while still exiting 0 (the only trace was a
"No artists with labels found" warning). Series are now discovered from the JSON
itself -- lowest available fraction as the "(low)" curves, highest as the
"(strong)" contrast -- so the plot cannot go stale against 07's grid again.

Run: python 08_layer_invariance_plot.py
"""

import json
import re

import numpy as np

import config

SWEEP_DIR = config.ART / "sweep"
DIAG = SWEEP_DIR / "artifact_diagnostic.json"
OUT = SWEEP_DIR / "layer_invariance"

DISPLAY_NAME = {
    "llama-1b": "Llama-3.2-1B",
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-4b": "Qwen3-4B",
    "olmo3-7b": "OLMo-3-7B",
    "llama-8b": "Llama-3.1-8B",
}

# Fixed colour/style per method so Llama and Qwen figures read identically.
STYLE = {
    "formality": ("C0", "-", "o"),
    "complexity": ("C2", "-", "o"),
    "sentiment": ("C3", "-", "o"),
    "random": ("C1", "--", "s"),
}
ORDER = ["formality", "complexity", "sentiment", "random"]
STRONG_METHOD = "sentiment"  # the one strong-signal contrast curve


def parse_key(key):
    """'formality_m0.1' -> ('formality', 0.1); None for summary keys."""
    m = re.match(r"^(.+)_m([0-9.]+)$", key)
    return (m.group(1), float(m.group(2))) if m else None


def main():
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", type=Path, default=SWEEP_DIR)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    diag_path = args.sweep / "artifact_diagnostic.json"
    out = args.out_dir / "layer_invariance"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(diag_path) as f:
        diag = json.load(f)

    # method -> [(fraction, original key), ...] sorted by fraction. Keep the key
    # verbatim rather than rebuilding it from the float, so no formatting
    # mismatch can reintroduce the silent-empty-plot bug.
    avail = {}
    for key in diag:
        parsed = parse_key(key)
        if parsed:
            avail.setdefault(parsed[0], []).append((parsed[1], key))
    for m in avail:
        avail[m].sort()
    if not avail:
        raise SystemExit(f"no '<method>_m<fraction>' entries in {DIAG}")

    any_key = next(iter(avail.values()))[0][1]
    layers = sorted(int(L) for L in diag[any_key]["act_by_layer"])

    series = []
    for m in ORDER:
        if m in avail:
            series.append((avail[m][0][1], f"{m} (low)", *STYLE[m]))
    if STRONG_METHOD in avail and len(avail[STRONG_METHOD]) > 1:
        series.append((avail[STRONG_METHOD][-1][1], f"{STRONG_METHOD} (strong)", "C4", ":", "D"))

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    steer_layer = config.STEER_LAYER
    plotted = 0

    for key, label, color, ls, marker in series:
        if key not in diag:
            print(f"  [warn] {key} absent from the diagnostic; skipping")
            continue
        d = diag[key]["act_by_layer"]
        y = [d[str(L)] for L in layers]
        ax.plot(
            layers, y, marker=marker, color=color, ls=ls, label=label, linewidth=1.6, markersize=5
        )
        # ring the peak layer: the paper's claim is that it is never the steered one
        pk = int(np.argmax(y))
        ax.plot(
            layers[pk],
            y[pk],
            marker="o",
            mfc="none",
            mec=color,
            markersize=13,
            markeredgewidth=1.6,
            ls="none",
        )
        plotted += 1

    if plotted == 0:
        raise SystemExit(
            f"no series matched. Keys present: {sorted(k for k in diag if parse_key(k))}"
        )

    summ = diag.get("_layer_invariance_summary", {})
    n_peak = summ.get("n_methods_steered_is_peak")
    n_meth = summ.get("n_methods")
    max_rng = summ.get("max_range")

    ax.axvline(steer_layer, color="gray", ls="-.", lw=1, alpha=0.7)
    ax.text(
        steer_layer,
        ax.get_ylim()[0] + 0.02,
        " steered layer",
        fontsize=8,
        color="gray",
        va="bottom",
    )
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Probe layer")
    ax.set_ylabel("Post-paraphrase detection AUC (symmetric)")

    title = f"{DISPLAY_NAME.get(config.ACTIVE, config.ACTIVE)}: detectability is layer-invariant"
    if n_peak is not None and max_rng is not None:
        title += f"\nsteered layer is peak in {n_peak}/{n_meth} directions; max range {max_rng:.3f}"
    ax.set_title(title, fontsize=10)
    ax.set_xticks(layers)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=150)
    print(f"Wrote {out}.pdf / .png  ({plotted} series)")

    # quantitative flatness summary for the paper text
    print("\nFlatness (max-min AUC across layers; small = layer-invariant):")
    for key, label, *_ in series:
        if key not in diag:
            continue
        vals = list(diag[key]["act_by_layer"].values())
        at_steer = diag[key]["act_by_layer"][str(steer_layer)]
        print(
            f"  {label:24s}: range={max(vals) - min(vals):.3f}  "
            f"peaks_at_steered_layer={at_steer >= max(vals) - 1e-9}"
        )
    if max_rng is not None:
        print(
            f"\nCite: max range {max_rng:.3f} across {summ.get('n_layers_probed')} "
            f"layers; steered layer is peak in {n_peak}/{n_meth} directions."
        )


if __name__ == "__main__":
    main()
