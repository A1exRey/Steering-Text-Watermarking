"""Rebuild paper analyses from saved data, without a GPU or model downloads."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"llama-1b": "Llama-3.2-1B", "qwen3-1.7b": "Qwen3-1.7B"}


def run_script(name, arguments, log, model=None):
    env = dict(os.environ, MPLBACKEND="Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
    if model:
        env["AOS_MODEL"] = model
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / name), *map(str, arguments)],
        env=env,
        capture_output=True,
        text=True,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {log}")


def combined_results(sweep):
    """Join attacks by tag; preserve every row and explicitly mark exclusions."""
    attacks = {}
    for attack, filename in (
        ("pegasus", "sweep_results.csv"),
        ("dipper", "sweep_results_dipper.csv"),
    ):
        with (sweep / filename).open() as handle:
            attacks[attack] = {r["tag"]: r for r in csv.DictReader(handle)}
    if attacks["pegasus"].keys() != attacks["dipper"].keys():
        raise ValueError("Pegasus and DIPPER cover different operating points")
    with (sweep / "degeneracy.csv").open() as handle:
        flagged = {r["tag"] for r in csv.DictReader(handle) if r["flagged"] == "1"}
    results = []
    for tag, peg in attacks["pegasus"].items():
        dip = attacks["dipper"][tag]
        ppl = float(peg["ppl_ratio_mean"])
        dip_ppl = float(dip["ppl_ratio_mean"])
        if not (math.isnan(ppl) and math.isnan(dip_ppl)) and not math.isclose(ppl, dip_ppl):
            raise ValueError(f"{tag}: the attacks disagree about generation perplexity")
        valid = math.isfinite(ppl) and 0 < ppl <= 10
        reason = "invalid_or_high_perplexity" if not valid else ""
        if tag in flagged:
            reason = ";".join(filter(None, (reason, "repetition_or_empty_text")))
        results.append(
            {
                "tag": tag,
                "method": peg["method"],
                "kind": peg["kind"],
                "param": float(peg["param"]),
                "ppl_ratio": ppl,
                "distortion": abs(math.log(ppl)) if math.isfinite(ppl) and ppl > 0 else math.nan,
                "auc_pegasus": float(peg["auc_post_sym"]),
                "auc_dipper": float(dip["auc_post_sym"]),
                "excluded": bool(reason),
                "exclusion_reason": reason,
            }
        )
    return results


def select_in_band(rows, attack, kind):
    candidates = [
        r
        for r in rows
        if not r["excluded"]
        and r["distortion"] <= math.log(1.15)
        and (r["kind"] == "kgw" if kind == "kgw" else r["kind"] in ("semantic", "random"))
    ]
    if not candidates:
        raise ValueError(f"No {kind} point in the deployable band")
    return max(candidates, key=lambda r: r[f"auc_{attack}"])


def write_tables(data_root, out):
    headline = []
    for model, label in MODELS.items():
        rows = combined_results(data_root / model / "sweep")
        target = out / model
        target.mkdir(parents=True, exist_ok=True)
        with (target / "all_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for attack in ("pegasus", "dipper"):
            kgw = select_in_band(rows, attack, "kgw")
            steer = select_in_band(rows, attack, "steering")
            headline.append(
                {
                    "model": label,
                    "attack": attack,
                    "kgw_tag": kgw["tag"],
                    "kgw_auc": kgw[f"auc_{attack}"],
                    "steering_tag": steer["tag"],
                    "steering_auc": steer[f"auc_{attack}"],
                    "gap": kgw[f"auc_{attack}"] - steer[f"auc_{attack}"],
                }
            )
    (out / "headline_results.json").write_text(json.dumps(headline, indent=2) + "\n")
    lines = ["| Model | Attack | KGW | Best steering | Gap |", "|---|---|---:|---:|---:|"]
    for r in headline:
        lines.append(
            f"| {r['model']} | {r['attack']} | {r['kgw_auc']:.3f} | {r['steering_auc']:.3f} | {r['gap']:+.3f} |"
        )
    (out / "headline_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def verify_degeneracy(source, generated):
    """Compare recomputed collapse rates with the archived analysis."""

    def load(path):
        with path.open() as handle:
            return {r["tag"]: r for r in csv.DictReader(handle)}

    original, rebuilt = load(source), load(generated)
    # Exploratory SemStamp files are retained but are not in the paper tables.
    for tag, row in original.items():
        other = rebuilt[tag]
        for key in ("n", "empty_rate", "degenerate_rate", "mean_distinct4", "flagged"):
            if not math.isclose(float(row[key]), float(other[key]), abs_tol=1e-12):
                raise ValueError(f"Recomputed degeneracy differs: {tag}/{key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/reproduced")
    parser.add_argument(
        "--tables-only", action="store_true", help="standard-library-only numeric summary"
    )
    parser.add_argument(
        "--bootstrap", action="store_true", help="also repeat the four cluster bootstrap gap tests"
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    data_root, out = args.data_dir.resolve(), args.out.resolve()
    if out == data_root or data_root in out.parents:
        parser.error("--out must be outside --data-dir")
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")
    out.mkdir(parents=True, exist_ok=True)
    write_tables(data_root, out)
    if args.tables_only:
        return
    for model, label in MODELS.items():
        sweep = data_root / model / "sweep"
        for attack, suffix in (("pegasus", ""), ("dipper", "_dipper")):
            target = out / model / attack
            run_script(
                "do_pareto.py",
                [sweep / f"sweep_results{suffix}.csv", "--model-name", label, "--out-dir", target],
                target / "plot.log",
            )
            if args.bootstrap:
                options = ["--sweep", sweep, "--n-boot", args.n_boot]
                if suffix:
                    options += ["--attack", "dipper"]
                run_script("12_gap_test.py", options, target / "bootstrap.log")
        run_script(
            "08_layer_invariance_plot.py",
            ["--sweep", sweep, "--out-dir", out / model],
            out / model / "layers.log",
            model=model,
        )
        run_script(
            "14_degeneracy.py",
            ["--sweep", sweep, "--out", out / model / "degeneracy.csv"],
            out / model / "degeneracy.log",
        )
        verify_degeneracy(sweep / "degeneracy.csv", out / model / "degeneracy.csv")
        print(
            f"Rebuilt {label}: both frontiers, layer plot, collapse rates"
            + (", bootstrap CIs" if args.bootstrap else "")
        )
    run_script(
        "17_attack_frontier_plot.py",
        [
            "--sweep",
            data_root / "llama-1b/sweep",
            "--name",
            MODELS["llama-1b"],
            "--sweep",
            data_root / "qwen3-1.7b/sweep",
            "--name",
            MODELS["qwen3-1.7b"],
            "--out",
            out / "attack_frontier.pdf",
        ],
        out / "attack_frontier.log",
    )
    print(f"Wrote analyses to {out}")


if __name__ == "__main__":
    main()
