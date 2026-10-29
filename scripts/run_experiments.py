"""Run the recovered GPU pipeline in a separate, resumable output directory."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llama-1b", "qwen3-1.7b")


def commands(stage, sweep, model):
    plan = []
    if stage == "directions":
        return [("01_extract_direction.py", []), ("01b_extra_directions.py", [])]
    if stage in ("sweep", "all"):
        plan.append(("06_pareto_sweep.py", []))
    if stage in ("dipper", "all"):
        plan.append(("15_dipper_attack.py", ["--sweep", sweep]))
    if stage == "evaluate":
        plan.append(("06_pareto_sweep.py", ["--eval-only"]))
    if stage in ("evaluate", "all"):
        plan.append(("06_pareto_sweep.py", ["--attack", "dipper", "--eval-only"]))
    if stage in ("diagnostics", "all"):
        # Both archived crossmodel_ppl.csv files record the same 8B judge.
        judge = "meta-llama/Llama-3.1-8B-Instruct"
        plan += [
            ("07_artifact_diagnostic.py", []),
            ("13_crossmodel_ppl.py", ["--sweep", sweep, "--judge", judge]),
            ("14_degeneracy.py", ["--sweep", sweep]),
        ]
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "directions", "sweep", "dipper", "evaluate", "diagnostics"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument(
        "--from-saved",
        action="store_true",
        help="copy the saved data into a NEW run directory before recomputing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without copying files or loading models",
    )
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    data_root = (ROOT / "data").resolve()
    if output == data_root or data_root in output.parents:
        parser.error("--output-root must be outside the archived data directory")
    target = output / args.model
    if args.from_saved and target.exists():
        parser.error("--from-saved requires a new model directory; choose another --output-root")
    if args.from_saved and args.stage in ("all", "sweep", "directions"):
        parser.error(
            "--from-saved is for evaluate, dipper, or diagnostics; omit it for a fresh sweep"
        )
    if args.stage == "directions" and (target / "sweep").exists():
        parser.error(
            "rebuild directions in a new output root, so cached generations cannot use stale vectors"
        )
    env = dict(os.environ, AOS_MODEL=args.model, AOS_ARTIFACT_ROOT=str(output), MPLBACKEND="Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
    print(f"Model: {args.model}; stage: {args.stage}; outputs: {target}", flush=True)
    if not args.dry_run:
        if args.from_saved:
            shutil.copytree(data_root / args.model, target)
        else:
            target.mkdir(parents=True, exist_ok=True)
            if args.stage != "directions" and not (target / "directions.pt").exists():
                shutil.copy2(data_root / args.model / "directions.pt", target / "directions.pt")
    for filename, arguments in commands(args.stage, target / "sweep", args.model):
        cmd = [sys.executable, str(ROOT / "experiments" / filename), *map(str, arguments)]
        print("Run:", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
