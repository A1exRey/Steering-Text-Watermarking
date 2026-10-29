"""Check archived data integrity, prompt alignment, and detector AUCs offline."""

import argparse
import base64
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llama-1b", "qwen3-1.7b")
PAPER_FIGURES = {
    "llama_32_1b": "llama-1b",
    "qwen3_1_7b": "qwen3-1.7b",
}
SECRET_PATTERNS = (
    rb"hf_[A-Za-z0-9]{20,}",
    rb"(?:ghp_|github_pat_|sk-)[A-Za-z0-9_\-]{20,}",
    rb"AKIA[0-9A-Z]{16}",
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
)
MACHINE_PATH = re.compile(
    rb"/(?:home|mnt|Users)/(?:alex|jovyan|pletenev|shares)(?:/|\b)|SR003\.nfs2"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_integrity(root):
    manifest = json.loads((root / "data/manifest.json").read_text())
    entries = manifest["files"] + manifest.get("derived_files", [])
    for entry in entries:
        path = root / entry["path"]
        require(path.is_file(), f"missing {entry['path']}")
        require(path.stat().st_size == entry["bytes"], f"size changed: {entry['path']}")
        require(digest(path) == entry["sha256"], f"checksum changed: {entry['path']}")
    for stem, model in PAPER_FIGURES.items():
        for figure in ("pareto_distortion", "layer_invariance"):
            paper = root / "paper/new_figures" / f"{stem}_{figure}.pdf"
            saved = root / "data" / model / "sweep" / f"{figure}.pdf"
            require(digest(paper) == digest(saved), f"paper figure differs: {paper.name}")
    print(f"PASS: {len(entries)} checksums and all four paper figures")


def scan_repository(root):
    """Report filenames only; never echo possible secret values."""
    skip = {".git", ".venv", "__pycache__", "outputs", ".pytest_cache", ".ruff_cache"}
    scanned = 0
    for path in root.rglob("*"):
        if not path.is_file() or skip.intersection(path.relative_to(root).parts):
            continue
        content = path.read_bytes()
        if path.suffix == ".svg":
            # Embedded font encodings can coincidentally resemble key prefixes.
            # Scan their decoded bytes, plus the SVG text, rather than treating
            # the transport encoding as a literal credential.
            content = re.sub(
                rb"data:font/(?:ttf|otf|woff2?);base64,([A-Za-z0-9+/=]+)",
                lambda match: base64.b64decode(match.group(1), validate=True),
                content,
            )
        require(
            not any(re.search(p, content) for p in SECRET_PATTERNS),
            f"possible credential in {path.relative_to(root)}",
        )
        # Research text may discuss filesystem paths; known private locations
        # must not survive even inside saved records or binary metadata.
        require(
            not MACHINE_PATH.search(content), f"private machine path in {path.relative_to(root)}"
        )
        require(path.stat().st_size < 100 * 2**20, f"oversized Git file: {path.relative_to(root)}")
        scanned += 1
    print(f"PASS: scanned {scanned} release files for credential patterns and private paths")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_results(root):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    prompts = json.loads((root / "data/prompts.json").read_text())
    require(len(prompts) == 200 and len(set(prompts)) == 200, "expected 200 unique prompts")
    totals = {"generations": 0, "paraphrases": 0, "score_arrays": 0}
    for model in MODELS:
        sweep = root / "data" / model / "sweep"
        reference_tags = None
        for attack in ("pegasus", "dipper"):
            suffix = "" if attack == "pegasus" else "_dipper"
            rows = list(csv.DictReader((sweep / f"sweep_results{suffix}.csv").open()))
            tags = {r["tag"] for r in rows}
            require(len(rows) == len(tags) == 25, f"{model}/{attack}: expected 25 operating points")
            if reference_tags is not None:
                require(tags == reference_tags, "attacks cover different operating points")
            reference_tags = tags
            for row in rows:
                tag = row["tag"]
                require(int(row["n_seeds"]) == 3, f"{tag}: expected three seeds")
                for seed in range(3):
                    generation = read_jsonl(sweep / f"gen_{tag}_s{seed}.jsonl")
                    paraphrase = read_jsonl(sweep / f"para{suffix}_{tag}_s{seed}.jsonl")
                    require(
                        len(generation) == len(paraphrase) == 200,
                        f"{model}/{tag}/s{seed}: incomplete rows",
                    )
                    for idx, (g, p) in enumerate(zip(generation, paraphrase)):
                        require(g["idx"] == p["idx"] == idx, f"{tag}: row indices differ")
                        require(
                            g["prompt"] == p["prompt"] == prompts[idx], f"{tag}: prompts differ"
                        )
                        require(
                            g["completion"] == p["completion"], f"{tag}: source completion differs"
                        )
                        require(isinstance(p["paraphrased"], str), f"{tag}: missing paraphrase")
                    totals["paraphrases"] += len(paraphrase)
                    if attack == "pegasus":
                        totals["generations"] += len(generation)
                if tag == "unsteered":
                    continue
                with np.load(
                    sweep / f"scores{suffix}" / f"scores_{tag}.npz", allow_pickle=False
                ) as scores:
                    y, score, group = (scores[k] for k in ("y", "score", "group"))
                    require(
                        y.shape == score.shape == group.shape == (1200,),
                        f"{tag}: unexpected score shape",
                    )
                    require(np.isfinite(score).all(), f"{tag}: non-finite detector scores")
                    require(
                        np.array_equal(y, np.repeat([1, 0], 600)), f"{tag}: unexpected class order"
                    )
                    require(
                        np.array_equal(group, np.tile(np.arange(200), 6)),
                        f"{tag}: unexpected prompt groups",
                    )
                    auc = roc_auc_score(y, score)
                    require(
                        math.isclose(auc, float(row["auc_post"]), abs_tol=1e-12),
                        f"{tag}: raw AUC disagrees",
                    )
                    require(
                        math.isclose(max(auc, 1 - auc), float(row["auc_post_sym"]), abs_tol=1e-12),
                        f"{tag}: symmetric AUC disagrees",
                    )
                totals["score_arrays"] += 1
        print(f"PASS: {model}: both attacks, 25 points, 3 seeds, 200 prompts")
    print("PASS:", ", ".join(f"{count:,} {label}" for label, count in totals.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integrity-only", action="store_true", help="no third-party dependencies needed"
    )
    args = parser.parse_args()
    verify_integrity(ROOT)
    scan_repository(ROOT)
    if not args.integrity_only:
        verify_results(ROOT)


if __name__ == "__main__":
    main()
