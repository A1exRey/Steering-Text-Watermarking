"""Shared configuration for the paraphrase-robust watermark experiment.

Model is selectable via the AOS_MODEL env var so the Llama and Qwen runs keep
SEPARATE artifact directories (no overwrites). Examples:

    python 06_pareto_sweep.py                 # default: llama-1b
    AOS_MODEL=qwen3-1.7b python 06_pareto_sweep.py
"""

import os
from pathlib import Path

# --- Model registry -----------------------------------------------------------
# Each entry pins: the generation/steering model, the (larger) model used to
# synthesize contrastive rewrites, the steered/probed layer (mid-depth), and a
# flag for Qwen-style thinking mode which MUST be disabled or it pollutes
# generations with <think> traces.
MODEL_REGISTRY = {
    "llama-1b": {
        "gen_model": "meta-llama/Llama-3.2-1B-Instruct",
        "direction_model": "meta-llama/Llama-3.1-8B-Instruct",
        "steer_layer": 8,  # of 16
        "qwen_thinking": False,  # n/a for llama
        # Norm-relative steering fractions. Per-model because each model's
        # degradation point differs; legitimacy comes from plotting against
        # measured PERPLEXITY (not the fraction), so curves remain comparable
        # in fluency space. Confirmed via 10_steer_calibration: 0.05 identical
        # to baseline (dropped); fluent-but-steered through ~0.40; degrading by
        # 0.60; broken by 0.80.
        "fractions": [0.10, 0.20, 0.40, 0.60, 0.80],
    },
    "qwen3-1.7b": {
        "gen_model": "Qwen/Qwen3-1.7B",
        "direction_model": "Qwen/Qwen3-8B",
        "steer_layer": 14,  # Qwen3-1.7B has 28 layers; mid-depth ~14
        "qwen_thinking": True,  # template supports thinking -> must disable
        # Per-token scaling stays fluent further on Qwen (still coherent at 0.40);
        # extend upward to reach the degradation regime. Confirm top end with
        # the extended calibration (tests up to 0.80).
        "fractions": [0.10, 0.20, 0.40, 0.60, 0.80],
    },
    "qwen3-4b": {
        "gen_model": "Qwen/Qwen3-4B",
        "direction_model": "Qwen/Qwen3-8B",
        "steer_layer": 18,  # Qwen3-4B has 36 layers
        "qwen_thinking": True,
        "fractions": [0.10, 0.20, 0.40, 0.60, 0.80],
    },
    "olmo3-7b": {
        "gen_model": "allenai/OLMo-3-7B-Instruct",
        "direction_model": "allenai/OLMo-3-7B-Instruct",  # self; no larger sibling needed
        "steer_layer": 16,  # OLMo-3-7B has 32 layers; mid-depth
        "qwen_thinking": False,
        "fractions": [0.10, 0.20, 0.40, 0.60, 0.80],
    },
    "llama-8b": {
        "gen_model": "meta-llama/Llama-3.1-8B-Instruct",
        "direction_model": "meta-llama/Llama-3.1-8B-Instruct",
        "steer_layer": 16,  # Llama-3.1-8B has 32 layers; mid-depth
        "qwen_thinking": False,
        "fractions": [0.10, 0.20, 0.40, 0.60, 0.80],
    },
}

ACTIVE = os.environ.get("AOS_MODEL", "llama-1b")
assert ACTIVE in MODEL_REGISTRY, f"AOS_MODEL must be one of {list(MODEL_REGISTRY)}"
_M = MODEL_REGISTRY[ACTIVE]

# --- Models ---
GEN_MODEL = _M["gen_model"]
DIRECTION_MODEL = _M["direction_model"]
PARAPHRASE_MODEL = "tuner007/pegasus_paraphrase"
# Whether the gen model uses a thinking-mode chat template that must be disabled.
QWEN_THINKING_DISABLE = _M["qwen_thinking"]

# --- Steering ---
STEER_LAYER = _M["steer_layer"]
# Norm-relative steering fraction (perturbation = STEER_MULTIPLIER * ||h|| * dir).
# ~0.10 gives fluent-but-changed text on both Llama and Qwen; >0.3 degrades.
STEER_MULTIPLIER = 0.10
# Per-model sweep grid of norm-relative fractions (see MODEL_REGISTRY).
STEER_FRACTIONS = _M["fractions"]

# --- Kirchenbauer baseline ---
KGW_GAMMA = 0.25
KGW_BIAS = 2.0
KGW_SEED = 42
KGW_SCHEME = "lefthash"

# --- Experiment scale ---
N_DIRECTION_PAIRS = 200
N_GEN_PROMPTS = 500
MAX_NEW_TOKENS = 150
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 0.9

# --- Paths (namespaced per active model so runs don't collide) ---
ROOT = Path(__file__).resolve().parents[1]
# New GPU runs are separate from the immutable, recovered paper data.
# Override with AOS_ARTIFACT_ROOT to resume a run at another location.
ARTIFACT_ROOT = (
    Path(os.environ.get("AOS_ARTIFACT_ROOT", ROOT / "outputs" / "runs")).expanduser().resolve()
)
ART = ARTIFACT_ROOT / ACTIVE
ART.mkdir(parents=True, exist_ok=True)
PROMPTS_PATH = Path(os.environ.get("AOS_PROMPTS_PATH", ROOT / "data" / "prompts.json")).expanduser()

PAIRS_PATH = ART / "pairs.jsonl"
DIR_PATH = ART / "directions.pt"
GEN_PATH = ART / "generations.jsonl"
PARA_PATH = ART / "paraphrased.jsonl"
RESULTS_PATH = ART / "results.json"

# --- Misc ---
SEED = 42
