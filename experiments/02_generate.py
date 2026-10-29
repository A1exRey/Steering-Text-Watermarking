"""Stage 2: generate completions under four conditions.

Conditions:
  - unsteered   : plain generation, baseline/negative class
  - formality   : steering with mean-diff formality direction
  - random      : steering with random orthogonal direction (control for H2)
  - kirchenbauer: HF WatermarkingConfig (green-list) baseline for H3

Output: artifacts/generations.jsonl (one record per prompt per condition)
"""

import json
import random

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import WatermarkingConfig

import config
from utils import apply_chat, load_llama, steering_hook


def load_prompts(n):
    """Use Alpaca instructions (no input) as prompts; they're varied and clean."""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    # Deterministic slice; reserve the first N_DIRECTION_PAIRS*2 for direction stage
    offset = config.N_DIRECTION_PAIRS * 2
    prompts = []
    for ex in ds.select(range(offset, offset + n * 3)):
        if not ex["input"]:
            prompts.append(ex["instruction"])
        if len(prompts) >= n:
            break
    assert len(prompts) == n, f"Only got {len(prompts)} prompts, need {n}"
    return prompts


def format_prompt(tok, instruction, device):
    msgs = [{"role": "user", "content": instruction}]
    return apply_chat(tok, msgs, device)


def build_kgw_config():
    return WatermarkingConfig(
        greenlist_ratio=config.KGW_GAMMA,
        bias=config.KGW_BIAS,
        hashing_key=config.KGW_SEED,
        seeding_scheme=config.KGW_SCHEME,
        context_width=1,
    )


def generate_one(model, tok, prompt, condition, directions, kgw_cfg, gen_kwargs):
    """Run a single-prompt generation under the given condition."""
    inputs = format_prompt(tok, prompt, model.device)
    prompt_len = inputs["input_ids"].shape[1]

    kwargs = dict(gen_kwargs)
    if condition == "kirchenbauer":
        kwargs["watermarking_config"] = kgw_cfg

    if condition in ("formality", "random"):
        direction = directions[condition]
        with steering_hook(model, config.STEER_LAYER, direction, config.STEER_MULTIPLIER):
            with torch.no_grad():
                out = model.generate(**inputs, **kwargs)
    else:
        with torch.no_grad():
            out = model.generate(**inputs, **kwargs)

    completion = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return completion


def sanity_sample(model, tok, prompt, directions, kgw_cfg, gen_kwargs):
    """Print one generation per condition for a single prompt.
    Run before the full sweep to catch catastrophic steering magnitudes."""
    print("\n--- sanity samples ---")
    print(f"prompt: {prompt[:120]}...")
    for cond in ["unsteered", "formality", "random", "kirchenbauer"]:
        out = generate_one(model, tok, prompt, cond, directions, kgw_cfg, gen_kwargs)
        print(f"\n[{cond}]")
        print(out[:300])
    print("\n--- end sanity samples ---\n")


def main():
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    dirs_blob = torch.load(config.DIR_PATH)
    directions = {
        "formality": dirs_blob["formality"],
        "random": dirs_blob["random_ortho"],
    }

    model, tok = load_llama(config.GEN_MODEL)
    prompts = load_prompts(config.N_GEN_PROMPTS)

    kgw_cfg = build_kgw_config()
    gen_kwargs = dict(
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=True,
        temperature=config.GEN_TEMPERATURE,
        top_p=config.GEN_TOP_P,
        pad_token_id=tok.eos_token_id,
    )

    # Sanity-check one prompt across all conditions before the full sweep
    sanity_sample(model, tok, prompts[0], directions, kgw_cfg, gen_kwargs)

    all_records = []
    for condition in ["unsteered", "formality", "random", "kirchenbauer"]:
        for i, prompt in enumerate(tqdm(prompts, desc=f"gen:{condition}")):
            completion = generate_one(
                model, tok, prompt, condition, directions, kgw_cfg, gen_kwargs
            )
            all_records.append(
                {
                    "idx": i,
                    "prompt": prompt,
                    "condition": condition,
                    "completion": completion,
                }
            )

    with open(config.GEN_PATH, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(all_records)} records to {config.GEN_PATH}")


if __name__ == "__main__":
    main()
