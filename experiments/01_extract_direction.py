"""Stage 1: extract a formality steering direction + a random orthogonal control.

Method:
1. Use a larger model (Llama-3.1-8B-Instruct) to generate paired formal/informal
   rewrites of neutral text. This avoids depending on external formality corpora
   with gating issues (GYAFC) and is fully reproducible.
2. Re-encode each pair through the *generation* model (Llama-3.2-1B-Instruct),
   capturing residual-stream activations at STEER_LAYER.
3. Formality direction = mean(formal_acts) - mean(informal_acts), unit-normalized.
4. Random orthogonal direction: Gaussian, then Gram-Schmidt vs formality direction.

Output: artifacts/pairs.jsonl (the contrastive pairs, for audit)
        artifacts/directions.pt (the two directions + metadata)
"""

import json
import random

import torch
from datasets import load_dataset
from tqdm import tqdm

import config
from utils import activation_capture, apply_chat, batched, load_llama, masked_mean


FORMAL_PROMPT = (
    "Rewrite the following text in a highly formal, professional register. "
    "Preserve the meaning exactly. Output only the rewritten text, with no preamble.\n\n"
    "Text: {text}\n\nRewritten:"
)

INFORMAL_PROMPT = (
    "Rewrite the following text in a very casual, conversational register with "
    "contractions and everyday language. Preserve the meaning exactly. Output "
    "only the rewritten text, with no preamble.\n\n"
    "Text: {text}\n\nRewritten:"
)


def chat_generate(model, tok, user_msg, max_new_tokens=200):
    msgs = [{"role": "user", "content": user_msg}]
    inputs = apply_chat(tok, msgs, model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def generate_contrastive_pairs(n_pairs):
    """Synthesize n_pairs formal/informal rewrites using the larger model."""
    print(f"Loading direction model: {config.DIRECTION_MODEL}")
    model, tok = load_llama(config.DIRECTION_MODEL)

    # Use alpaca outputs as seed text: varied register, reasonable length
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    seeds = []
    for ex in ds:
        txt = ex["output"]
        if 30 < len(txt) < 300:
            seeds.append(txt)
        if len(seeds) >= n_pairs:
            break

    pairs = []
    for seed in tqdm(seeds, desc="contrastive pairs"):
        formal = chat_generate(model, tok, FORMAL_PROMPT.format(text=seed))
        informal = chat_generate(model, tok, INFORMAL_PROMPT.format(text=seed))
        # Light filtering: skip if the model refused or output something trivial
        if len(formal) < 10 or len(informal) < 10:
            continue
        pairs.append({"seed": seed, "formal": formal, "informal": informal})

    del model, tok
    torch.cuda.empty_cache()
    print(f"Retained {len(pairs)} / {n_pairs} pairs.")
    return pairs


def pool_acts(model, tok, texts, layer_idx, batch_size=8, max_length=256):
    """Return mean-pooled residual-stream activations at layer_idx, shape (N, D)."""
    pooled_batches = []
    for batch in tqdm(list(batched(texts, batch_size)), desc="pool acts"):
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        store = {}
        with activation_capture(model, layer_idx, store):
            with torch.no_grad():
                model(**enc)
        pooled = masked_mean(store["acts"].float(), enc["attention_mask"])
        pooled_batches.append(pooled.cpu())
    return torch.cat(pooled_batches, dim=0)


def extract_directions(pairs):
    """Compute formality direction (mean-diff) and a random orthogonal direction."""
    print(f"Loading generation model for direction extraction: {config.GEN_MODEL}")
    model, tok = load_llama(config.GEN_MODEL)

    formal_acts = pool_acts(model, tok, [p["formal"] for p in pairs], config.STEER_LAYER)
    informal_acts = pool_acts(model, tok, [p["informal"] for p in pairs], config.STEER_LAYER)

    diff = formal_acts.mean(0) - informal_acts.mean(0)
    formality_dir = diff / diff.norm()

    # Random orthogonal direction: Gaussian, then project out formality component
    g = torch.Generator().manual_seed(config.SEED)
    rand = torch.randn(formality_dir.shape, generator=g)
    rand = rand - (rand @ formality_dir) * formality_dir
    random_dir = rand / rand.norm()

    print(f"formality_dir: norm={formality_dir.norm():.4f}, dim={tuple(formality_dir.shape)}")
    print(f"random_dir   : norm={random_dir.norm():.4f}")
    print(f"dot(formality, random) = {(formality_dir @ random_dir).item():.2e}  (should be ~0)")
    print(f"raw diff magnitude before normalization: {diff.norm():.4f}")

    del model, tok
    torch.cuda.empty_cache()
    return formality_dir, random_dir


def main():
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    pairs = generate_contrastive_pairs(config.N_DIRECTION_PAIRS)
    with open(config.PAIRS_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} pairs to {config.PAIRS_PATH}")

    formality_dir, random_dir = extract_directions(pairs)
    torch.save(
        {
            "formality": formality_dir,
            "random_ortho": random_dir,
            "layer": config.STEER_LAYER,
            "gen_model": config.GEN_MODEL,
            "n_pairs": len(pairs),
        },
        config.DIR_PATH,
    )
    print(f"Wrote directions to {config.DIR_PATH}")


if __name__ == "__main__":
    main()
