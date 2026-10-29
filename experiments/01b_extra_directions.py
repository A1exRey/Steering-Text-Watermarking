"""Extend directions.pt with extra semantic directions.

The mechanistic claim ("detection tracks distortion, not semantics") is far
stronger with several unrelated semantic axes than with formality alone: if
formality, sentiment, AND complexity directions all fall on the same
AUC-vs-perplexity line as the random control, "semantics doesn't matter" is
hard to argue against. Two points (formality + random) cannot establish that.

This reuses the exact extraction method of 01_extract_direction.py (mean-diff
of pooled activations over contrastive rewrites), just with different contrast
prompts, and APPENDS the new directions to directions.pt. 06_pareto_sweep.py
treats every non-"random_ortho" tensor as a semantic curve, so no other code
needs to change.

Run AFTER 01_extract_direction.py, BEFORE 06_pareto_sweep.py:
    python 01b_extra_directions.py
"""

import json

import torch
from datasets import load_dataset
from tqdm import tqdm

import config
from utils import activation_capture, batched, load_llama, masked_mean

# Each axis: a name, plus a "high" and "low" rewrite instruction. Direction is
# mean(high_acts) - mean(low_acts), matching the formality convention.
AXES = {
    "sentiment": {
        "high": "Rewrite the following text with a strongly positive, upbeat, enthusiastic tone. "
        "Preserve the factual meaning. Output only the rewritten text.\n\nText: {text}\n\nRewritten:",
        "low": "Rewrite the following text with a strongly negative, critical, pessimistic tone. "
        "Preserve the factual meaning. Output only the rewritten text.\n\nText: {text}\n\nRewritten:",
    },
    "complexity": {
        "high": "Rewrite the following text using sophisticated, complex vocabulary and long, "
        "subordinated sentences. Preserve the meaning. Output only the rewritten text.\n\nText: {text}\n\nRewritten:",
        "low": "Rewrite the following text using very simple words and short sentences a child "
        "could read. Preserve the meaning. Output only the rewritten text.\n\nText: {text}\n\nRewritten:",
    },
}


def chat_generate(model, tok, user_msg, max_new_tokens=200):
    from utils import apply_chat

    inputs = apply_chat(tok, [{"role": "user", "content": user_msg}], model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def pool_acts(model, tok, texts, layer, bs=8, max_length=256):
    out = []
    for batch in tqdm(list(batched(texts, bs)), desc="pool"):
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model.device)
        store = {}
        with activation_capture(model, layer, store):
            with torch.no_grad():
                model(**enc)
        out.append(masked_mean(store["acts"].float(), enc["attention_mask"]).cpu())
    return torch.cat(out, 0)


def main():
    torch.manual_seed(config.SEED)
    blob = torch.load(config.DIR_PATH)
    formality = blob["formality"]  # for orthogonality reporting

    # seed texts: same source/slice convention as stage 1
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    seeds = []
    for ex in ds:
        t = ex["output"]
        if 30 < len(t) < 300:
            seeds.append(t)
        if len(seeds) >= config.N_DIRECTION_PAIRS:
            break

    big, btok = load_llama(config.DIRECTION_MODEL)
    pairs_by_axis = {ax: [] for ax in AXES}
    for ax, prompts in AXES.items():
        for seed in tqdm(seeds, desc=f"pairs:{ax}"):
            hi = chat_generate(big, btok, prompts["high"].format(text=seed))
            lo = chat_generate(big, btok, prompts["low"].format(text=seed))
            if len(hi) > 10 and len(lo) > 10:
                pairs_by_axis[ax].append((hi, lo))
    del big, btok
    torch.cuda.empty_cache()

    # The recovered run retained the vectors but not these extra contrast pairs.
    # Preserve them for auditing all future direction-extraction runs.
    with open(config.ART / "extra_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs_by_axis, f, ensure_ascii=False, indent=2)
        f.write("\n")

    model, tok = load_llama(config.GEN_MODEL)
    for ax, pairs in pairs_by_axis.items():
        hi_acts = pool_acts(model, tok, [p[0] for p in pairs], config.STEER_LAYER)
        lo_acts = pool_acts(model, tok, [p[1] for p in pairs], config.STEER_LAYER)
        diff = hi_acts.mean(0) - lo_acts.mean(0)
        d = diff / diff.norm()
        blob[ax] = d
        cos = float(d @ (formality / formality.norm()))
        print(
            f"{ax}: dim={tuple(d.shape)}  cos(formality)={cos:+.3f}  raw_diff_norm={diff.norm():.3f}"
        )

    torch.save(blob, config.DIR_PATH)
    print(f"\nAppended {list(AXES.keys())} to {config.DIR_PATH}")
    print(f"directions.pt now holds: {[k for k, v in blob.items() if torch.is_tensor(v)]}")


if __name__ == "__main__":
    main()
