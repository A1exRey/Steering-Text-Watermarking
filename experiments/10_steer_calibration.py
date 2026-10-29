"""Verify steering actually changes generations, and calibrate the multiplier.

The confound diagnostic showed steered == unsteered text on Qwen3-1.7B for every
method/multiplier, including sentiment m3. The hook mechanism itself works on
plain-tensor layer outputs (verified separately), so the most likely cause is
MAGNITUDE: a unit-norm direction times STEER_MULTIPLIER is negligible relative
to Qwen's residual-stream norm at the steered layer, so it never changes a
sampled token. STEER_MULTIPLIER does not transfer across models.

This script:
  1. Measures the typical residual-stream L2 norm at STEER_LAYER (per token),
     so we know the scale a steering vector must reach to matter.
  2. Runs a tiny generation A/B: same prompt+seed, hook OFF vs hook ON at several
     multipliers, printing the outputs. If outputs are identical at the current
     multiplier but diverge at larger ones, magnitude was the bug.
  3. Suggests a multiplier on the order of (residual_norm fraction).

Run: AOS_MODEL=qwen3-1.7b python 10_steer_calibration.py
"""

import torch

import config
from utils import activation_capture, apply_chat, load_llama, steering_hook


def measure_resid_norm(model, tok, prompts, layer):
    """Mean per-token L2 norm of the residual stream at `layer`."""
    norms = []
    for p in prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        store = {}
        with activation_capture(model, layer, store):
            with torch.no_grad():
                model(**enc)
        h = store["acts"].float()[0]  # (T, D)
        norms.append(h.norm(dim=-1).mean().item())
    return sum(norms) / len(norms)


def gen_once(model, tok, prompt, gen_kwargs):
    inputs = apply_chat(tok, [{"role": "user", "content": prompt}], model.device)
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return tok.decode(out[0][plen:], skip_special_tokens=True).strip()


def main():
    model, tok = load_llama(config.GEN_MODEL)
    layer = config.STEER_LAYER

    sample_prompts = [
        "Explain how a refrigerator works.",
        "Write a short note inviting a colleague to lunch.",
        "What makes a good password?",
    ]

    rnorm = measure_resid_norm(model, tok, sample_prompts, layer)
    print(f"model={config.GEN_MODEL} layer={layer}")
    print(f"mean residual-stream L2 norm @ layer {layer}: {rnorm:.1f}")
    print(
        f"STEER_MULTIPLIER={config.STEER_MULTIPLIER} is now a NORM-RELATIVE "
        f"fraction: perturbation = {config.STEER_MULTIPLIER:.0%} of each token's "
        f"residual norm.\n"
    )

    # build a direction to test (formality if available, else random)
    blob = torch.load(config.DIR_PATH)
    direction = blob.get("formality")
    if direction is None:
        d = torch.randn(model.config.hidden_size)
        direction = d / d.norm()
    direction = direction / direction.norm()

    gen_kwargs = dict(
        max_new_tokens=60,
        do_sample=True,
        temperature=config.GEN_TEMPERATURE,
        top_p=config.GEN_TOP_P,
        pad_token_id=tok.eos_token_id,
    )

    prompt = sample_prompts[0]
    # Norm-relative fractions to test (the hook now scales by ||h|| internally).
    # Extended upward because per-token scaling stays fluent much further than
    # the old absolute hook -- we need to find where it actually degrades.
    test_fracs = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]

    print("A/B generations (same prompt, same seed, varying norm-relative fraction):\n")
    torch.manual_seed(0)
    base = gen_once(model, tok, prompt, gen_kwargs)
    print(f"[hook OFF] {base!r}\n")

    for frac in test_fracs:
        torch.manual_seed(0)
        with steering_hook(model, layer, direction, frac, norm_relative=True):
            out = gen_once(model, tok, prompt, gen_kwargs)
        same = "IDENTICAL to OFF" if out == base else "changed"
        print(f"[frac={frac:5.2f}] {same}")
        print(f"           {out[:160]!r}\n")

    print("=" * 64)
    print("READING:")
    print("  * Pick the largest fraction that still yields FLUENT but CHANGED text.")
    print("    That single fraction should now work for BOTH models because the")
    print("    hook scales by the per-token residual norm internally.")
    print("  * If text degrades into word-salad at some fraction (e.g. 0.40),")
    print("    keep the sweep grid below it.")


if __name__ == "__main__":
    main()
