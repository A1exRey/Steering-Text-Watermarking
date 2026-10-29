"""Shared utilities: model loading, forward hooks for steering / activation capture."""

from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llama(model_id, device="cuda", dtype=torch.bfloat16):
    """Load a Llama model + tokenizer with sensible defaults for generation and hooking."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"  # left-padding is correct for causal LM + batched generation
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,  # `torch_dtype` is deprecated in recent transformers
        device_map=device,
    )
    model.eval()
    return model, tok


def apply_chat(tok, messages, device):
    """Render a chat template into a BatchEncoding (input_ids + attention_mask) on device.

    In recent transformers, apply_chat_template(return_tensors="pt") returns a
    BatchEncoding dict, not a bare tensor. We force return_dict=True so the result
    is unambiguous and carries an attention_mask, then unpack with **inputs into
    model.generate(). Passing the BatchEncoding positionally is what triggered the
    `KeyError: 'shape'` / AttributeError.

    For Qwen3 models the chat template supports a thinking mode that emits
    <think>...</think> traces before the answer. That would pollute generations,
    perplexity, and steering, so we pass enable_thinking=False when the active
    model requires it (config.QWEN_THINKING_DISABLE). The kwarg is ignored by
    templates that don't define it, but we guard with a try/except to be safe
    across tokenizer versions.
    """
    import config

    kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    if getattr(config, "QWEN_THINKING_DISABLE", False):
        try:
            return tok.apply_chat_template(messages, enable_thinking=False, **kwargs).to(device)
        except TypeError:
            # template doesn't accept enable_thinking; fall through to default
            pass
    return tok.apply_chat_template(messages, **kwargs).to(device)


def get_layer(model, layer_idx):
    """Return the LlamaDecoderLayer at the given index."""
    return model.model.layers[layer_idx]


@contextmanager
def steering_hook(model, layer_idx, direction, multiplier, norm_relative=True):
    """Context manager: add a steering vector to the residual-stream output of
    the specified decoder layer.

    `direction` must be unit-normalized. Two scaling modes:

    - norm_relative=True (default): the added vector is
          multiplier * ||h_t|| * direction
      per token position t, i.e. `multiplier` is the perturbation magnitude as a
      FRACTION of each token's residual-stream norm. This makes `multiplier`
      comparable across models/layers despite very different absolute norms
      (e.g. Llama-3.2-1B @ L8 ~ tens, Qwen3-1.7B @ L14 ~ 2000). Sensible range is
      roughly 0.02-0.20; beyond that text degrades.

    - norm_relative=False: legacy absolute mode, adds `multiplier * direction`.
      Only meaningful if you've hand-tuned `multiplier` to a specific model.

    Works during .generate() with KV caching; handles both tuple and plain-tensor
    layer outputs (transformers 5.x decoder layers return a bare tensor).
    """
    direction = direction.to(device=model.device, dtype=model.dtype)

    def add_vec(hidden):
        if norm_relative:
            # per-token norm: hidden is (B, T, D); scale each position
            scale = hidden.norm(dim=-1, keepdim=True) * multiplier  # (B, T, 1)
            return hidden + scale * direction
        return hidden + multiplier * direction

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (add_vec(output[0]),) + output[1:]
        return add_vec(output)

    handle = get_layer(model, layer_idx).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def activation_capture(model, layer_idx, store):
    """Context manager: store the residual-stream output of the given layer
    into store['acts'] (shape: batch, seq, d_model)."""

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        store["acts"] = hidden.detach()

    handle = get_layer(model, layer_idx).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def masked_mean(h, attention_mask):
    """Mean-pool hidden states over non-padding positions.

    h: (B, T, D) float tensor
    attention_mask: (B, T) {0,1} tensor
    returns: (B, D)
    """
    mask = attention_mask.unsqueeze(-1).to(h.dtype)
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def batched(seq, n):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def batch_chat_generate(model, tok, instructions, gen_kwargs, batch_size=16):
    """Batched chat generation. Returns list of completion strings (prompts stripped).

    Relies on left-padding (set in load_llama): with left padding every sequence's
    generated tokens begin at input_ids.shape[1], so slicing is uniform across the
    batch. A steering hook, if active in the caller's context, applies to all
    positions in the batch identically.
    """
    out_texts = []
    import config

    extra = {}
    if getattr(config, "QWEN_THINKING_DISABLE", False):
        extra["enable_thinking"] = False
    for chunk in batched(instructions, batch_size):
        msgs_batch = [[{"role": "user", "content": x}] for x in chunk]
        try:
            enc = tok.apply_chat_template(
                msgs_batch,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                **extra,
            ).to(model.device)
        except TypeError:
            enc = tok.apply_chat_template(
                msgs_batch,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
            ).to(model.device)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        for row in out:
            out_texts.append(tok.decode(row[prompt_len:], skip_special_tokens=True).strip())
    return out_texts
