"""DIPPER paraphrase attack (Reviewer 1: Pegasus is a weak, sentence-level attacker).

Pegasus rewrites one sentence at a time with no cross-sentence context, so it
cannot reorder or merge material. DIPPER (Krishna et al., 2023) is an 11B T5
paraphraser that rewrites in windows of several sentences with the running
paraphrase as context, and takes explicit lexical- and order-diversity codes. It
is the standard strong attack in the watermarking literature.

This script only PARAPHRASES. It reuses the cached generations from
06_pareto_sweep.py, so no regeneration is needed:

    artifacts/<model>/sweep/gen_<tag>_s<seed>.jsonl        (input, existing)
    artifacts/<model>/sweep/para_dipper_<tag>_s<seed>.jsonl (output)

Run SemStamp BEFORE this. 16_semstamp.py writes its own gen_semstamp_s*.jsonl,
and those generations have to be paraphrased like every other operating point;
doing SemStamp first means DIPPER covers all 26 tags in a single pass instead of
being re-run for one tag later.

Then score it with the evaluate phase pointed at the new attack:

    AOS_MODEL=llama-1b python 06_pareto_sweep.py --attack dipper --eval-only
    python 12_gap_test.py --sweep artifacts/llama-1b/sweep --attack dipper

which writes sweep_results_dipper.csv alongside the Pegasus one, leaving the
existing results untouched so the two attacks can be compared directly.

Throughput
----------
Windows within a document are strictly sequential -- each one is conditioned on
the rolling paraphrase of the ones before it -- but documents are independent.
So the batching is across documents: keep `--batch` documents in flight and
generate window i of each of them in one forward pass, retiring documents as
they finish and refilling from the queue. Utilisation stays flat even though
documents have different sentence counts.

The model is ~11B, so bfloat16 needs roughly 22GB. On an 80GB card that leaves
~58GB for activations and KV cache; --batch 24 is comfortable, 32 works. On a
40GB card drop to --batch 8.

With two GPUs, run two processes rather than sharding the model across both:
the model fits on one card, so model parallelism would only add pipeline
bubbles. Split the (tag, seed) work units instead:

    CUDA_VISIBLE_DEVICES=0 python 15_dipper_attack.py --sweep ... --shard 0 --num-shards 2 &
    CUDA_VISIBLE_DEVICES=1 python 15_dipper_attack.py --sweep ... --shard 1 --num-shards 2 &
    wait

Each process writes disjoint files, so there is no coordination and either one
can be interrupted and resumed independently.

Fidelity note: prompt construction, the 100-minus-diversity code convention, the
sentence window and the rolling prefix follow the reference implementation
released with the paper. Defaults are lex=60, order=60, the setting most often
reported as the strong attack. Batching does not change any of that -- padding
is masked out, and sampling is per-sequence -- but it does change the order in
which the sampler consumes randomness, so batched output is not token-identical
to unbatched output at the same seed. DIPPER is stochastic either way.

Run it over every tag. 06 --attack refuses to score unless all 25 of its
operating points have a paraphrase, so --tags is for smoke tests only, not for
saving wall-clock on a real run.

Usage:
  python 15_dipper_attack.py --sweep artifacts/llama-1b/sweep
  python 15_dipper_attack.py --sweep artifacts/qwen3-1.7b/sweep --lex 60 --order 60
  # smoke test on one tag before committing the full run
  python 15_dipper_attack.py --sweep artifacts/llama-1b/sweep \
      --tags unsteered --batch 8
"""

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

SEEDS = [0, 1, 2]
MODEL_ID = "kalpeshk2011/dipper-paraphraser-xxl"
TOKENIZER_ID = "google/t5-v1_1-xxl"  # DIPPER ships no tokenizer of its own
VALID_CODES = {0, 20, 40, 60, 80, 100}
MAX_INPUT_TOKENS = 1024

_SENT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    """nltk if available (what the reference implementation uses), else regex."""
    try:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    except Exception:
        return [s for s in _SENT.split(text.strip()) if s.strip()]


class Dipper:
    def __init__(self, model_id=MODEL_ID, dtype=torch.bfloat16, device="cuda"):
        self.tok = T5Tokenizer.from_pretrained(TOKENIZER_ID)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map=device
        )
        self.model.eval()
        self.n_truncated = 0

    def build_input(self, prefix, window, lex_code, order_code):
        inp = f"lexical = {lex_code}, order = {order_code}"
        if prefix:
            inp += f" {prefix}"
        return inp + f" <sent> {window} </sent>"

    def run_batch(self, inputs, max_new_tokens, top_p, temperature):
        """One generate() call over a batch of independent windows.

        T5 is encoder-decoder, so the default right-padding is correct here --
        the encoder attention mask masks the pad positions out. (Left-padding is
        only needed for decoder-only models, where pads would otherwise sit
        between the prompt and the first generated token.)
        """
        enc = self.tok(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_TOKENS
        ).to(self.model.device)
        self.n_truncated += int((enc.attention_mask.sum(1) >= MAX_INPUT_TOKENS).sum())
        with torch.inference_mode():
            gen = self.model.generate(
                **enc,
                do_sample=True,
                top_p=top_p,
                top_k=None,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        return [s.strip() for s in self.tok.batch_decode(gen, skip_special_tokens=True)]


def paraphrase_documents(
    dipper, texts, lex, order, sent_interval, batch, max_new_tokens, top_p, temperature, desc
):
    """Paraphrase a list of documents, keeping `batch` of them in flight.

    Returns a list aligned with `texts`. Documents are refilled into the active
    set as they finish, so a document with two windows does not leave a hole in
    the batch for the rest of the pass.
    """
    assert lex in VALID_CODES and order in VALID_CODES, (
        f"diversity codes must be in {sorted(VALID_CODES)}"
    )
    lex_code, order_code = 100 - lex, 100 - order

    docs, results = [], [""] * len(texts)
    for i, text in enumerate(texts):
        text = " ".join(text.split())
        sents = split_sentences(text) if text else []
        windows = [
            " ".join(sents[j : j + sent_interval]) for j in range(0, len(sents), sent_interval)
        ]
        if windows:
            docs.append({"i": i, "windows": windows, "ptr": 0, "prefix": "", "out": []})

    pbar = tqdm(total=sum(len(d["windows"]) for d in docs), desc=desc, unit="win")
    active, queued = [], 0
    while active or queued < len(docs):
        while len(active) < batch and queued < len(docs):
            active.append(docs[queued])
            queued += 1

        pieces = dipper.run_batch(
            [
                dipper.build_input(d["prefix"], d["windows"][d["ptr"]], lex_code, order_code)
                for d in active
            ],
            max_new_tokens,
            top_p,
            temperature,
        )

        still_running = []
        for d, piece in zip(active, pieces):
            d["out"].append(piece)
            d["prefix"] = f"{d['prefix']} {piece}".strip()
            d["ptr"] += 1
            pbar.update(1)
            if d["ptr"] < len(d["windows"]):
                still_running.append(d)
            else:
                results[d["i"]] = " ".join(d["out"]).strip()
        active = still_running
    pbar.close()
    return results


def discover_tags(sweep_dir):
    tags = {
        re.sub(r"^gen_(.*)_s0\.jsonl$", r"\1", p.name) for p in sweep_dir.glob("gen_*_s0.jsonl")
    }
    if "unsteered" not in tags:
        raise SystemExit(f"no gen_unsteered_s0.jsonl in {sweep_dir}")
    return ["unsteered"] + sorted(t for t in tags if t != "unsteered")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="artifacts/<model>/sweep")
    ap.add_argument("--lex", type=int, default=60, help="lexical diversity")
    ap.add_argument("--order", type=int, default=60, help="order diversity")
    ap.add_argument("--sent-interval", type=int, default=3)
    ap.add_argument(
        "--tags",
        default=None,
        help="comma-separated subset; default is every tag. Note that "
        "06 --attack refuses to run unless every one of its 25 "
        "operating points has a paraphrase, so a subset here is "
        "only useful for a smoke test.",
    )
    ap.add_argument(
        "--name", default="dipper", help="output prefix, para_<name>_<tag>_s<seed>.jsonl"
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=24,
        help="documents in flight; 24 fits an 80GB card, 8 a 40GB one",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="per window, not per document; a 3-sentence window is "
        "well under this, and a batch runs until its slowest "
        "member stops, so an oversized cap costs real time",
    )
    ap.add_argument("--top-p", type=float, default=0.75)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="split (tag, seed) units across processes; pair with "
        "CUDA_VISIBLE_DEVICES to use one GPU each",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep)
    tags = args.tags.split(",") if args.tags else discover_tags(sweep_dir)
    if "unsteered" not in tags:
        # every AUC is measured against the unsteered negatives; without them
        # the attack cannot be scored at all.
        raise SystemExit("'unsteered' must be included in --tags")

    units = [(t, s) for s in SEEDS for t in tags]
    todo = [
        (t, s) for (t, s) in units if not (sweep_dir / f"para_{args.name}_{t}_s{s}.jsonl").exists()
    ]
    mine = todo[args.shard :: args.num_shards]
    print(
        f"{len(units)} (tag, seed) units; {len(todo)} to do, "
        f"{len(units) - len(todo)} already present"
    )
    if args.num_shards > 1:
        print(f"shard {args.shard}/{args.num_shards}: {len(mine)} units")
    if not mine:
        return

    torch.manual_seed(args.seed + args.shard)
    print(f"loading {MODEL_ID} (~11B, this takes a while)")
    dipper = Dipper()
    print(
        f"lex={args.lex} order={args.order} window={args.sent_interval} sentences "
        f"batch={args.batch}\n"
    )

    for tag, seed in mine:
        src = sweep_dir / f"gen_{tag}_s{seed}.jsonl"
        dst = sweep_dir / f"para_{args.name}_{tag}_s{seed}.jsonl"
        with open(src) as f:
            recs = [json.loads(line) for line in f]

        paras = paraphrase_documents(
            dipper,
            [r["completion"] for r in recs],
            args.lex,
            args.order,
            args.sent_interval,
            args.batch,
            args.max_new_tokens,
            args.top_p,
            args.temperature,
            desc=f"{tag} s{seed}",
        )

        tmp = dst.with_suffix(".jsonl.partial")
        with open(tmp, "w") as f:
            for r, para in zip(recs, paras):
                f.write(json.dumps({**r, "paraphrased": para}) + "\n")
        # rename only on success, so an interrupted run is retried rather than
        # silently skipped as "already present" on the next invocation
        tmp.rename(dst)

    if dipper.n_truncated:
        # the rolling prefix grows with each window, so a long enough document
        # can push the <sent> block itself past the encoder limit
        print(
            f"\nWARNING: {dipper.n_truncated} windows hit the "
            f"{MAX_INPUT_TOKENS}-token encoder limit and were truncated"
        )

    print(f"\nDone. Now score it:\n  python 06_pareto_sweep.py --attack {args.name} --eval-only")


if __name__ == "__main__":
    main()
