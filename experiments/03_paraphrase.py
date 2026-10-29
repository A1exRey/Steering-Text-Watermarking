"""Stage 3: paraphrase all generations with Pegasus (sentence-by-sentence).

Pegasus has a 60-token input cap, so we split generations into sentences,
paraphrase each, and re-join. This is the standard paraphrase attack used
in SemStamp / SemaMark / Kirchenbauer evaluations. For a full paper, also
run DIPPER and/or GPT-4 paraphrase; Pegasus is the cheap fast baseline.

Output: artifacts/paraphrased.jsonl (adds "paraphrased" field to each record)
"""

import json
import re

import torch
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

import config


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    parts = SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def paraphrase_sentences(model, tok, sentences, device, max_length=60):
    """Paraphrase a batch of sentences; returns a list of paraphrased strings."""
    if not sentences:
        return []
    enc = tok(
        sentences,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=max_length,
            num_beams=5,
            num_return_sequences=1,
            temperature=1.5,
        )
    return tok.batch_decode(out, skip_special_tokens=True)


def paraphrase_text(model, tok, text, device, batch_size=16):
    sents = split_sentences(text)
    if not sents:
        return text
    out = []
    for i in range(0, len(sents), batch_size):
        chunk = sents[i : i + batch_size]
        out.extend(paraphrase_sentences(model, tok, chunk, device))
    return " ".join(out)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading paraphraser: {config.PARAPHRASE_MODEL}")
    tok = PegasusTokenizer.from_pretrained(config.PARAPHRASE_MODEL)
    model = PegasusForConditionalGeneration.from_pretrained(config.PARAPHRASE_MODEL).to(device)
    model.eval()

    with open(config.GEN_PATH) as f:
        records = [json.loads(line) for line in f]

    out_records = []
    for r in tqdm(records, desc="paraphrase"):
        para = paraphrase_text(model, tok, r["completion"], device)
        out_records.append({**r, "paraphrased": para})

    with open(config.PARA_PATH, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(out_records)} paraphrased records to {config.PARA_PATH}")


if __name__ == "__main__":
    main()
