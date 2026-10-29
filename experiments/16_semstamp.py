"""SemStamp-style sentence-embedding watermark (Reviewer 3's missing baseline).

Reviewer 3's central objection: we compare only against a logit watermark, while
the related-work section names sentence-embedding watermarks as the approach that
shares our goal -- paraphrase robustness by operating on meaning rather than
tokens. This implements that baseline so the comparison can be made.

Algorithm (Hou et al., 2024)
----------------------------
Partition sentence-embedding space with `--bits` random hyperplanes seeded by the
key, giving 2^bits LSH regions. Generation proceeds sentence by sentence:

  1. the signature of the PREVIOUS sentence seeds a PRF that marks a gamma
     fraction of regions "valid" for the next one;
  2. candidate sentences are sampled from the LM, embedded, and accepted only if
     their region is valid AND their embedding is at least `--margin` from every
     hyperplane (the margin is what buys robustness: a sentence near a boundary
     can be flipped by a paraphrase that barely changes meaning);
  3. after `--max-trials` failures the best-margin candidate is accepted, so
     generation always terminates.

Detection needs no probe and no training: split the candidate text, embed, and
count how many sentences land in the region their predecessor licensed. Under the
null that is Binomial(n, gamma), giving a one-proportion z-test. We report AUC of
that z-score, so the number is directly comparable to the KGW column.

FIDELITY -- read before citing any number from this
---------------------------------------------------
This follows the algorithm as described, but it is a reimplementation, and one
deviation matters. SemStamp uses a sentence encoder CONTRASTIVELY FINE-TUNED so
that paraphrases embed close together; that fine-tuning is a substantial part of
why the mark survives paraphrase. `--encoder` defaults to a general-purpose
SBERT model, which will UNDERSTATE SemStamp. Since the paper's claim is that
activation steering loses to existing methods, understating a baseline biases the
comparison in our own favour -- the direction of error that most needs guarding
against. Use the authors' released encoder via `--encoder` for any number that
goes in the paper, and state in the text which encoder was used.

Two smaller deviations, both noted for completeness: sentence-level rejection
sampling is done by sampling `--candidates` continuations in parallel and taking
the first that passes, rather than sequentially; and sentence splitting uses nltk
when present, else a regex.

Throughput. Sentences within a prompt are sequential -- each one's valid-region
set is seeded by its predecessor's signature -- but prompts are independent, so
generation batches over prompts: `--batch-prompts` of them in flight, each
drawing `--candidates` continuations, giving one generate() call of
batch_prompts x candidates sequences per step and one embedder call over all
the candidates. Batching changes the order in which the sampler consumes
randomness, so output is not token-identical to a per-prompt run at the same
seed; the watermark itself is unaffected, since acceptance depends only on a
candidate's own embedding.

Run this BEFORE 15_dipper_attack.py. Generation here writes
gen_semstamp_s<seed>.jsonl, which DIPPER then has to paraphrase along with every
other operating point; doing it in the other order means re-running the 11B
paraphraser for one extra tag.

Usage:
  # generate watermarked text on the same prompts as the sweep. AOS_MODEL picks
  # the generator, exactly as it does for 06, and also drives the Qwen
  # thinking-mode fix -- so use it rather than --model.
  AOS_MODEL=llama-1b python 16_semstamp.py generate --sweep artifacts/llama-1b/sweep

  # after DIPPER has paraphrased everything, score under that attack
  AOS_MODEL=llama-1b python 16_semstamp.py detect --sweep artifacts/llama-1b/sweep \
      --field paraphrased --attack dipper
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config
from utils import apply_chat, load_llama

SEEDS = [0, 1, 2]
CONFIG_NAME = "semstamp_config.json"
_SENT = re.compile(r"(?<=[.!?])\s+")


def settings(args):
    """The knobs that MUST agree between generation and detection.

    The LSH partition is defined in the encoder's embedding space and seeded by
    (bits, gamma, key). Detecting under a different encoder scores signatures
    the generator never targeted, which does not error -- it just returns
    AUC ~ 0.5 and looks like a negative result. Recorded at generation time and
    checked at detection time so that failure is loud instead of silent.
    """
    return {"encoder": args.encoder, "bits": args.bits, "gamma": args.gamma, "key": args.key}


def split_sentences(text):
    try:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    except Exception:
        return [s for s in _SENT.split(text.strip()) if s.strip()]


# --------------------------------------------------------------------------
# LSH partition
# --------------------------------------------------------------------------
class LSHPartition:
    """Random-hyperplane LSH over sentence embeddings, plus the valid-region PRF."""

    def __init__(self, dim, bits, key, gamma):
        rng = np.random.default_rng(key)
        # rows are unit normals; sign of the projection gives one signature bit
        H = rng.normal(size=(bits, dim))
        self.H = H / np.linalg.norm(H, axis=1, keepdims=True)
        self.bits, self.key, self.gamma = bits, key, gamma
        self.n_regions = 2**bits
        # bit 0 of the signature is the FIRST hyperplane, i.e. the MSB
        self._weights = 1 << np.arange(bits - 1, -1, -1, dtype=np.int64)
        self._valid_cache = {}

    def signatures_and_margins(self, embs):
        """Vectorised over a batch of unit-norm embeddings.

        Returns (region indices, distance to nearest hyperplane) as arrays. The
        scalar path below delegates here so the two can never disagree about
        bit order -- generation and detection must assign identical signatures
        or the watermark simply will not verify.
        """
        embs = np.atleast_2d(np.asarray(embs, dtype=np.float64))
        proj = embs @ self.H.T  # (n, bits)
        bits = (~np.signbit(proj)).astype(np.int64)  # matches `int(not signbit)`
        return (bits * self._weights).sum(1), np.abs(proj).min(1)

    def signature_and_margin(self, emb):
        """(region index, distance to the nearest hyperplane) for unit-norm emb."""
        sigs, margins = self.signatures_and_margins(emb)
        return int(sigs[0]), float(margins[0])

    def valid_regions(self, prev_sig):
        """Regions licensed for the next sentence, given the previous signature.

        Keyed by (key, prev_sig) through a hash, so the verifier can reproduce the
        set exactly and an attacker without the key cannot.

        Memoised: this is a pure function of (key, prev_sig) and there are only
        n_regions distinct inputs, but it is called once per candidate sentence
        during generation and once per sentence during detection.
        """
        prev_sig = int(prev_sig)
        hit = self._valid_cache.get(prev_sig)
        if hit is None:
            seed = int.from_bytes(
                hashlib.sha256(f"{self.key}:{prev_sig}".encode()).digest()[:8], "big"
            )
            rng = np.random.default_rng(seed)
            k = max(1, int(round(self.gamma * self.n_regions)))
            hit = set(rng.choice(self.n_regions, size=k, replace=False).tolist())
            self._valid_cache[prev_sig] = hit
        return hit


class Embedder:
    def __init__(self, name, device="cuda", batch_size=256):
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(name, device=device)
        self.batch_size = batch_size

    def __call__(self, sents):
        if not len(sents):
            return np.zeros((0, self.m.get_sentence_embedding_dimension()))
        return self.m.encode(
            sents,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def _first_sentence(text):
    sents = split_sentences(text)
    return sents[0].strip() if sents else ""


def generate_prompts(
    model,
    tok,
    embedder,
    lsh,
    prompts,
    max_sentences,
    max_new_tokens,
    candidates,
    max_trials,
    margin,
    temperature,
    top_p,
    batch_prompts,
    max_words,
    desc,
):
    """Rejection-sample a watermarked continuation for each prompt.

    Sentences within a prompt are strictly sequential -- each one's valid-region
    set is seeded by its predecessor's signature -- but prompts are independent,
    so the batch axis is PROMPTS, not sentences. Each round issues one
    generate() call of `active x candidates` sequences and one embedder call
    over all the resulting candidate sentences. The earlier per-prompt version
    had an effective batch of `candidates` (8), which left an 80GB card at
    roughly 4GB and idle; here the batch is batch_prompts x candidates.

    Prompts are refilled from the queue as they retire, so short completions do
    not leave holes in the batch.
    """
    # Chat-format each prompt once. These prefixes stay fixed and accepted
    # sentences are appended to them: re-rendering the template around the
    # accumulated text would drop the assistant turn after the first sentence.
    base = [
        apply_chat(tok, [{"role": "user", "content": p}], model.device)["input_ids"][0].tolist()
        for p in prompts
    ]

    states = [
        {"i": i, "base": base[i], "text": "", "prev_sig": 0, "n_sent": 0, "trials": 0, "best": None}
        for i in range(len(prompts))
    ]
    results = [""] * len(prompts)

    pbar = tqdm(total=len(prompts), desc=desc, unit="prompt")

    def finish(st):
        results[st["i"]] = st["text"]
        pbar.update(1)

    active, queued, n_accept, n_fallback = [], 0, 0, 0
    while active or queued < len(states):
        while len(active) < batch_prompts and queued < len(states):
            active.append(states[queued])
            queued += 1

        rows = []
        for st in active:
            ids = st["base"] + (
                tok(st["text"], add_special_tokens=False).input_ids if st["text"] else []
            )
            rows.extend([ids] * candidates)
        enc = tok.pad({"input_ids": rows}, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
        # load_llama sets padding_side="left", so every row shares one prompt
        # width and the continuations slice off at a single offset
        conts = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        sents = [_first_sentence(c) for c in conts]

        keep = [j for j, s in enumerate(sents) if s]
        sigs, margins = lsh.signatures_and_margins(embedder([sents[j] for j in keep]))
        scored = {j: (int(sigs[k]), float(margins[k])) for k, j in enumerate(keep)}

        still = []
        for k, st in enumerate(active):
            valid = lsh.valid_regions(st["prev_sig"])
            chosen = None
            for j in range(k * candidates, (k + 1) * candidates):
                if j not in scored:
                    continue
                sig, m = scored[j]
                if st["best"] is None or m > st["best"][0]:
                    st["best"] = (m, sents[j], sig)
                if sig in valid and m >= margin:
                    chosen = (m, sents[j], sig)
                    break

            st["trials"] += 1
            fell_back = False
            if chosen is None:
                if st["trials"] < max_trials:
                    still.append(st)  # same sentence position, another draw
                    continue
                # exhausted: fall back to the widest-margin candidate seen, so
                # generation always terminates
                chosen, fell_back = st["best"], True
            if chosen is None:
                finish(st)  # no draw ever yielded a sentence at all
                continue
            n_accept += 1
            n_fallback += fell_back

            _, sent, sig = chosen
            st["text"] = f"{st['text']} {sent}".strip()
            st["prev_sig"] = sig
            st["n_sent"] += 1
            st["trials"], st["best"] = 0, None
            if st["n_sent"] >= max_sentences or len(st["text"].split()) > max_words:
                finish(st)
            else:
                still.append(st)
        active = still

    pbar.close()
    if n_accept:
        # A high fallback rate means the valid-region + margin constraint is
        # rarely satisfiable, so most sentences carry no mark at all and the
        # AUC will be near chance for reasons that have nothing to do with the
        # attack. Expect a few percent; tens of percent means --margin is too
        # strict or the encoder is not the one the partition was tuned for.
        print(
            f"  accepted {n_accept} sentences, {n_fallback} "
            f"({100 * n_fallback / n_accept:.1f}%) by best-margin fallback "
            f"after {max_trials} failed trials"
        )
    return results


def cmd_generate(args):
    sweep_dir = Path(args.sweep)
    model, tok = load_llama(args.model or config.GEN_MODEL)
    embedder = Embedder(args.encoder)
    dim = embedder.m.get_sentence_embedding_dimension()
    lsh = LSHPartition(dim, args.bits, args.key, args.gamma)
    print(
        f"encoder={args.encoder} dim={dim} bits={args.bits} "
        f"regions={lsh.n_regions} gamma={args.gamma} margin={args.margin}"
    )
    print(
        f"batch: {args.batch_prompts} prompts x {args.candidates} candidates "
        f"= {args.batch_prompts * args.candidates} sequences per step\n"
    )

    # Refuse to mix encoders across seeds: if some gen_semstamp files were made
    # under different settings, the resulting set cannot be detected coherently.
    cfg_path, cfg = sweep_dir / CONFIG_NAME, settings(args)
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        stale = [s for s in SEEDS if (sweep_dir / f"gen_semstamp_s{s}.jsonl").exists()]
        if old != cfg and stale:
            raise SystemExit(
                f"{CONFIG_NAME} records {old}\nbut this run asks for {cfg}.\n"
                "Generations made under one encoder cannot be detected under "
                "another. Delete and regenerate all of:\n"
                + "".join(f"  {sweep_dir}/gen_semstamp_s{s}.jsonl\n" for s in stale)
                + f"  {sweep_dir}/para_*_semstamp_s*.jsonl\n"
                f"  {sweep_dir}/semstamp_scores_*.npz"
            )
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # the sweep's prompts, so SemStamp is scored on identical inputs
    with open(sweep_dir / "gen_unsteered_s0.jsonl") as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    for seed in SEEDS:
        out_path = sweep_dir / f"gen_semstamp_s{seed}.jsonl"
        if out_path.exists():
            print(f"[gen] skip seed={seed} (exists)")
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        texts = generate_prompts(
            model,
            tok,
            embedder,
            lsh,
            prompts,
            args.max_sentences,
            args.max_new_tokens,
            args.candidates,
            args.max_trials,
            args.margin,
            args.temperature,
            args.top_p,
            args.batch_prompts,
            args.max_words,
            desc=f"semstamp s{seed}",
        )
        tmp = out_path.with_suffix(".jsonl.partial")
        with open(tmp, "w") as f:
            for i, (p, txt) in enumerate(zip(prompts, texts)):
                f.write(json.dumps({"idx": i, "prompt": p, "completion": txt}) + "\n")
        tmp.rename(out_path)

    print(
        f"\nNow paraphrase these with the same attacker as the rest of the sweep."
        f"\n15_dipper_attack.py picks the semstamp tag up automatically:"
        f"\n  python 15_dipper_attack.py --sweep {sweep_dir}"
        f"\nthen score:"
        f"\n  python 16_semstamp.py detect --sweep {sweep_dir} "
        f"--field paraphrased --attack dipper"
    )


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------
def z_scores(texts, embedder, lsh):
    """One-proportion z for licensed-region hits, per text. NaN if < 2 sentences.

    Every sentence of every text is embedded in one batched call. Scoring text
    by text meant thousands of ~8-sentence forward passes, which on a sentence
    encoder is almost all launch overhead.
    """
    per_text = [[s for s in split_sentences(t) if s.strip()] for t in texts]
    flat = [s for sents in per_text for s in sents]
    sigs, _ = lsh.signatures_and_margins(embedder(flat))

    g, out, cur = lsh.gamma, [], 0
    for sents in per_text:
        n = len(sents)
        chunk, cur = sigs[cur : cur + n], cur + n
        if n < 2:
            out.append(float("nan"))
            continue
        # prev_sig starts at 0, matching the initial state in generate_prompts,
        # so the first sentence is tested against the region set it was sampled
        # under.
        prev_sig, hits = 0, 0
        for sig in chunk:
            if int(sig) in lsh.valid_regions(prev_sig):
                hits += 1
            prev_sig = int(sig)
        out.append((hits - g * n) / math.sqrt(n * g * (1 - g)))
    return out


def cmd_detect(args):
    from sklearn.metrics import roc_auc_score

    sweep_dir = Path(args.sweep)

    cfg_path, cfg = sweep_dir / CONFIG_NAME, settings(args)
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old != cfg:
            raise SystemExit(
                "detection settings disagree with the ones the text was "
                f"generated under.\n  generated: {old}\n  detecting: {cfg}\n"
                "This would score signatures the generator never targeted and "
                "silently return AUC ~ 0.5.\n"
                f"Re-run with --encoder {old['encoder']}"
            )
    else:
        print(
            f"WARNING: no {CONFIG_NAME} in {sweep_dir}; cannot verify that "
            "--encoder matches the one used at generation time."
        )

    embedder = Embedder(args.encoder)
    lsh = LSHPartition(
        embedder.m.get_sentence_embedding_dimension(), args.bits, args.key, args.gamma
    )

    def load(stem, field):
        out = []
        for seed in SEEDS:
            p = sweep_dir / f"{stem}_s{seed}.jsonl"
            if not p.exists():
                raise SystemExit(f"missing {p}")
            with open(p) as f:
                out += [json.loads(line)[field] for line in f]
        return out

    # Paraphrase files are named by whichever attacker produced them: Pegasus
    # (via 06) writes para_<tag>_s<seed>, DIPPER (via 15) writes
    # para_dipper_<tag>_s<seed>. --attack selects between them so SemStamp is
    # scored under exactly the attack the rest of the paper uses.
    pre = "para" if args.attack is None else f"para_{args.attack}"
    if args.field == "paraphrased":
        pos_stem, neg_stem, neg_field = f"{pre}_semstamp", f"{pre}_unsteered", "paraphrased"
    else:
        pos_stem, neg_stem, neg_field = "gen_semstamp", "gen_unsteered", "completion"

    pos = load(pos_stem, args.field)
    neg = load(neg_stem, neg_field)

    print(f"scoring {len(pos)} watermarked + {len(neg)} unwatermarked texts")
    zp = z_scores(pos, embedder, lsh)
    zn = z_scores(neg, embedder, lsh)

    scores = np.array(zp + zn)
    y = np.array([1] * len(zp) + [0] * len(zn))
    ok = np.isfinite(scores)
    dropped = int((~ok).sum())
    auc = float(roc_auc_score(y[ok], scores[ok]))

    print(f"\nSemStamp {args.field}: AUC = {auc:.3f} (symmetric {max(auc, 1 - auc):.3f})")
    print(f"  mean z: watermarked {np.nanmean(zp):+.2f}, unwatermarked {np.nanmean(zn):+.2f}")
    if dropped:
        print(f"  {dropped} texts had too few sentences to score and were dropped")
    print("\nCompare against the kgw_* rows of sweep_results.csv. State the encoder")
    print("used: a general-purpose one understates SemStamp (see module docstring).")

    # the attack belongs in the name: without it a Pegasus run and a DIPPER run
    # at the same --field overwrite each other
    suffix = "" if args.attack is None else f"_{args.attack}"
    np.savez(
        sweep_dir / f"semstamp_scores_{args.field}{suffix}.npz",
        score=scores,
        y=y,
        auc=auc,
        encoder=args.encoder,
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--sweep", required=True)
        p.add_argument(
            "--encoder",
            default="sentence-transformers/all-mpnet-base-v2",
            help="USE THE AUTHORS' FINE-TUNED ENCODER for paper numbers; "
            "the default understates SemStamp",
        )
        p.add_argument("--bits", type=int, default=8)
        p.add_argument("--gamma", type=float, default=0.25)
        p.add_argument("--key", type=int, default=42)

    g = sub.add_parser("generate")
    common(g)
    g.add_argument(
        "--model",
        default=None,
        help="HF id; defaults to config.GEN_MODEL for the active "
        "AOS_MODEL. Prefer the default: AOS_MODEL is also what "
        "decides whether Qwen thinking mode gets disabled, so "
        "naming a model here that disagrees with it is a trap.",
    )
    g.add_argument("--margin", type=float, default=0.02)
    g.add_argument("--candidates", type=int, default=8)
    g.add_argument("--max-trials", type=int, default=8)
    g.add_argument("--max-sentences", type=int, default=8)
    g.add_argument("--max-new-tokens", type=int, default=48)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument(
        "--batch-prompts",
        type=int,
        default=32,
        help="prompts held in flight; the GPU batch is this times "
        "--candidates, so 32 x 8 = 256 sequences. Sized to be "
        "safe for a 1-2B model on an 80GB card (~15GB for "
        "Qwen3-1.7B); raise to 64 if headroom allows.",
    )
    g.add_argument(
        "--max-words",
        type=int,
        default=200,
        help="stop a completion once it passes this many words",
    )

    d = sub.add_parser("detect")
    common(d)
    d.add_argument("--field", default="paraphrased", choices=["completion", "paraphrased"])
    d.add_argument(
        "--attack",
        default=None,
        help="which attacker's paraphrases to score, e.g. 'dipper'. "
        "Omit for the default Pegasus files.",
    )

    args = ap.parse_args()
    (cmd_generate if args.cmd == "generate" else cmd_detect)(args)


if __name__ == "__main__":
    main()
