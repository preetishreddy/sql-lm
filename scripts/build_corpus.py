"""Build the pretraining corpus — tokenize, pack, and save per-source .npy files.

Run one source at a time:
    python -m scripts.build_corpus --source sqale
    python -m scripts.build_corpus --source all   # runs all in recommended order

Each source is fully independent. If a run crashes, just resume from that
source — earlier sources are already saved.

Run order (non-gated first, then gated):
    sqale, gretelai, nstext2sql, sql_create_context, fineweb, oasst, bird,
    stack_sql, stack_markdown, stack_python, stack_ruby

Output:
    data/tokenized/{name}_train.npy  — shape [N, 512], dtype int16
    data/tokenized/{name}_val.npy    — shape [V, 512], dtype int16

After all sources are built, run:
    python -m scripts.write_manifest
    python -m scripts.verify_pipeline
"""
import argparse
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from tokenizers import Tokenizer

from scripts.build_corpus_helpers import pack_sequences, sha256_file, train_val_split
from scripts.corpus_sources import RUN_ORDER, SOURCES
from scripts.preprocess import preprocess

TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
OUTPUT_DIR = Path("data/tokenized")
ENCODE_BATCH_SIZE = 1000   # docs per encode_batch call — engages Rust threadpool
PROGRESS_EVERY = 10_000_000  # print progress every 10M tokens


def check_hf_auth():
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists():
        return
    print(
        "WARNING: no HF token detected. Gated sources (The Stack) will fail.\n"
        "Run:  huggingface-cli login",
        file=sys.stderr,
    )


def _generate_encoded_docs(source, tokenizer):
    """Generator: yields list[int] of raw token IDs for each accepted document.

    Cycles through the source dataset until the token budget is reached.
    Cycling is correct for small datasets (sql_create_context, bird) where the
    spec budget intentionally exceeds the unique data available.
    Uses encode_batch() — 10-15x faster than a Python encode() loop.
    """
    name = source["name"]
    budget = source["token_budget"]
    batch_texts = []
    tokens_so_far = 0
    docs_accepted = 0
    epoch = 0
    next_milestone = PROGRESS_EVERY

    while tokens_so_far < budget:
        epoch += 1
        stream = source["loader"]()
        epoch_had_docs = False

        for row in stream:
            raw = source["text_extractor"](row)
            if not raw:
                continue
            text = preprocess(raw)
            if not (100 <= len(text) <= 1_000_000):
                continue
            if not source["filter"](row, text):
                continue

            batch_texts.append(text)
            docs_accepted += 1
            epoch_had_docs = True

            if len(batch_texts) >= ENCODE_BATCH_SIZE:
                for enc in tokenizer.encode_batch(batch_texts):
                    yield enc.ids
                    tokens_so_far += len(enc.ids)
                    if tokens_so_far >= next_milestone:
                        print(
                            f"  [{name}] {tokens_so_far:,} / {budget:,} tokens "
                            f"({tokens_so_far / budget:.1%}), {docs_accepted:,} docs"
                            + (f", epoch {epoch}" if epoch > 1 else "")
                        )
                        next_milestone += PROGRESS_EVERY
                    if tokens_so_far >= budget:
                        return
                batch_texts.clear()

        # Flush the last partial batch from this epoch
        if batch_texts:
            for enc in tokenizer.encode_batch(batch_texts):
                yield enc.ids
                tokens_so_far += len(enc.ids)
                if tokens_so_far >= budget:
                    return
            batch_texts.clear()

        # If the source yielded nothing at all, it is empty — bail out
        if not epoch_had_docs:
            print(f"  [{name}] source exhausted with no docs — stopping at {tokens_so_far:,} tokens")
            return


def build_source(name: str, tokenizer: Tokenizer, force: bool = False) -> dict:
    """Tokenize, pack, split, and save one source. Returns token count info."""
    source = SOURCES[name]
    out_train = OUTPUT_DIR / f"{name}_train.npy"
    out_val = OUTPUT_DIR / f"{name}_val.npy"

    if out_train.exists() and out_val.exists() and not force:
        train = np.load(out_train, mmap_mode="r")
        val = np.load(out_val, mmap_mode="r")
        train_tokens = train.shape[0] * 512
        val_tokens = val.shape[0] * 512
        print(
            f"[{name}] already exists — {train_tokens + val_tokens:,} tokens "
            f"({train.shape[0]:,} train + {val.shape[0]:,} val sequences). "
            f"Use --force to rebuild."
        )
        return {
            "train_sequences": train.shape[0],
            "val_sequences": val.shape[0],
            "train_sha256": sha256_file(out_train),
            "val_sha256": sha256_file(out_val),
        }

    print(f"\n=== Building {name} (budget {source['token_budget']:,} tokens) ===")

    doc_gen = _generate_encoded_docs(source, tokenizer)

    # pack_sequences handles BOS/EOS boundaries and returns [N, 512] int16
    packed = pack_sequences(doc_gen)

    if len(packed) == 0:
        print(f"  WARNING: [{name}] produced 0 sequences — check filter/loader.")
        return {"train_sequences": 0, "val_sequences": 0,
                "train_sha256": "", "val_sha256": ""}

    # Deterministic 99/1 train/val split (seed=42 per spec)
    train, val = train_val_split(packed, val_frac=0.01, seed=42)
    del packed  # free before saving

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out_train, train)
    np.save(out_val, val)

    train_sha = sha256_file(out_train)
    val_sha = sha256_file(out_val)

    print(
        f"  [{name}] DONE — {train.shape[0]:,} train + {val.shape[0]:,} val sequences "
        f"= {(train.shape[0] + val.shape[0]) * 512:,} tokens\n"
        f"  train sha256: {train_sha}\n"
        f"  val   sha256: {val_sha}"
    )

    return {
        "train_sequences": train.shape[0],
        "val_sequences": val.shape[0],
        "train_sha256": train_sha,
        "val_sha256": val_sha,
    }


def main():
    parser = argparse.ArgumentParser(description="Build pretraining corpus .npy files")
    parser.add_argument(
        "--source",
        required=True,
        choices=list(SOURCES.keys()) + ["all"],
        help="Source name to build, or 'all' to build all in recommended order",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild source even if output files already exist",
    )
    args = parser.parse_args()

    if not TOKENIZER_PATH.exists():
        print(f"ERROR: tokenizer not found at {TOKENIZER_PATH}", file=sys.stderr)
        sys.exit(1)

    check_hf_auth()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    sources_to_run = RUN_ORDER if args.source == "all" else [args.source]

    for name in sources_to_run:
        if SOURCES[name]["gated"]:
            check_hf_auth()
        build_source(name, tokenizer, force=args.force)

    print("\nDone. Run `python -m scripts.write_manifest` next.")


if __name__ == "__main__":
    main()
