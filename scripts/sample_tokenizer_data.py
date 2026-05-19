"""Stream HF datasets, NFC-normalize, write ~100M chars to a single text file
for BPE tokenizer training.

Budgets per tokenizer.md:
  50M chars — bigcode/the-stack-dedup (lang=SQL, gated dataset, requires HF auth)
  30M chars — koutch/stackoverflow_sql
  20M chars — HuggingFaceFW/fineweb-edu (sample-10BT config)

Run from the sql-lm/ repo root:
    python -m scripts.sample_tokenizer_data
"""
import os
import sys
from pathlib import Path

from datasets import load_dataset

from scripts.preprocess import preprocess

OUTPUT = Path("data/tokenizer_training_text.txt")
PROGRESS_EVERY = 5_000_000  # chars

SOURCES = [
    {
        "name": "the-stack-dedup (SQL)",
        "loader": lambda: load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/sql",
            split="train",
            streaming=True,
        ),
        "text_field": "content",
        "budget": 50_000_000,
    },
    {
        # Substituted for koutch/stackoverflow_sql which was removed from the Hub.
        # b-mc2/sql-create-context: 78k curated (question, schema, SQL) triples.
        # Better text-to-SQL signal than raw SO posts; same role in the mix.
        "name": "sql-create-context",
        "loader": lambda: load_dataset(
            "b-mc2/sql-create-context",
            split="train",
            streaming=True,
        ),
        "text_field": None,  # concat question + context + answer via fallback
        "budget": 30_000_000,
    },
    {
        "name": "fineweb-edu (sample-10BT)",
        "loader": lambda: load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        ),
        "text_field": "text",
        "budget": 20_000_000,
    },
]


def extract_text(row, field):
    if field and field in row and row[field]:
        return row[field]
    return " ".join(str(v) for v in row.values() if isinstance(v, str) and v)


def sample_source(source, fh):
    name = source["name"]
    budget = source["budget"]
    print(f"\n=== {name} (budget {budget:,} chars) ===")

    chars_written = 0
    docs_written = 0
    next_milestone = PROGRESS_EVERY

    ds = source["loader"]()
    for row in ds:
        raw = extract_text(row, source["text_field"])
        if not raw:
            continue
        text = preprocess(raw).strip()
        if not text:
            continue
        fh.write(text + "\n")
        chars_written += len(text) + 1
        docs_written += 1
        if chars_written >= next_milestone:
            print(f"  {chars_written:,} chars / {docs_written:,} docs")
            next_milestone += PROGRESS_EVERY
        if chars_written >= budget:
            break

    print(f"  DONE: {chars_written:,} chars, {docs_written:,} docs")
    return chars_written


def check_hf_auth():
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists():
        return
    print(
        "WARNING: no HF token detected (env HF_TOKEN or ~/.cache/huggingface/token).\n"
        "The Stack is gated and will fail without auth. Run:\n"
        "    huggingface-cli login",
        file=sys.stderr,
    )


def main():
    if OUTPUT.exists():
        resp = input(f"{OUTPUT} already exists. Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    check_hf_auth()

    total = 0
    with open(OUTPUT, "w", encoding="utf-8") as fh:
        for source in SOURCES:
            total += sample_source(source, fh)

    print(f"\n=== TOTAL: {total:,} chars written to {OUTPUT} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
