"""Train BPE tokenizer per tokenizer.md.

8192 vocab. 23 special tokens with pinned IDs (0-22). Pre-tokenizer chain:
Digits(individual_digits=True) -> ByteLevel(add_prefix_space=False).
Decoder: ByteLevel. min_frequency=5.

Run from the sql-lm/ repo root:
    python -m scripts.train_tokenizer
"""
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, trainers
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence

INPUT = Path("data/tokenizer_training_text.txt")
OUTPUT = Path("tokenizer/tokenizer.json")

SPECIAL_TOKENS = [
    "<bos>", "<eos>", "<pad>", "<unk>",
    "<schema>", "<question>", "<sql>",
    "</schema>", "</question>", "</sql>",
    "<reserved_0>", "<reserved_1>", "<reserved_2>", "<reserved_3>",
    "<reserved_4>", "<reserved_5>", "<reserved_6>", "<reserved_7>",
    "<reserved_8>", "<reserved_9>", "<reserved_10>", "<reserved_11>",
    "<reserved_12>",
]


def main():
    if not INPUT.exists():
        raise SystemExit(
            f"Missing {INPUT}. Run scripts/sample_tokenizer_data.py first."
        )

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Sequence([
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=12288,  # bumped from 8192 — extended validation showed long-tail keyword splits
        special_tokens=SPECIAL_TOKENS,
        min_frequency=3,  # lowered from 5 — first run split common keywords
        show_progress=True,
    )

    print(f"Training BPE on {INPUT}...")
    tokenizer.train([str(INPUT)], trainer)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT))
    print(f"Saved -> {OUTPUT}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
