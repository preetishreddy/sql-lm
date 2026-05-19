"""Write data/tokenized/manifest.json after all source .npy files are built.

Records shapes, token counts, actual proportions, sha256 checksums per file,
and the tokenizer's git object hash. Separated from build_corpus.py because
the manifest must aggregate across all sources and is easy to re-run if one
source is rebuilt.

Usage:
    python -m scripts.write_manifest

The tokenizer git hash catches the case where you retrain the tokenizer and
forget to regenerate the corpus — the mismatch will be visible in the manifest.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.build_corpus_helpers import sha256_file
from scripts.corpus_sources import SOURCES, TARGET_PROPORTIONS

OUTPUT_DIR = Path("data/tokenized")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
SEQ_LEN = 512


def _tokenizer_git_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"HEAD:{TOKENIZER_PATH}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "unknown (not in git or not at HEAD)"


def main():
    if not OUTPUT_DIR.exists():
        print(f"ERROR: {OUTPUT_DIR} does not exist. Run build_corpus.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {OUTPUT_DIR} ...")

    sources_data = {}
    total_train_tokens = 0

    for name in SOURCES:
        train_path = OUTPUT_DIR / f"{name}_train.npy"
        val_path = OUTPUT_DIR / f"{name}_val.npy"

        if not train_path.exists() or not val_path.exists():
            print(f"  WARNING: {name} missing — skipping (run build_corpus --source {name})")
            continue

        print(f"  {name} — hashing ...", end="", flush=True)
        train_arr = np.load(train_path, mmap_mode="r")
        val_arr = np.load(val_path, mmap_mode="r")

        train_seqs, train_tokens = train_arr.shape[0], train_arr.shape[0] * SEQ_LEN
        val_seqs, val_tokens = val_arr.shape[0], val_arr.shape[0] * SEQ_LEN
        total_train_tokens += train_tokens

        train_sha = sha256_file(train_path)
        val_sha = sha256_file(val_path)
        print(f" {train_tokens + val_tokens:,} tokens")

        sources_data[name] = {
            "train": {
                "path": str(train_path),
                "shape": list(train_arr.shape),
                "tokens": train_tokens,
                "sequences": train_seqs,
                "sha256": train_sha,
            },
            "val": {
                "path": str(val_path),
                "shape": list(val_arr.shape),
                "tokens": val_tokens,
                "sequences": val_seqs,
                "sha256": val_sha,
            },
            "token_budget": SOURCES[name]["token_budget"],
            "target_proportion": round(TARGET_PROPORTIONS[name], 6),
        }

    # Compute actual proportions from train split (val is held out)
    for name, data in sources_data.items():
        actual = data["train"]["tokens"] / total_train_tokens if total_train_tokens else 0
        data["actual_proportion"] = round(actual, 6)

    manifest = {
        "vocab_size": 12288,
        "sequence_length": SEQ_LEN,
        "total_train_tokens": total_train_tokens,
        "total_sources": len(sources_data),
        "tokenizer_git_hash": _tokenizer_git_hash(),
        "created": datetime.now(timezone.utc).isoformat(),
        "sources": sources_data,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {MANIFEST_PATH}")
    print(f"Total train tokens: {total_train_tokens:,}")
    print(f"Sources included: {list(sources_data.keys())}")

    missing = [n for n in SOURCES if n not in sources_data]
    if missing:
        print(f"\nWARNING: missing sources not in manifest: {missing}")
        print("Run build_corpus.py for each, then re-run write_manifest.py")


if __name__ == "__main__":
    main()
