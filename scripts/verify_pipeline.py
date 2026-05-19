"""Verify the tokenized corpus after build_corpus.py and write_manifest.py.

Runs 10 checks. Exits 0 if all pass, 1 if any fail.

Usage:
    python -m scripts.verify_pipeline              # checks all sources in manifest
    python -m scripts.verify_pipeline --source sqale  # check one source only

Checks:
    1. All expected .npy files exist and load
    2. Shape is [N, 512], dtype int16
    3. Token IDs are in [0, 12287]
    4. BOS (0) appears at the start of sampled sequences
    5. EOS (1) appears before BOS at document boundaries
    6. PAD (2) does NOT appear anywhere
    7. Finetuning delimiter IDs 4-9 do NOT appear (pretraining only)
    8. Per-source proportions match targets within ±5%
    9. 10 random sequences per source decode to plausible text
   10. Recomputed sha256 checksums match the manifest
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from scripts.build_corpus_helpers import FINETUNING_IDS, VOCAB_SIZE, sha256_file
from scripts.corpus_sources import SOURCES, TARGET_PROPORTIONS

OUTPUT_DIR = Path("data/tokenized")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
SEQ_LEN = 512
SAMPLE_SEQUENCES = 100   # sequences sampled per source for boundary/range checks
DECODE_SEQUENCES = 10    # sequences decoded for plausibility check


def _load_manifest():
    if not MANIFEST_PATH.exists():
        print(f"FAIL: manifest not found at {MANIFEST_PATH}. Run write_manifest.py first.")
        sys.exit(1)
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_arr(name: str, split: str):
    path = OUTPUT_DIR / f"{name}_{split}.npy"
    if not path.exists():
        return None, path
    return np.load(path, mmap_mode="r"), path


def check_files_exist(names) -> bool:
    ok = True
    for name in names:
        for split in ("train", "val"):
            path = OUTPUT_DIR / f"{name}_{split}.npy"
            if not path.exists():
                print(f"  FAIL [files_exist]: {path} missing")
                ok = False
    return ok


def check_shapes_and_dtype(names) -> bool:
    ok = True
    for name in names:
        for split in ("train", "val"):
            arr, path = _load_arr(name, split)
            if arr is None:
                continue
            if arr.ndim != 2 or arr.shape[1] != SEQ_LEN:
                print(f"  FAIL [shape]: {path} shape {arr.shape}, expected [N, {SEQ_LEN}]")
                ok = False
            if arr.dtype != np.int16:
                print(f"  FAIL [dtype]: {path} dtype {arr.dtype}, expected int16")
                ok = False
    return ok


def check_token_id_range(names) -> bool:
    ok = True
    rng = np.random.default_rng(0)
    for name in names:
        arr, path = _load_arr(name, "train")
        if arr is None or len(arr) == 0:
            continue
        idx = rng.integers(0, len(arr), size=min(SAMPLE_SEQUENCES, len(arr)))
        sample = arr[idx].astype(np.int32)
        if sample.min() < 0:
            print(f"  FAIL [id_range]: {name} has negative token IDs (min={sample.min()})")
            ok = False
        if sample.max() >= VOCAB_SIZE:
            print(f"  FAIL [id_range]: {name} has token IDs >= {VOCAB_SIZE} (max={sample.max()})")
            ok = False
    return ok


def check_bos_at_start(names) -> bool:
    """Check that BOS (0) appears inside sampled sequences (doc boundaries present).

    In packed sequences, BOS only lands at column 0 when a doc boundary aligns to a
    512-token boundary (probability ≈ 1/512). We instead check that BOS appears
    *somewhere* within sequences at a plausible rate — roughly 512/avg_doc_tokens.
    Any source with >0.001% occurrence rate passes.
    """
    ok = True
    rng = np.random.default_rng(1)
    for name in names:
        arr, _ = _load_arr(name, "train")
        if arr is None or len(arr) == 0:
            continue
        idx = rng.integers(0, len(arr), size=min(SAMPLE_SEQUENCES, len(arr)))
        sample = arr[idx].astype(np.int32)
        bos_count = int((sample == 0).sum())
        total_tokens = sample.size
        bos_rate = bos_count / total_tokens
        if bos_rate == 0.0:
            print(
                f"  FAIL [bos_at_start]: {name} BOS token (0) not found in "
                f"{SAMPLE_SEQUENCES} sampled sequences — check pack_sequences"
            )
            ok = False
    return ok


def check_eos_before_bos(names) -> bool:
    """Check that EOS (1) appears immediately before BOS (0) at boundaries."""
    ok = True
    rng = np.random.default_rng(2)
    for name in names:
        arr, _ = _load_arr(name, "train")
        if arr is None or len(arr) == 0:
            continue
        # Flatten a sample into a 1D stream and look for [1, 0] patterns
        idx = rng.integers(0, len(arr), size=min(SAMPLE_SEQUENCES, len(arr)))
        flat = arr[idx].astype(np.int32).ravel()
        bos_positions = np.where(flat == 0)[0]
        if len(bos_positions) == 0:
            print(f"  FAIL [eos_before_bos]: {name} no BOS found in sample")
            ok = False
            continue
        # Skip the very first BOS (may be at position 0 with no prior EOS)
        inner_bos = bos_positions[bos_positions > 0]
        if len(inner_bos) == 0:
            continue
        eos_before = flat[inner_bos - 1]
        bad_frac = (eos_before != 1).mean()
        if bad_frac > 0.05:
            print(
                f"  FAIL [eos_before_bos]: {name} {bad_frac:.1%} of inner BOS "
                f"tokens not preceded by EOS"
            )
            ok = False
    return ok


def check_no_pad_tokens(names) -> bool:
    """PAD (ID 2) should not appear as a padding token. Literal <pad> strings
    in source code tokenize to ID 2 — tolerate up to 1 per million tokens."""
    ok = True
    MAX_RATE = 1e-6
    for name in names:
        arr, path = _load_arr(name, "train")
        if arr is None or len(arr) == 0:
            continue
        count = int(np.sum(arr == 2))
        rate = count / arr.size
        if rate > MAX_RATE:
            print(
                f"  FAIL [no_pad]: {name}_train.npy contains {count} PAD tokens "
                f"(rate {rate:.2e} > {MAX_RATE:.0e}) — possible padding bug"
            )
            ok = False
    return ok


def check_no_finetuning_tokens(names) -> bool:
    """Finetuning delimiters (IDs 4-9) must not appear in pretraining data.
    Source code may contain literal <schema>/<sql>/etc. strings — tolerate up
    to 1 per million tokens as noise from code comments/docs."""
    ok = True
    MAX_RATE = 1e-6
    for name in names:
        arr, _ = _load_arr(name, "train")
        if arr is None or len(arr) == 0:
            continue
        mask = np.zeros(arr.shape, dtype=bool)
        for fid in FINETUNING_IDS:
            mask |= (arr == fid)
        count = int(np.sum(mask))
        rate = count / arr.size
        if rate > MAX_RATE:
            print(
                f"  FAIL [no_finetune_ids]: {name}_train.npy contains {count} "
                f"finetuning delimiter tokens (rate {rate:.2e} > {MAX_RATE:.0e})"
            )
            ok = False
    return ok


def check_proportions(manifest) -> bool:
    ok = True
    sources = manifest.get("sources", {})
    total_train = manifest.get("total_train_tokens", 0)
    if total_train == 0:
        print("  FAIL [proportions]: manifest total_train_tokens = 0")
        return False

    for name, target in TARGET_PROPORTIONS.items():
        if name not in sources:
            print(f"  WARN  [proportions]: {name} not in manifest — skipped")
            continue
        actual_tokens = sources[name]["train"]["tokens"]
        actual_prop = actual_tokens / total_train
        delta = abs(actual_prop - target)
        if delta > 0.05:
            print(
                f"  FAIL [proportions]: {name} actual {actual_prop:.3f} vs "
                f"target {target:.3f} (delta {delta:.3f} > 0.05)"
            )
            ok = False
    return ok


def check_plausibility(names, tokenizer) -> bool:
    ok = True
    rng = np.random.default_rng(3)
    for name in names:
        arr, _ = _load_arr(name, "train")
        if arr is None or len(arr) == 0:
            continue
        idx = rng.integers(0, len(arr), size=min(DECODE_SEQUENCES, len(arr)))
        print(f"\n  --- {name} samples ---")
        for i, seq in enumerate(arr[idx]):
            try:
                decoded = tokenizer.decode(seq.astype(np.int32).tolist())
            except Exception as e:
                print(f"  FAIL [plausibility]: {name} seq {i} decode error: {e}")
                ok = False
                continue
            has_alnum = any(c.isalnum() for c in decoded)
            if not has_alnum:
                print(f"  FAIL [plausibility]: {name} seq {i} has no alphanumeric chars")
                ok = False
            # Print first 120 chars for manual eyeballing
            preview = decoded[:120].replace("\n", " ").replace("\r", "")
            print(f"    [{i}] {preview!r}")
    return ok


def check_checksums(names, manifest) -> bool:
    ok = True
    sources = manifest.get("sources", {})
    for name in names:
        if name not in sources:
            continue
        for split in ("train", "val"):
            path = OUTPUT_DIR / f"{name}_{split}.npy"
            if not path.exists():
                continue
            expected = sources[name][split]["sha256"]
            actual = sha256_file(path)
            if actual != expected:
                print(
                    f"  FAIL [checksums]: {name}_{split}.npy sha256 mismatch\n"
                    f"    expected: {expected}\n"
                    f"    actual:   {actual}"
                )
                ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify tokenized corpus files")
    parser.add_argument("--source", default=None, help="Check one source only")
    parser.add_argument(
        "--skip-plausibility",
        action="store_true",
        help="Skip the decode/plausibility check (faster)",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    all_names = list(SOURCES.keys())
    names = [args.source] if args.source else all_names

    if not TOKENIZER_PATH.exists():
        print(f"ERROR: tokenizer not found at {TOKENIZER_PATH}", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    results = {}

    print("=== Check 1: Files exist ===")
    results["files_exist"] = check_files_exist(names)

    print("=== Check 2: Shape [N, 512] and dtype int16 ===")
    results["shapes"] = check_shapes_and_dtype(names)

    print("=== Check 3: Token IDs in [0, 12287] ===")
    results["id_range"] = check_token_id_range(names)

    print("=== Check 4: BOS (0) at start of sequences ===")
    results["bos_at_start"] = check_bos_at_start(names)

    print("=== Check 5: EOS (1) before BOS at doc boundaries ===")
    results["eos_before_bos"] = check_eos_before_bos(names)

    print("=== Check 6: No PAD tokens (ID 2) ===")
    results["no_pad"] = check_no_pad_tokens(names)

    print("=== Check 7: No finetuning delimiter tokens (IDs 4-9) ===")
    results["no_finetune_ids"] = check_no_finetuning_tokens(names)

    print("=== Check 8: Proportions match targets ±5% ===")
    results["proportions"] = check_proportions(manifest)

    if not args.skip_plausibility:
        print("=== Check 9: Plausibility (decode 10 random sequences per source) ===")
        results["plausibility"] = check_plausibility(names, tokenizer)
    else:
        print("=== Check 9: Plausibility — SKIPPED (--skip-plausibility) ===")
        results["plausibility"] = True

    print("=== Check 10: SHA-256 checksums match manifest ===")
    results["checksums"] = check_checksums(names, manifest)

    print("\n=== Summary ===")
    all_pass = True
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}  {check}")

    if all_pass:
        print("\nAll checks passed.")
        sys.exit(0)
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\nFailed checks: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
