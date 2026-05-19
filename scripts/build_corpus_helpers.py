"""Utility functions for the pretraining corpus pipeline.

All functions here are pure (no I/O, no global state) so they are easy to
unit-test independently of the HuggingFace dataset infrastructure.
"""
import array
import hashlib

import numpy as np

BOS = 0
EOS = 1
PAD = 2
SEQ_LEN = 512
VOCAB_SIZE = 12288
# IDs 4-9 are finetuning delimiters (<schema>,<question>,<sql> + closing tags).
# They must never appear in pretraining data.
FINETUNING_IDS = frozenset(range(4, 10))


def count_alphanum_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalnum()) / len(text)


def pack_sequences(
    doc_ids_iter,
    seq_len: int = SEQ_LEN,
    bos: int = BOS,
    eos: int = EOS,
) -> np.ndarray:
    """Consume an iterable of raw token-ID lists and pack into fixed-length sequences.

    BOS and EOS are prepended/appended around each document. Tokens that do
    not fill a complete sequence at the end are discarded — no padding added.

    Args:
        doc_ids_iter: iterable of list[int], token IDs without BOS/EOS.
        seq_len: sequence length (default 512).
        bos: BOS token ID (default 0).
        eos: EOS token ID (default 1).

    Returns:
        np.ndarray of shape [N, seq_len], dtype int16.
    """
    buf = array.array("h")  # signed int16; token IDs 0-12287 all fit
    for ids in doc_ids_iter:
        buf.append(bos)
        buf.extend(ids)
        buf.append(eos)
    if len(buf) < seq_len:
        return np.empty((0, seq_len), dtype=np.int16)
    flat = np.frombuffer(buf, dtype=np.int16)
    n = len(flat) // seq_len
    return flat[: n * seq_len].reshape(n, seq_len).copy()


def train_val_split(
    arr: np.ndarray,
    val_frac: float = 0.01,
    seed: int = 42,
):
    """Deterministic shuffle then train/val split.

    Args:
        arr: [N, seq_len] packed sequence array.
        val_frac: fraction of rows to hold out as validation.
        seed: RNG seed for reproducibility.

    Returns:
        (train, val) tuple of np.ndarray.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(arr))
    shuffled = arr[idx]
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def sha256_file(path) -> str:
    """Streaming SHA-256 hash of a file. Works for multi-GB .npy files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
