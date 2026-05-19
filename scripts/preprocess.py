import unicodedata


def preprocess(text: str) -> str:
    """Canonical text normalization. MUST be used by every script that feeds
    text into the tokenizer — sampling, training-corpus tokenization, and
    inference. Changing this function invalidates the tokenizer."""
    return unicodedata.normalize("NFC", text)
