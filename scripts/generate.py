import functools
import numpy as np
import jax
import jax.numpy as jnp
from tokenizers import Tokenizer
from scripts.config import BOS_ID, EOS_ID, PAD_ID, CONTEXT_LENGTH


@functools.partial(jax.jit, static_argnames=['model'])
def _forward(params, model, token_ids):
    return model.apply({'params': params}, token_ids)  # [1, T, vocab]


def generate(params, model, tokenizer, prompt: str,
             max_new_tokens: int = 200,
             temperature: float = 1.0,
             top_p: float = 0.95,
             seed: int = 0) -> str:
    """
    Autoregressively generate tokens from a prompt.

    Args:
        params:         model parameters from a loaded checkpoint
        model:          SQLTransformer instance
        tokenizer:      tokenizers.Tokenizer loaded from tokenizer.json
        prompt:         text to complete
        max_new_tokens: maximum tokens to generate
        temperature:    >1 = more random, <1 = sharper. 1.0 = unchanged
        top_p:          nucleus sampling — only sample from top-p probability mass
        seed:           random seed for reproducibility

    Returns:
        generated text (prompt not included)
    """
    rng = np.random.default_rng(seed)

    prompt_ids = [BOS_ID] + tokenizer.encode(prompt).ids
    ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        window = ids[-CONTEXT_LENGTH:]
        seq_len = len(window)

        # Right-pad to CONTEXT_LENGTH so _forward always gets a fixed shape [1, 512]
        padded = window + [PAD_ID] * (CONTEXT_LENGTH - seq_len)
        x = jnp.array([padded], dtype=jnp.int32)

        logits = _forward(params, model, x)                   # [1, 512, vocab]
        next_logits = np.array(logits[0, seq_len - 1, :],
                               dtype=np.float32)              # last real position

        # Greedy decoding
        if temperature == 0.0:
            next_token = int(np.argmax(next_logits))
        else:
            if temperature != 1.0:
                next_logits /= temperature

            # Nucleus (top-p) sampling
            probs = np.exp(next_logits - next_logits.max())
            probs /= probs.sum()

            sorted_idx   = np.argsort(-probs)
            sorted_probs = probs[sorted_idx]
            cum_probs    = np.cumsum(sorted_probs)
            cutoff       = int(np.searchsorted(cum_probs, top_p)) + 1
            keep_idx     = sorted_idx[:cutoff]

            filtered = np.zeros_like(probs)
            filtered[keep_idx] = probs[keep_idx]
            filtered /= filtered.sum()

            next_token = int(rng.choice(len(filtered), p=filtered))

        if next_token == EOS_ID:
            break

        ids.append(next_token)

    generated_ids = ids[len(prompt_ids):]
    return tokenizer.decode(generated_ids)
