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


def beam_search(params, model, tokenizer, prompt: str,
                num_beams: int = 4,
                max_new_tokens: int = 150,
                length_penalty: float = 0.6) -> str:
    """
    Beam search decoding. Batches all beams into a single forward pass per step.

    length_penalty: score /= (gen_len ** length_penalty). 0 = no normalisation,
                    0.6 = mild (good default for SQL), 1.0 = full length norm.
    """
    prompt_ids = [BOS_ID] + tokenizer.encode(prompt).ids
    n_prompt   = len(prompt_ids)

    # (token_ids, cumulative_log_prob, is_finished)
    beam_ids    = [list(prompt_ids) for _ in range(num_beams)]
    beam_scores = [0.0] * num_beams
    beam_done   = [False] * num_beams

    for _ in range(max_new_tokens):
        if all(beam_done):
            break

        # Batched forward pass — all k beams in one call
        seq_len = len(beam_ids[0])          # all beams are same length
        pos     = min(seq_len, CONTEXT_LENGTH) - 1
        windows = [ids[-CONTEXT_LENGTH:] for ids in beam_ids]
        padded  = np.array(
            [w + [PAD_ID] * (CONTEXT_LENGTH - len(w)) for w in windows],
            dtype=np.int32)                 # [k, CONTEXT_LENGTH]

        logits     = _forward(params, model, jnp.array(padded))   # [k, T, vocab]
        nxt_logits = np.array(logits[:, pos, :], dtype=np.float32) # [k, vocab]

        # Numerically stable log-softmax per beam
        shifted   = nxt_logits - nxt_logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

        # Expand candidates
        candidates = []
        for i in range(num_beams):
            if beam_done[i]:
                candidates.append((beam_ids[i], beam_scores[i], True))
                continue
            for tok_id in np.argsort(-log_probs[i])[:num_beams]:
                new_ids   = beam_ids[i] + [int(tok_id)]
                new_score = beam_scores[i] + float(log_probs[i, tok_id])
                is_done   = (int(tok_id) == EOS_ID)
                candidates.append((new_ids, new_score, is_done))

        # Length-normalised ranking
        def _rank(c):
            gen_len = max(1, len(c[0]) - n_prompt)
            return c[1] / (gen_len ** length_penalty)

        candidates.sort(key=_rank, reverse=True)
        top = candidates[:num_beams]

        beam_ids    = [c[0] for c in top]
        beam_scores = [c[1] for c in top]
        beam_done   = [c[2] for c in top]

    best = max(range(num_beams),
               key=lambda i: beam_scores[i] / (max(1, len(beam_ids[i]) - n_prompt) ** length_penalty))
    generated_ids = beam_ids[best][n_prompt:]
    if generated_ids and generated_ids[-1] == EOS_ID:
        generated_ids = generated_ids[:-1]
    return tokenizer.decode(generated_ids)
