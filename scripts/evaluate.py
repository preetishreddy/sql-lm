import jax
import jax.numpy as jnp
import optax
from scripts.config import BATCH_SIZE, VOCAB_SIZE

def loss_fn(params, batch, model):
    """
    Standard causal language modeling loss: predict next token.
    """
    # batch: [B, 512] int32
    inputs  = batch[:, :-1]  # [B, 511]
    targets = batch[:, 1:]   # [B, 511]

    logits = model.apply({'params': params}, inputs)
    # logits: [B, 511, vocab_size]

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, VOCAB_SIZE),
        targets.reshape(-1)
    ).mean()

    return loss

import functools

@functools.partial(jax.jit, static_argnames=['model'])
def eval_step(state, batch, model):
    """
    Compute loss for a single batch without updating gradients.
    """
    return loss_fn(state.params, batch, model)

def evaluate(state, val_loader, model, n_batches: int = 50) -> float:
    """
    Compute representative validation loss across the corpus.
    
    Args:
        state: Current TrainState.
        val_loader: CorpusLoader instance for the validation split.
        model: SQLTransformer instance.
        n_batches: Number of batches to average over.
        
    Returns:
        Average validation loss.
    """
    losses = []
    
    # Use val_batch_iter to get a balanced sample across all sources
    for batch in val_loader.val_batch_iter(BATCH_SIZE, n_batches):
        batch_jax = jnp.array(batch)
        loss = eval_step(state, batch_jax, model)
        losses.append(float(loss))
        
    return sum(losses) / len(losses)

if __name__ == "__main__":
    print("SQL-LM Evaluation Module")
