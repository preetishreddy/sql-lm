"""Fine-tune the pretrained SQL transformer on instruction-tuning data.

Loads pretrained params from step_76500 (best full-eval val loss = 1.5618),
creates a fresh AdamW optimizer at 1e-5 LR, and trains with masked
cross-entropy — only SQL response tokens (mask=1) contribute to the loss.

Usage (Colab cell):
    from scripts.finetune import finetune
    finetune()

Data:    /content/data/finetune/{train,val}_{tokens,mask}.npy
Start:   /content/drive/MyDrive/sql-lm-data/checkpoints/step_76500
Outputs: /content/checkpoints/ft_step_XXXXX  (+ Drive copy)
         /content/ft_metrics.jsonl            (+ Drive copy)
"""

import os
import json
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

from scripts.config import (
    VOCAB_SIZE, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS,
    HEAD_DIM, INTERMEDIATE_DIM, CONTEXT_LENGTH, ROPE_BASE,
)
from scripts.model import SQLTransformer

# -----------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------
FT_BATCH_SIZE       = 32
FT_TOTAL_STEPS      = 5_000
FT_WARMUP_STEPS     = 100
FT_PEAK_LR          = 1e-5
FT_MIN_LR           = 1e-6
FT_WEIGHT_DECAY     = 0.01   # lighter than pretraining's 0.1
FT_GRAD_CLIP        = 1.0
FT_DROPOUT          = 0.1
FT_EVAL_EVERY       = 250
FT_CHECKPOINT_EVERY = 500


# -----------------------------------------------------------------------
# Data loader
# -----------------------------------------------------------------------
class FinetuneLoader:
    def __init__(self, data_dir: str, split: str, batch_size: int, seed: int = 0):
        self.tokens = np.load(os.path.join(data_dir, f'{split}_tokens.npy'))
        self.masks  = np.load(os.path.join(data_dir, f'{split}_mask.npy'))
        assert len(self.tokens) == len(self.masks)
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._idx = 0
        self._shuffle()
        print(f"FinetuneLoader [{split}]: {len(self.tokens):,} examples")

    def _shuffle(self):
        perm = self._rng.permutation(len(self.tokens))
        self.tokens = self.tokens[perm]
        self.masks  = self.masks[perm]
        self._idx = 0

    def next_batch(self):
        if self._idx + self.batch_size > len(self.tokens):
            self._shuffle()
        t = self.tokens[self._idx : self._idx + self.batch_size].astype(np.int32)
        m = self.masks [self._idx : self._idx + self.batch_size].astype(np.float32)
        self._idx += self.batch_size
        return t, m


# -----------------------------------------------------------------------
# LR schedule and optimizer
# -----------------------------------------------------------------------
def create_ft_schedule():
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, FT_PEAK_LR, FT_WARMUP_STEPS),
            optax.cosine_decay_schedule(
                FT_PEAK_LR,
                FT_TOTAL_STEPS - FT_WARMUP_STEPS,
                alpha=FT_MIN_LR / FT_PEAK_LR,
            ),
        ],
        boundaries=[FT_WARMUP_STEPS],
    )


def create_ft_optimizer(schedule):
    return optax.chain(
        optax.clip_by_global_norm(FT_GRAD_CLIP),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=FT_WEIGHT_DECAY,
            mask=lambda p: jax.tree_util.tree_map(lambda x: x.ndim >= 2, p),
        ),
    )


# -----------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------
def _drive_base():
    return next(
        (p for p in ['/content/drive/MyDrive', '/drive/MyDrive']
         if os.path.exists(p)),
        None,
    )


def load_pretrain_params(checkpoint_path: str, template_params):
    """Restore only the params pytree from a pretraining checkpoint.

    Orbax requires the item template to match the saved structure exactly.
    We satisfy that by providing a zero-initialised opt_state with the same
    shape tree; we discard it after loading and only return params.
    """
    ckptr = ocp.PyTreeCheckpointer()
    # Build an opt_state template whose *structure* matches the saved checkpoint.
    # Shape/values don't matter — Orbax only uses the template for dtype/shape info.
    opt_state_template = create_ft_optimizer(create_ft_schedule()).init(template_params)
    restored = ckptr.restore(
        checkpoint_path,
        item={
            'params':    template_params,
            'opt_state': opt_state_template,
            'step':      0,
        },
    )
    return restored['params']


def save_ft_checkpoint(state, step: int, tag: str = None):
    label  = tag or f'ft_step_{step:05d}'
    local  = f'/content/checkpoints/{label}'
    drive  = _drive_base()
    remote = f'{drive}/sql-lm-data/checkpoints/{label}' if drive else None

    payload = {
        'params':    state.params,
        'opt_state': state.opt_state,
        'step':      int(state.step),
    }
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(local, payload)

    if remote:
        import shutil
        try:
            os.makedirs(os.path.dirname(remote), exist_ok=True)
            if os.path.exists(remote):
                shutil.rmtree(remote)
            shutil.copytree(local, remote)
            print(f"  Checkpoint → Drive: {label}")
        except Exception as e:
            print(f"  Drive copy failed: {e}")
    else:
        print(f"  Checkpoint saved locally: {local}")


# -----------------------------------------------------------------------
# Train / eval steps
# -----------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=['model'])
def train_step(state, tokens, mask, model, dropout_rng):
    def loss_fn(params):
        logits      = model.apply({'params': params}, tokens, train=True,
                                  rngs={'dropout': dropout_rng})  # [B, T, V]
        pred_logits = logits[:, :-1, :].astype(jnp.float32)      # [B, T-1, V]
        targets     = tokens[:, 1:]                               # [B, T-1]
        loss_mask   = mask[:, :-1]                                # [B, T-1]
        per_token   = optax.softmax_cross_entropy_with_integer_labels(
                          pred_logits, targets)                    # [B, T-1]
        return (per_token * loss_mask).sum() / (loss_mask.sum() + 1e-9)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grad_norm   = optax.global_norm(grads)
    state       = state.apply_gradients(grads=grads)
    return state, loss, grad_norm


@functools.partial(jax.jit, static_argnames=['model'])
def _eval_step(params, tokens, mask, model):
    logits      = model.apply({'params': params}, tokens)
    pred_logits = logits[:, :-1, :].astype(jnp.float32)
    targets     = tokens[:, 1:]
    loss_mask   = mask[:, :-1]
    per_token   = optax.softmax_cross_entropy_with_integer_labels(pred_logits, targets)
    return (per_token * loss_mask).sum() / (loss_mask.sum() + 1e-9)


def evaluate_ft(state, val_loader, model, n_batches: int = 50):
    losses = []
    for _ in range(n_batches):
        t, m = val_loader.next_batch()
        loss = _eval_step(state.params, jnp.array(t), jnp.array(m), model)
        losses.append(float(loss))
    return float(np.mean(losses))


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------
def finetune(
    data_dir:      str = '/content/data/finetune',
    pretrain_ckpt: str = None,   # auto-detected if None
    output_dir:    str = '/content/checkpoints',
):
    print(f"JAX devices: {jax.devices()}")

    # --- Data ---
    train_loader = FinetuneLoader(data_dir, 'train', FT_BATCH_SIZE, seed=0)
    val_loader   = FinetuneLoader(data_dir, 'val',   FT_BATCH_SIZE, seed=1)

    # --- Model ---
    model = SQLTransformer(
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        context_length=CONTEXT_LENGTH,
        rope_base=ROPE_BASE,
        dtype=jnp.bfloat16,
        dropout_rate=FT_DROPOUT,
    )

    # Initialise once to get the params template structure for Orbax restore.
    dummy          = jnp.ones((FT_BATCH_SIZE, CONTEXT_LENGTH), dtype=jnp.int32)
    template_params = model.init(jax.random.PRNGKey(0), dummy)['params']

    # --- Locate pretrain checkpoint ---
    if pretrain_ckpt is None:
        drive = _drive_base()
        candidates = []
        if drive:
            candidates.append(f'{drive}/sql-lm-data/checkpoints/step_76500')
        candidates.append('/content/checkpoints/step_76500')
        pretrain_ckpt = next((p for p in candidates if os.path.exists(p)), None)
        if pretrain_ckpt is None:
            raise FileNotFoundError(
                "step_76500 checkpoint not found. Mount Drive or pass pretrain_ckpt= explicitly."
            )

    print(f"Loading pretrained params from: {pretrain_ckpt}")
    params = load_pretrain_params(pretrain_ckpt, template_params)

    # --- Fine-tune TrainState (fresh optimizer — don't carry pretraining moments) ---
    schedule  = create_ft_schedule()
    optimizer = create_ft_optimizer(schedule)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Parameters: {param_count:,}")

    # --- XLA warm-up compile ---
    print("Compiling train_step...")
    rng = jax.random.PRNGKey(42)
    t0, m0 = train_loader.next_batch()
    rng, dropout_rng = jax.random.split(rng)
    _ = train_step(state, jnp.array(t0), jnp.array(m0), model, dropout_rng)
    print("Compilation complete.")

    # --- Metrics logging ---
    os.makedirs(output_dir, exist_ok=True)
    local_metrics = '/content/ft_metrics.jsonl'
    drive         = _drive_base()
    drive_metrics = f'{drive}/sql-lm-data/ft_metrics.jsonl' if drive else None

    def log_metric(record: dict):
        line = json.dumps(record) + '\n'
        with open(local_metrics, 'a') as f:
            f.write(line)
        if drive_metrics:
            with open(drive_metrics, 'a') as f:
                f.write(line)

    # --- Training loop ---
    print(f"\nFine-tuning: {FT_TOTAL_STEPS} steps | "
          f"batch={FT_BATCH_SIZE} | peak_lr={FT_PEAK_LR:.0e}\n")
    losses = []

    for step in range(1, FT_TOTAL_STEPS + 1):
        t, m = train_loader.next_batch()
        rng, dropout_rng = jax.random.split(rng)
        state, loss, grad_norm = train_step(
            state, jnp.array(t), jnp.array(m), model, dropout_rng)
        losses.append(float(loss))

        if step % 100 == 0:
            avg_loss = float(np.mean(losses[-100:]))
            lr       = float(schedule(step))
            print(f"Step {step:5d}/{FT_TOTAL_STEPS} | "
                  f"loss: {avg_loss:.4f} | "
                  f"grad_norm: {float(grad_norm):.3f} | "
                  f"lr: {lr:.2e}")
            log_metric({
                'step': step, 'kind': 'train',
                'loss': avg_loss, 'grad_norm': float(grad_norm), 'lr': lr,
            })

        if step % FT_EVAL_EVERY == 0:
            val_loss = evaluate_ft(state, val_loader, model)
            print(f"  Val loss: {val_loss:.4f}")
            log_metric({'step': step, 'kind': 'val', 'loss': val_loss})

        if step % FT_CHECKPOINT_EVERY == 0:
            if jnp.isfinite(loss):
                save_ft_checkpoint(state, step)
            else:
                print(f"NaN at step {step}, stopping.")
                break

    save_ft_checkpoint(state, FT_TOTAL_STEPS, tag='ft_final')
    print(f"\nDone. Final train loss: {float(np.mean(losses[-100:])):.4f}")
    return state


if __name__ == '__main__':
    finetune()
