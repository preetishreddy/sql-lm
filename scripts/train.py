import os
import glob
import json
import time
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import shutil
import numpy as np
from flax.training import train_state
from typing import Any

from scripts.config import *
from scripts.model import SQLTransformer
from scripts.data_loader import CorpusLoader
from scripts.evaluate import evaluate, loss_fn

# --- Learning Rate Schedule (WSD: Warmup-Stable-Decay) ---
def create_learning_rate_schedule():
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, PEAK_LR, WARMUP_STEPS),
            optax.constant_schedule(PEAK_LR),
            optax.cosine_decay_schedule(PEAK_LR, DECAY_STEPS,
                                        alpha=MIN_LR / PEAK_LR),
        ],
        boundaries=[WARMUP_STEPS, WARMUP_STEPS + STABLE_STEPS]
    )

# Module-level export so pre-flight can do: from scripts.train import schedule
schedule = create_learning_rate_schedule()

# --- Optimizer ---
def create_optimizer(schedule):
    return optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adamw(
            learning_rate=schedule,
            b1=BETA1,
            b2=BETA2,          # 0.95 — NOT optax default of 0.999
            eps=EPSILON,
            weight_decay=WEIGHT_DECAY,
            # Apply weight decay to matrices only (ndim >= 2)
            # NOT to RMSNorm scale vectors or any 1D params
            mask=lambda p: jax.tree_util.tree_map(
                lambda x: x.ndim >= 2, p)
        )
    )

# --- Training State ---
class TrainState(train_state.TrainState):
    # optax.TrainState already has step, apply_fn, params, tx, opt_state
    pass

def create_train_state(model, rng, optimizer):
    dummy = jnp.ones((BATCH_SIZE, CONTEXT_LENGTH), dtype=jnp.int32)
    variables = model.init(rng, dummy)
    params = variables['params']
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

# --- Checkpointing (Orbax) ---
def save_checkpoint(state, step: int, tag: str = None):
    label = tag if tag else f'step_{step:05d}'
    local  = f'/content/checkpoints/{label}'
    remote = f'/drive/MyDrive/sql-lm-data/checkpoints/{label}'

    # Strip apply_fn and tx — they aren't pytree-serializable.
    payload = {
        'params':    state.params,
        'opt_state': state.opt_state,
        'step':      int(state.step),
    }

    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(local, payload)

    # Copy to Drive immediately if in Colab environment
    if os.path.exists('/drive/MyDrive'):
        try:
            os.makedirs(os.path.dirname(remote), exist_ok=True)
            if os.path.exists(remote):
                shutil.rmtree(remote)
            shutil.copytree(local, remote)
            print(f"Checkpoint copied to Drive: {label}")
        except Exception as e:
            print(f"Failed to copy checkpoint to Drive: {e}")
    else:
        print(f"Checkpoint saved locally: {local}")

def load_checkpoint(path: str, template_state):
    """Restore params/opt_state/step into the structure of `template_state`."""
    ckptr = ocp.PyTreeCheckpointer()
    # Orbax restore needs a template 'item' to know the structure
    payload = ckptr.restore(path, item={
        'params':    template_state.params,
        'opt_state': template_state.opt_state,
        'step':      0,
    })
    return template_state.replace(
        params=payload['params'],
        opt_state=payload['opt_state'],
        step=payload['step'],
    )

import functools

# --- Training functions ---
@functools.partial(jax.jit, static_argnames=['model'])
def train_step(state, batch, model):
    def loss_w_model(params):
        return loss_fn(params, batch, model)
    
    (loss, grads) = jax.value_and_grad(loss_w_model)(state.params)
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    return state, loss, grad_norm

def train():
    # Detect devices
    print(f"JAX Devices: {jax.devices()}")
    
    # Absolute path — Colab copies corpus to /content/data/tokenized via Cell 1.
    data_dir = '/content/data/tokenized'
    loader     = CorpusLoader(data_dir, split='train')
    val_loader = CorpusLoader(data_dir, split='val')
    
    # Initialize model
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
    )

    # Initialize optimizer and state
    schedule = create_learning_rate_schedule()
    optimizer = create_optimizer(schedule)
    
    rng = jax.random.PRNGKey(42)
    state = create_train_state(model, rng, optimizer)

    # Parameter count
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Parameters: {param_count:,}")

    # Resume from checkpoint logic
    start_step = 0
    checkpoint_dir = '/content/checkpoints'
    drive_checkpoint_dir = '/drive/MyDrive/sql-lm-data/checkpoints'
    
    # Check Drive first if available
    search_dirs = []
    if os.path.exists(drive_checkpoint_dir):
        search_dirs.append(drive_checkpoint_dir)
    if os.path.exists(checkpoint_dir):
        search_dirs.append(checkpoint_dir)
        
    latest_cp = None
    for s_dir in search_dirs:
        cps = sorted(glob.glob(os.path.join(s_dir, 'step_*')), reverse=True)
        if cps:
            latest_cp = cps[0]
            break
            
    if latest_cp:
        try:
            state = load_checkpoint(latest_cp, state)
            start_step = int(state.step)
            print(f"Resumed from step {start_step:,} / {TOTAL_STEPS:,}")
        except Exception as e:
            print(f"Failed to load checkpoint {latest_cp}: {e}. Starting fresh.")

    # Main Loop
    losses = []
    local_metrics  = '/content/metrics.jsonl'
    drive_metrics  = '/drive/MyDrive/sql-lm-data/metrics.jsonl'

    def log_metric(record: dict):
        line = json.dumps(record) + '\n'
        # Always write locally — readable in this notebook with no sync lag.
        with open(local_metrics, 'a') as f:
            f.write(line)
        # Also mirror to Drive so it survives session death and is readable
        # from other notebooks (may have a short sync lag in a second notebook).
        if os.path.exists('/drive/MyDrive'):
            with open(drive_metrics, 'a') as f:
                f.write(line)

    print(f"Starting training from step {start_step}...")

    # Trigger XLA compilation without consuming a training step.
    # jax.jit caches by input shape/dtype, so the real loop reuses this compilation.
    print("Compiling train_step...")
    dummy_batch = jnp.array(loader.next_batch())
    _ = train_step(state, dummy_batch, model)
    print("Compilation complete.")

    for step in range(start_step + 1, TOTAL_STEPS + 1):
        batch = jnp.array(loader.next_batch())
        state, loss, grad_norm = train_step(state, batch, model)
        
        losses.append(float(loss))

        if step % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            lr = schedule(step)
            print(f"Step {step:6d}/{TOTAL_STEPS} | loss: {avg_loss:.4f} | grad_norm: {float(grad_norm):.3f} | lr: {lr:.2e}")
            log_metric({
                'step': step, 
                'kind': 'train', 
                'loss': avg_loss, 
                'grad_norm': float(grad_norm), 
                'lr': float(lr)
            })

        if step % EVAL_EVERY == 0:
            val_loss = evaluate(state, val_loader, model)
            print(f"  Val loss: {val_loss:.4f}")
            log_metric({'step': step, 'kind': 'val', 'loss': val_loss})

        if step % CHECKPOINT_EVERY == 0:
            if jnp.isfinite(loss):
                save_checkpoint(state, step)
            else:
                print(f"NaN detected at step {step}, skipping checkpoint.")
                break

if __name__ == "__main__":
    train()
