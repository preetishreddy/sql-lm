# Model & Training — Full Context & Implementation Plan

**THIS IS THE SINGLE SOURCE OF TRUTH for model architecture, training, and
the Colab session workflow.** Do not create parallel docs (`TRAINING.md`,
`colab_setup.py`, etc.) — update this file instead. The hyperparameters and
code structure below are locked; the practical bits (install commands,
Drive paths, session count estimate) are kept current here.

This file gives Claude Code everything needed to implement the model
architecture and training loop. Read the entire document before writing
any code. The JAX/TPU programming model has specific constraints that
differ significantly from PyTorch — skipping this document causes bugs
that are hard to diagnose.

Depends on:
- `tokenizer/tokenizer.json` (vocab_size=12288, done)
- `data/tokenized/manifest.json` (2.51B tokens across 11 sources, done)

Target environment:
- Google Colab, **TPU v5e-1** runtime (1 device, 1 TensorCore, 16GB HBM)
- 1-hour session limit (free Colab); checkpoint to Drive between sessions
- Storage: Google Drive only (no GCS available)

---

## Critical Correction From CORPUS.md Section 9

CORPUS.md says "PyTorch Dataset". **This is wrong for our stack.**
We use JAX + Flax + Optax on TPU v5e-1. There is no PyTorch anywhere.
The CorpusLoader is a plain Python class returning numpy arrays.
JAX consumes numpy arrays directly — no DataLoader, no torch.Tensor.

---

## Change Log (Decisions Made After Initial Draft)

### GRAD_ACCUM_STEPS dropped from 2 → 1; BATCH_SIZE halved from 256 → 128
The original spec used gradient accumulation. After in-session testing, the
TPU v5e-1 hit RESOURCE_EXHAUSTED at BATCH_SIZE=256. Two fixes were applied:
1. BATCH_SIZE reduced to 128 — halves activation memory per step.
2. `nn.remat` (gradient checkpointing) applied to each TransformerBlock —
   instead of storing all activations for the backward pass, remat recomputes
   them, trading a 33% compute overhead for a large reduction in peak HBM.

GRAD_ACCUM_STEPS stays at 1 (no accumulation loop). Effective batch is now
65k tokens/step, which doubles TOTAL_STEPS relative to the original 256 plan.
Training throughput is lower but the run is stable.

### 2-epoch training plan
1 epoch = 82 tokens/param (conservative for a specialist model).
2 epochs = 164 tokens/param, which consistently produces better downstream
task performance for domain-specific models. Evidence: TinyLlama trained
at 2,727 tokens/param with strong results; specialist models benefit more
from repetition than general ones. Total tokens: ~5.03B.
Step counts are set for 2 epochs. If you want to stop at 1 epoch, stop
at step 38,354 — the checkpoint at that step is your 1-epoch model.

### train_step_accumulated removed — replaced with jitted train_step
The accumulated version had un-jitted value_and_grad calls inside a Python
loop — every call crossed the Python/XLA boundary, defeating the purpose
of TPU training. With GRAD_ACCUM=1, a single jitted train_step handles
everything correctly.

### Resume-from-checkpoint now mandatory
With 4-11 Colab sessions expected across 2 epochs, resume logic is
load-bearing, not optional. The training loop reads start_step from
the latest checkpoint at session startup.

### Dead scan_fn removed from SQLTransformer
The SQLTransformer previously defined a scan_fn that was never called
(nn.scan was used instead). Removed to avoid confusion.

---

## Architecture Hyperparameters (Final, Frozen)

```python
# Model
VOCAB_SIZE       = 12288   # tokenizer vocab (was 8192, bumped after fertility review)
HIDDEN_DIM       = 384     # d_model — core width of entire model
NUM_LAYERS       = 16      # transformer blocks — deep-narrow per MobileLLM findings
NUM_HEADS        = 6       # attention heads
HEAD_DIM         = 64      # HIDDEN_DIM // NUM_HEADS = 384 // 6 = 64
INTERMEDIATE_DIM = 896     # SwiGLU MLP width (~2.34× hidden_dim)
CONTEXT_LENGTH   = 512     # max sequence length
ROPE_BASE        = 10_000  # RoPE frequency base, sufficient for 512 ctx

# Tokenizer IDs (pinned — do not change)
BOS_ID = 0
EOS_ID = 1
PAD_ID = 2

# Training
TOTAL_TOKENS_PER_EPOCH = 2_513_584_640
NUM_EPOCHS             = 2
TOTAL_TOKENS           = TOTAL_TOKENS_PER_EPOCH * NUM_EPOCHS   # 5,027,169,280

BATCH_SIZE        = 128    # sequences per step (halved from 256 to fix RESOURCE_EXHAUSTED)
GRAD_ACCUM_STEPS  = 1      # no accumulation needed; nn.remat handles activation memory
                            # effective batch = 128 × 512 × 1 = 65,536 tokens/step

TOTAL_STEPS       = 76_708  # TOTAL_TOKENS // (BATCH_SIZE * CONTEXT_LENGTH)
                             # = 5,027,169,280 // 65,536
STEPS_PER_EPOCH   = 38_354  # checkpoint at this step = 1-epoch model

WARMUP_STEPS      = 2_000
STABLE_STEPS      = 70_708  # TOTAL_STEPS - WARMUP_STEPS - DECAY_STEPS
DECAY_STEPS       = 4_000   # slightly longer decay for 2-epoch run
PEAK_LR           = 5e-4
MIN_LR            = 5e-5    # 10% of peak, end of decay

WEIGHT_DECAY      = 0.1
BETA1             = 0.9
BETA2             = 0.95    # NOT 0.999 — critical for LLM training stability
EPSILON           = 1e-8
GRAD_CLIP         = 1.0
EVAL_EVERY        = 500     # steps between validation loss checks
CHECKPOINT_EVERY  = 500     # steps between saves to Drive
```

**Why 16 × 384 and not shallower/wider:**
Reviewed against alternatives (10 × 512, 12 × 448, 8 × 576). The
MobileLLM paper (Meta 2023) specifically studied sub-100M parameter models
and found deep-narrow consistently outperforms wide-shallow at the same
parameter count on downstream tasks. 16 × 384 is the architecture guide's
deliberate choice based on that evidence. 10 × 512 is the most replicated
alternative at this scale — if training is unstable in the first 1000 steps,
that is the fallback. Keep 16 × 384 unless you observe instability.

**Parameter count verification (must match ~30.7M):**
```
Embedding (weight-tied):   12288 × 384          =  4,718,592
Per layer:
  Attn Q:   384 × 384  =  147,456
  Attn K:   384 × 384  =  147,456
  Attn V:   384 × 384  =  147,456
  Attn O:   384 × 384  =  147,456
  MLP gate: 384 × 896  =  344,064
  MLP up:   384 × 896  =  344,064
  MLP down: 896 × 384  =  344,064
  RMSNorm1: 384        =      384
  RMSNorm2: 384        =      384
  Layer total:          =  1,622,784
16 layers:              = 25,964,544
Final RMSNorm:          =      384
Grand total:            ~ 30,683,520  (~30.7M)
```
If your count differs by more than 100k, something is wrong.

---

## JAX Programming Model — Read This Before Writing Anything

JAX is not PyTorch with different syntax. It has fundamentally different
constraints. Violating these produces errors that are hard to diagnose.

### Rule 1: Pure Functions Only
JAX traces functions by running them with abstract values. Any side effect
(print, global variable mutation, random state) inside a jit-compiled
function breaks tracing. The model forward pass must be a pure function:
same inputs → same outputs, no side effects.

```python
# WRONG — print inside jit
@jax.jit
def train_step(state, batch):
    print(f"batch shape: {batch.shape}")  # breaks tracing
    ...

# RIGHT — pure function
@jax.jit
def train_step(state, batch):
    ...  # no prints, no globals, no mutations
```

### Rule 2: jit Everything That Runs in a Loop
Un-jitted functions on TPU are extremely slow — every operation crosses
the Python/XLA boundary. The train_step function MUST be jit-compiled.
First call takes 30-90 seconds to compile. Every subsequent call takes
milliseconds. Do not skip jit to "simplify debugging" — it makes
debugging on TPU nearly impossible due to slowness.

### Rule 3: Use nn.scan for the 16 Transformer Layers
A Python for loop inside jit unrolls all 16 iterations at compile time.
This creates a compiled program 16× larger, uses 16× more memory during
compilation, and takes much longer to compile (sometimes 10+ minutes).

Use `nn.scan` (Flax's version) instead — it compiles one layer and scans
over stacked parameters. Compilation is fast, memory is low.

```python
# WRONG — Python loop inside jit (compiles 16 copies)
def forward(params, x):
    for i in range(16):
        x = transformer_block(x, params['layers'][i])
    return x

# RIGHT — nn.scan (compiles one layer, scans over stacked params)
ScanBlock = nn.scan(
    TransformerBlock,
    variable_axes={'params': 0},
    split_rngs={'params': True},
    length=self.num_layers,
)(...)
x = ScanBlock(x, cos, sin, mask)
```

### Rule 4: Random Keys Are Explicit
JAX has no global random state. Every random operation takes a key.
Keys must be split before use.

```python
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
```

During pretraining dropout=0.0, so random keys are only needed for
weight initialization. Training is otherwise deterministic.

### Rule 5: Shape Errors Appear at Trace Time, Not Runtime
JAX compiles the function on first call with the exact input shapes.
Always use consistent batch sizes. Drop the last partial batch — never
pad to fill it.

### Rule 6: BF16 for Compute, FP32 for Optimizer State
TPU v5e natively supports BF16. Use it for model weights and activations.
Use FP32 for optimizer state (Adam moment estimates) and loss accumulation.

```python
model = SQLTransformer(dtype=jnp.bfloat16)
# Optax AdamW stores state in FP32 by default — no action needed
```

---

## File Structure to Build

```
sql-lm/
  scripts/
    config.py          # all constants — single source of truth
    model.py           # SQLTransformer, TransformerBlock, RoPE, RMSNorm, SwiGLU
    data_loader.py     # CorpusLoader — reads .npy files, yields batches
    train.py           # training loop, WSD schedule, checkpointing
    evaluate.py        # validation loss on val splits
  checkpoints/
    step_XXXXX/        # orbax checkpoint directories
    best/              # copy of best val-loss checkpoint
```

---

## Component 1: config.py

Single source of truth. Every other file imports from here.
Never hardcode a constant that appears in this document.

```python
# scripts/config.py

VOCAB_SIZE        = 12288
HIDDEN_DIM        = 384
NUM_LAYERS        = 16
NUM_HEADS         = 6
HEAD_DIM          = 64
INTERMEDIATE_DIM  = 896
CONTEXT_LENGTH    = 512
ROPE_BASE         = 10_000

BOS_ID = 0
EOS_ID = 1
PAD_ID = 2

TOTAL_TOKENS_PER_EPOCH = 2_513_584_640
NUM_EPOCHS             = 2
TOTAL_TOKENS           = TOTAL_TOKENS_PER_EPOCH * NUM_EPOCHS

BATCH_SIZE        = 128
GRAD_ACCUM_STEPS  = 1
TOTAL_STEPS       = 76_708
STEPS_PER_EPOCH   = 38_354

WARMUP_STEPS      = 2_000
STABLE_STEPS      = 70_708
DECAY_STEPS       = 4_000
PEAK_LR           = 5e-4
MIN_LR            = 5e-5
WEIGHT_DECAY      = 0.1
BETA1             = 0.9
BETA2             = 0.95
EPSILON           = 1e-8
GRAD_CLIP         = 1.0
EVAL_EVERY        = 500
CHECKPOINT_EVERY  = 500
```

---

## Component 2: model.py

### RoPE (Rotary Position Embeddings)

Precompute `(cos, sin)` once outside the model, pass as arguments.
Do not recompute inside jit on every step.

```python
def precompute_freqs(head_dim: int, max_seq_len: int,
                     base: int = 10_000) -> tuple:
    # Returns (cos, sin) of shape [max_seq_len, head_dim // 2]
    theta = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))
    positions = jnp.arange(max_seq_len)
    freqs = jnp.outer(positions, theta)
    return jnp.cos(freqs), jnp.sin(freqs)

def apply_rope(x, cos, sin):
    # x: [batch, seq_len, num_heads, head_dim]
    x1 = x[..., ::2]    # even dimensions
    x2 = x[..., 1::2]   # odd dimensions
    rotated = jnp.concatenate(
        [x1 * cos - x2 * sin,
         x1 * sin + x2 * cos], axis=-1)
    return rotated
```

### RMSNorm

No mean subtraction, no beta parameter. One gamma per hidden dimension.

```python
class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        gamma = self.param('scale', nn.initializers.ones, (self.dim,))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * gamma
```

### SwiGLU MLP

Three projections: gate (SiLU), up (linear), down. Element-wise product
of gate and up before down-projection.

```python
class SwiGLU(nn.Module):
    hidden_dim: int        # 384
    intermediate_dim: int  # 896
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        gate = nn.silu(nn.Dense(self.intermediate_dim,
                                use_bias=False, dtype=self.dtype)(x))
        up   = nn.Dense(self.intermediate_dim,
                        use_bias=False, dtype=self.dtype)(x)
        return nn.Dense(self.hidden_dim,
                        use_bias=False, dtype=self.dtype)(gate * up)
```

### Multi-Head Attention with RoPE

Pre-Norm is applied BEFORE attention (see TransformerBlock below).
RoPE is applied to Q and K only, not V.

```python
class MultiHeadAttention(nn.Module):
    hidden_dim: int
    num_heads: int
    head_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, cos, sin, mask):
        B, T, _ = x.shape

        q = nn.Dense(self.num_heads * self.head_dim,
                     use_bias=False, dtype=self.dtype)(x)
        k = nn.Dense(self.num_heads * self.head_dim,
                     use_bias=False, dtype=self.dtype)(x)
        v = nn.Dense(self.num_heads * self.head_dim,
                     use_bias=False, dtype=self.dtype)(x)

        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K (not V)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        # [B, heads, T, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return nn.Dense(self.hidden_dim, use_bias=False,
                        dtype=self.dtype)(out)
```

### Causal Mask

Build once before training. Shape `[1, 1, T, T]` broadcasts over
batch and head dimensions.

```python
def make_causal_mask(seq_len: int) -> jnp.ndarray:
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    return mask[None, None, :, :]  # [1, 1, T, T]
```

### TransformerBlock

Pre-Norm: normalize input, compute sublayer, add residual.
The residual always carries the un-normalized input.

```python
class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    head_dim: int
    intermediate_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, cos, sin, mask):
        # Attention sublayer
        x = x + MultiHeadAttention(
            self.hidden_dim, self.num_heads,
            self.head_dim, self.dtype)(
            RMSNorm(self.hidden_dim)(x), cos, sin, mask)

        # MLP sublayer
        x = x + SwiGLU(
            self.hidden_dim, self.intermediate_dim,
            self.dtype)(RMSNorm(self.hidden_dim)(x))

        return x
```

### SQLTransformer (Full Model)

Weight tying: `embed` is declared once and used for both input lookup
and output projection (`x @ embed.T`). One matrix, shared.

**Note:** `nn.scan` is used for the 16 layers. There is no `scan_fn`
helper — `nn.scan` wraps `TransformerBlock` directly. Do not add a
separate `scan_fn`; it would be dead code.

```python
class SQLTransformer(nn.Module):
    vocab_size: int        # 12288
    hidden_dim: int        # 384
    num_layers: int        # 16
    num_heads: int         # 6
    head_dim: int          # 64
    intermediate_dim: int  # 896
    context_length: int    # 512
    rope_base: int         # 10_000
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, token_ids, train: bool = False):
        B, T = token_ids.shape

        # Input embedding (std=0.02 init for training stability)
        embed = self.param('embedding',
                           nn.initializers.normal(stddev=0.02),
                           (self.vocab_size, self.hidden_dim))
        x = embed[token_ids].astype(self.dtype)  # [B, T, hidden_dim]

        # RoPE and causal mask (precomputed, not recomputed per step)
        cos, sin = precompute_freqs(self.head_dim, T, self.rope_base)
        mask = make_causal_mask(T)

        # 16 Transformer blocks via nn.scan
        # nn.scan stacks params along axis 0: each weight is [16, ...]
        # Compiles ONE block, scans over it — fast compile, low memory
        ScanBlock = nn.scan(
            TransformerBlock,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            length=self.num_layers,
        )(self.hidden_dim, self.num_heads, self.head_dim,
          self.intermediate_dim, self.dtype)

        x = ScanBlock(x, cos, sin, mask)

        # Final RMSNorm
        x = RMSNorm(self.hidden_dim)(x)

        # LM Head — weight-tied to embedding (same matrix, no new params)
        logits = x @ embed.T  # [B, T, vocab_size]

        return logits
```

---

## Component 3: data_loader.py

Plain Python. No JAX, no PyTorch. Returns numpy arrays.
JAX consumes numpy arrays directly.

```python
import numpy as np
import json
from pathlib import Path
from scripts.config import BATCH_SIZE, CONTEXT_LENGTH

class CorpusLoader:
    def __init__(self, data_dir: str, split: str = 'train',
                 seed: int = 42):
        self.manifest = json.loads(
            (Path(data_dir) / 'manifest.json').read_text())
        self.split = split
        self.rng = np.random.default_rng(seed)

        # mmap_mode='r': reads from disk on demand, no 5GB RAM spike
        self.arrays = {}
        self.sizes = {}
        for name, info in self.manifest['sources'].items():
            path = Path(data_dir) / f'{name}_{split}.npy'
            if path.exists():
                self.arrays[name] = np.load(str(path), mmap_mode='r')
                self.sizes[name] = info[split]['sequences']

        # Sampling probabilities from manifest proportions
        names = list(self.arrays.keys())
        props = np.array([self.manifest['sources'][n]['target_proportion']
                          for n in names])
        self.names = names
        self.probs = props / props.sum()

        total = sum(self.sizes.values())
        print(f"CorpusLoader: {len(self.arrays)} sources, "
              f"{total:,} sequences, split={split}")

    def next_batch(self, batch_size: int = BATCH_SIZE) -> np.ndarray:
        source = self.rng.choice(self.names, p=self.probs)
        arr = self.arrays[source]
        indices = self.rng.integers(0, len(arr), size=batch_size)
        # Cast int16 → int32: JAX embedding lookup requires int32
        return arr[indices].astype(np.int32)  # [batch_size, 512]

    def val_batch_iter(self, batch_size: int, n_batches: int):
        # Representative validation iterator — samples sources by their
        # corpus proportions, same as training. Use this for evaluate().
        # A naive sequential scan would exhaust the first source first
        # (fineweb val alone has 95+ batches at bs=256), producing a
        # fineweb-only val loss that misrepresents the corpus.
        for _ in range(n_batches):
            source = self.rng.choice(self.names, p=self.probs)
            arr    = self.arrays[source]
            i      = int(self.rng.integers(0, len(arr) - batch_size))
            yield arr[i:i+batch_size].astype(np.int32)

    def epoch_iterator(self, batch_size: int = BATCH_SIZE):
        # Sequential scan over every val sequence — use this only when
        # you want EXACT val loss across the full set (slow). For periodic
        # in-training checks, prefer val_batch_iter.
        for arr in self.arrays.values():
            for i in range(0, len(arr) - batch_size, batch_size):
                yield arr[i:i+batch_size].astype(np.int32)
```

---

## Component 4: Learning Rate Schedule (WSD)

Warmup-Stable-Decay. The stable phase holds PEAK_LR for the bulk of
training; the decay phase refines. Can extend training by adding more
stable steps without restarting.

```python
import optax
from scripts.config import (WARMUP_STEPS, STABLE_STEPS, DECAY_STEPS,
                              PEAK_LR, MIN_LR)

schedule = optax.join_schedules(
    schedules=[
        optax.linear_schedule(0.0, PEAK_LR, WARMUP_STEPS),
        optax.constant_schedule(PEAK_LR),
        optax.cosine_decay_schedule(PEAK_LR, DECAY_STEPS,
                                    alpha=MIN_LR / PEAK_LR),
    ],
    boundaries=[WARMUP_STEPS, WARMUP_STEPS + STABLE_STEPS]
)
```

---

## Component 5: train.py

### Optimizer

```python
import optax
from scripts.config import (GRAD_CLIP, BETA1, BETA2, EPSILON,
                              WEIGHT_DECAY)

optimizer = optax.chain(
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
```

### Training State

```python
from flax.training import train_state

class TrainState(train_state.TrainState):
    pass  # default has: step, apply_fn, params, tx, opt_state

def create_train_state(model, rng):
    dummy = jnp.ones((BATCH_SIZE, CONTEXT_LENGTH), dtype=jnp.int32)
    params = model.init(rng, dummy)['params']
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
```

### Loss Function

```python
from scripts.config import VOCAB_SIZE

def loss_fn(params, batch, model):
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
```

### Train Step (jitted, no accumulation)

```python
@jax.jit
def train_step(state, batch):
    (loss, grads) = jax.value_and_grad(loss_fn)(
        state.params, batch, model)
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    return state, loss, grad_norm
```

No accumulation wrapper needed. One batch → one optimizer step.
The jit boundary covers the entire forward + backward + update.

### Main Training Loop

**Resume is mandatory.** With 4-11 sessions expected across 2 epochs,
every session except the first will resume from a checkpoint. The
schedule uses the global `step` number — it must be the real step
from the checkpoint, not reset to zero.

```python
import glob, os, json
import jax.numpy as jnp
from scripts.config import (TOTAL_STEPS, STEPS_PER_EPOCH, BATCH_SIZE,
                              CONTEXT_LENGTH, EVAL_EVERY, CHECKPOINT_EVERY,
                              VOCAB_SIZE, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS,
                              HEAD_DIM, INTERMEDIATE_DIM, ROPE_BASE)

def train():
    loader     = CorpusLoader('data/tokenized', split='train')
    val_loader = CorpusLoader('data/tokenized', split='val')

    model = SQLTransformer(
        vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        head_dim=HEAD_DIM, intermediate_dim=INTERMEDIATE_DIM,
        context_length=CONTEXT_LENGTH, rope_base=ROPE_BASE,
    )

    # --- Build template state first (needed both for fresh start AND
    #     as a restoration template for load_checkpoint) ---
    rng = jax.random.PRNGKey(42)
    state = create_train_state(model, rng)

    # Verify parameter count
    param_count = sum(x.size for x in
                      jax.tree_util.tree_leaves(state.params))
    print(f"Parameters: {param_count:,}")
    assert 30_000_000 < param_count < 33_000_000, \
        f"Unexpected param count: {param_count}"

    # --- Resume from checkpoint if one exists, else fresh start ---
    # Try newest first; fall back if it's corrupt (e.g., Colab died mid
    # Drive-copy and left a half-written dir).
    checkpoints = sorted(glob.glob('/content/checkpoints/step_*'),
                         reverse=True)
    start_step = 0
    for cp in checkpoints:
        try:
            state = load_checkpoint(cp, state)
            start_step = int(os.path.basename(cp).split('_')[1])
            print(f"Resumed from step {start_step:,} / {TOTAL_STEPS:,}")
            break
        except Exception as e:
            print(f"  [warn] {cp} unreadable ({type(e).__name__}); "
                  "trying previous")
    else:
        print(f"Fresh start, training {TOTAL_STEPS:,} steps total")

    # --- Force XLA compilation before timing ---
    print("Compiling train_step (30-90s first session, ~5s after)...")
    dummy = jnp.array(loader.next_batch())
    state, loss, grad_norm = train_step(state, dummy)
    print(f"Compiled. Current loss: {float(loss):.4f}")

    # --- Training loop ---
    losses = []
    best_val_loss = float('inf')
    metrics_path = '/drive/MyDrive/sql-lm-data/metrics.jsonl'

    def log_metric(record: dict):
        # Append one JSON line per log event. Lives on Drive so it
        # persists across sessions and can be plotted from any notebook
        # even while training is running.
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    for step in range(start_step + 1, TOTAL_STEPS + 1):
        batch = jnp.array(loader.next_batch())
        state, loss, grad_norm = train_step(state, batch)
        losses.append(float(loss))

        # Log every 100 steps
        if step % 100 == 0:
            avg = sum(losses[-100:]) / min(len(losses), 100)
            lr  = float(schedule(step))
            print(f"Step {step:6d}/{TOTAL_STEPS} | "
                  f"loss: {avg:.4f} | "
                  f"grad_norm: {float(grad_norm):.3f} | "
                  f"lr: {lr:.2e}")
            log_metric({'step': step, 'kind': 'train',
                        'loss': avg, 'grad_norm': float(grad_norm),
                        'lr': lr})

        # Note when epoch 1 completes
        if step == STEPS_PER_EPOCH:
            print(f"=== Epoch 1 complete (step {step}) "
                  f"— this checkpoint is your 1-epoch model ===")

        # Validation
        if step % EVAL_EVERY == 0:
            val_loss = evaluate(state, val_loader, model)
            print(f"  Val loss: {val_loss:.4f}")
            log_metric({'step': step, 'kind': 'val', 'loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(state, step, tag='best')

        # Checkpoint — guard against NaN
        if step % CHECKPOINT_EVERY == 0:
            if jnp.isfinite(loss).item():
                save_checkpoint(state, step)
            else:
                print(f"NaN at step {step} — skipping save.")
                print("Revert to last good checkpoint, "
                      "halve PEAK_LR, and resume.")
                log_metric({'step': step, 'kind': 'nan'})
                break
```

**Expected loss curve:**
```
Step      0: ~9.4   (random, log(12288))
Step    200: ~7-8   (token frequency learning)
Step  1,000: ~5-6   (SQL patterns emerging)
Step  2,000: ~4-5   (warmup complete)
Step  6,000: ~3-3.5 (mid epoch 1)
Step 20,000: ~2.5-3 (late epoch 1)
Step 38,354: ~2.0-2.5 (epoch 1 complete)
Step 76,708: ~1.5-2.2 (epoch 2 complete)
```
Loss below 1.0 before step 10,000 = data bug (likely repeated data).
Loss above 5.0 after step 6,000 = training bug (LR, jit, beta2).

---

## Component 6: Checkpointing

`flax.training.TrainState` holds `apply_fn` (= `model.apply`) and `tx`
(= the optax optimizer). Both are Python callables — Orbax cannot
serialize them. Save **only** the JAX-array parts (`params`, `opt_state`,
`step`), and restore by injecting them into a fresh template state.

```python
import orbax.checkpoint as ocp
import shutil

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

    # Copy to Drive immediately — don't wait until end of session
    shutil.copytree(local, remote, dirs_exist_ok=True)
    print(f"Checkpoint saved: {label}")

def load_checkpoint(path: str, template_state):
    """Restore params/opt_state/step into the structure of `template_state`."""
    ckptr = ocp.PyTreeCheckpointer()
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
```

**Resume requires building the template state first.** The training loop
in Component 5 must `create_train_state(...)` BEFORE calling
`load_checkpoint(latest, state)`. The fresh state's `params` and
`opt_state` have the correct pytree shapes that Orbax needs as a
restoration template — they get overwritten by the saved arrays.

**NaN guard in the training loop prevents writing garbage over good
checkpoints.** If loss → NaN: the save is skipped, training halts,
you revert to the last good checkpoint on Drive, halve PEAK_LR in
config.py, and resume. Most divergences happen before step 1,000.
If training survives past step 1,000 it almost always finishes cleanly.

---

## Component 7: evaluate.py

```python
@jax.jit
def eval_step(state, batch):
    return loss_fn(state.params, batch, model)

def evaluate(state, val_loader, model,
             n_batches: int = 50) -> float:
    # Uses val_batch_iter — samples representatively across all sources,
    # so the reported val loss reflects whole-corpus generalization, not
    # just whichever source happens to be first.
    losses = [
        float(eval_step(state, jnp.array(batch)))
        for batch in val_loader.val_batch_iter(BATCH_SIZE, n_batches)
    ]
    return sum(losses) / len(losses)
```

---

## Colab Session Startup Script

Run these cells at the start of every session, before any training code.

```python
# Cell 0: Clone repo
# Use Python subprocess + shutil — avoids Colab shell cwd ambiguity and
# the URL-mangling bug that occurs when passing a path to !git clone.
import os, shutil, subprocess, sys

if os.path.exists('/content/sql-lm'):
    shutil.rmtree('/content/sql-lm')

subprocess.run(
    ['git', 'clone', 'https://github.com/preetishreddy/sql-lm.git', '/content/sql-lm'],
    check=True
)

sys.path = [p for p in sys.path if 'sql-lm' not in p]
sys.path.insert(0, '/content/sql-lm')
print("scripts:", os.listdir('/content/sql-lm/scripts'))
print("sys.path[0]:", sys.path[0])
```

```python
# Cell 1: Mount Drive and copy data
import sys, os
from google.colab import drive
drive.mount('/drive')

# Make the repo root importable as a package.
# Adjust the path below if you cloned to a different location.
REPO_ROOT = '/content/sql-lm'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shutil, os, glob

for d in ['/content/data/tokenized', '/content/tokenizer',
          '/content/checkpoints']:
    os.makedirs(d, exist_ok=True)

# Copy tokenized corpus (~5GB, 2-3 min)
print("Copying corpus...")
shutil.copytree('/drive/MyDrive/sql-lm-data/tokenized',
                '/content/data/tokenized', dirs_exist_ok=True)

shutil.copy('/drive/MyDrive/sql-lm-data/tokenizer/tokenizer.json',
            '/content/tokenizer/tokenizer.json')

# Copy latest checkpoint if resuming
ckpts = sorted(glob.glob('/drive/MyDrive/sql-lm-data/checkpoints/step_*'))
if ckpts:
    latest = ckpts[-1]
    name = os.path.basename(latest)
    print(f"Resuming from checkpoint: {name}")
    shutil.copytree(latest, f'/content/checkpoints/{name}',
                    dirs_exist_ok=True)
else:
    print("No checkpoint found — fresh start.")

print("Ready.")
```

```python
# Cell 2: Install dependencies
# DO NOT reinstall jax[tpu] — Colab TPU runtimes come with the exact
# libtpu version matching the kernel. Reinstalling breaks TPU integration
# silently, producing cryptic XLA errors.
!pip install -q -U flax optax orbax-checkpoint tokenizers tqdm
```

```python
# Cell 3: Verify TPU
import jax
print(f"JAX: {jax.__version__}")
print(f"Devices: {jax.devices()}")
# Expect: [TpuDevice(id=0, ...)]
# v5e-1 = ONE device. Use jit, not pmap.
```

```python
# Cell 4: Enable persistent XLA compile cache
# Without this: 60-90s recompilation at the start of every session.
# With this: ~5s on session 2 onwards. Cache is ~10MB on Drive.
jax.config.update('jax_compilation_cache_dir',
                  '/drive/MyDrive/sql-lm-data/xla-cache')
jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1)
```

### Session Count Estimate

```
Effective batch:   65,536 tokens/step (128 seq × 512 tokens)
Total steps:       76,708  (2 epochs)
TPU v5e-1 speed:   ~100k–180k tokens/sec (nn.remat adds ~33% compute overhead)

At 100k tok/s:  5.03B tokens ÷ 100k = 13.9h = ~17 sessions
At 140k tok/s:  5.03B tokens ÷ 140k = 9.9h  = ~12 sessions
At 180k tok/s:  5.03B tokens ÷ 180k = 7.8h  = ~10 sessions
```

Each session: ~3 min setup + ~5s compile (after session 1) + ~50 min training.
Checkpoints every 500 steps. A session death loses at most 500 steps.

---

## Crash & Resume Behavior

**What happens if Colab kills the runtime mid-training:**

1. Up to the last checkpoint (≤500 steps) is lost. At realistic throughput
   that's 5-15 minutes of compute.
2. The next session's Cell 1 copies the latest checkpoint folder from
   Drive back to `/content/checkpoints/`.
3. `train()` builds a fresh template state, then loops over checkpoints
   newest-first and loads the first one that restores cleanly. If the
   newest dir is partial (Colab died mid-Drive-copy), the previous one is
   used and ~500 more steps are repeated. Still resumes; just a small
   double-counting of training tokens.

**Why local Orbax saves are safe:** Orbax writes to a `.tmp` subdir then
renames atomically. A crash during local save leaves the previous good
checkpoint intact — nothing corrupt is ever named `step_NNNNN/`.

**Why Drive copies need defense:** `shutil.copytree` is not atomic.
A half-copied directory on Drive looks valid by name but fails to load.
The newest-first-with-fallback loop handles this.

**Don't delete old checkpoints during training.** The fallback only works
if at least two complete checkpoints exist on Drive. With CHECKPOINT_EVERY=500
that's ~16 MB per checkpoint × keep last 3 = ~50 MB on Drive. Cheap.
Add cleanup logic that keeps only the last 3 + the best, if Drive fills up.

---

## Pre-flight Verification (run before training)

Before kicking off a multi-hour training run, verify every component
works end-to-end. Each check is independent and produces a clear
pass/fail. Run this as a single Colab cell after Cell 4 (XLA cache).
If any check fails, **do not start training** — fix the failure first.

```python
# Cell 5: Pre-flight checks. Run once per session before train.py.
import jax, jax.numpy as jnp, numpy as np, json
from pathlib import Path
from tokenizers import Tokenizer

print("="*60)
print("PRE-FLIGHT VERIFICATION")
print("="*60)

# --- 1. Tokenizer loads and round-trips ---
tok = Tokenizer.from_file('/content/tokenizer/tokenizer.json')
sample = "SELECT name FROM users WHERE id = 1;"
ids = tok.encode(sample).ids
back = tok.decode(ids)
assert tok.get_vocab_size() == 12288, f"vocab {tok.get_vocab_size()} != 12288"
assert 'SELECT' in back and 'users' in back, f"decode failed: {back!r}"
print(f"[1] tokenizer OK   ({len(ids)} tok roundtrip: {back!r})")

# --- 2. Manifest + all 22 .npy files present ---
manifest = json.loads(Path('/content/data/tokenized/manifest.json').read_text())
assert manifest['total_train_tokens'] > 2.5e9, "corpus suspiciously small"
for name in manifest['sources']:
    for split in ['train', 'val']:
        p = Path(f'/content/data/tokenized/{name}_{split}.npy')
        assert p.exists(), f"missing {p}"
print(f"[2] corpus OK      ({manifest['total_train_tokens']:,} train tokens, "
      f"{len(manifest['sources'])} sources)")

# --- 3. Data loader yields the right shape/dtype, and tokens decode ---
from scripts.data_loader import CorpusLoader
loader = CorpusLoader('/content/data/tokenized', split='train')
batch = loader.next_batch(4)
assert batch.shape == (4, 512), f"batch shape {batch.shape}"
assert batch.dtype == np.int32, f"batch dtype {batch.dtype}"
assert batch.min() >= 0 and batch.max() < 12288, "tokens out of range"
preview = tok.decode(batch[0][:60].tolist())
print(f"[3] dataloader OK  (sample: {preview[:80]!r}...)")

# --- 4. Model initializes with expected param count ---
from scripts.model import SQLTransformer
from scripts.config import *
model = SQLTransformer(
    vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS, num_heads=NUM_HEADS, head_dim=HEAD_DIM,
    intermediate_dim=INTERMEDIATE_DIM, context_length=CONTEXT_LENGTH,
    rope_base=ROPE_BASE,
)
params = model.init(jax.random.PRNGKey(0),
                    jnp.ones((1, 512), dtype=jnp.int32))['params']
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
assert 30_000_000 < n_params < 33_000_000, \
    f"param count {n_params:,} outside expected range"
print(f"[4] model OK       ({n_params:,} params)")

# --- 5. Forward pass produces finite loss near ln(vocab) at init ---
logits = model.apply({'params': params}, jnp.array(batch))
import optax
loss0 = optax.softmax_cross_entropy_with_integer_labels(
    logits[:, :-1].reshape(-1, VOCAB_SIZE),
    jnp.array(batch[:, 1:]).reshape(-1)
).mean()
expected = float(jnp.log(VOCAB_SIZE))   # ≈ 9.42
assert jnp.isfinite(loss0), f"loss is {loss0} (NaN/Inf)"
assert abs(float(loss0) - expected) < 1.5, \
    f"init loss {float(loss0):.2f} far from ln(vocab)={expected:.2f}"
print(f"[5] forward OK     (loss {float(loss0):.3f}, expected ~{expected:.2f})")

# --- 6. Backward pass runs without error ---
def lf(p): return optax.softmax_cross_entropy_with_integer_labels(
    model.apply({'params': p}, jnp.array(batch))[:, :-1].reshape(-1, VOCAB_SIZE),
    jnp.array(batch[:, 1:]).reshape(-1)).mean()
grads = jax.grad(lf)(params)
gnorm = float(optax.global_norm(grads))
assert jnp.isfinite(gnorm), f"grad norm is {gnorm}"
print(f"[6] backward OK    (grad norm {gnorm:.3f})")

# --- 7. LR schedule produces expected values at key steps ---
from scripts.train import schedule  # the optax schedule object
assert abs(float(schedule(0))               - 0.0    ) < 1e-9
assert abs(float(schedule(WARMUP_STEPS))    - PEAK_LR) < 1e-6
assert abs(float(schedule(TOTAL_STEPS))     - MIN_LR ) < 1e-5
print(f"[7] schedule OK    (warmup→{PEAK_LR}→decay→{MIN_LR})")

# --- 8. Drive paths writable ---
test = Path('/drive/MyDrive/sql-lm-data/.preflight_test')
test.write_text('ok'); test.unlink()
print(f"[8] drive write OK")

print("="*60)
print("ALL CHECKS PASSED — safe to start training")
print("="*60)
```

Total runtime: ~30 seconds (the model init and compile are the slow
parts). If any assert fires, the message tells you which component
to fix.

### What the pre-flight does NOT cover

- **Corpus quality** — that's `scripts/verify_pipeline.py` from the
  CORPUS.md pipeline (10 checks on shapes, IDs, packing, sha256s).
  Run it once after you build the corpus; no need to re-run per session.
- **Long-run dynamics** — whether loss actually decreases, whether
  optimization is stable. That's the metrics.jsonl + plot loop.
- **Checkpoint round-trip** — the first `save_checkpoint` + restore
  cycle is the real test. After step 500 in your first session,
  manually load the saved checkpoint into a second template state
  and check that params are identical to the running state. If they
  match, you're safe for all subsequent sessions.

### Logging layer summary

| Layer | What it checks | When |
|---|---|---|
| `verify_pipeline.py` | Corpus integrity (the .npy files) | Once, after building corpus |
| Pre-flight cell above | Tokenizer + dataloader + model + forward + backward + schedule + drive | Once per session, before train.py |
| Training stdout logs | Loss, grad_norm, lr every 100 steps | Continuously during training |
| `metrics.jsonl` on Drive | Same as stdout but persistent + plottable | Continuously, survives crashes |
| Plot cell | Visual: loss curve, grad norm, lr schedule, NaN markers | Anytime, from any notebook |
| Best-val checkpoint | The model that generalized best so far | Updated after each eval |

If a check fails in this stack, you know exactly where to look — you
don't have to discover at step 5,000 that the dataloader was returning
zeros all along.

---

## Monitoring & Visualization

The training loop appends one JSON line per log event to
`/drive/MyDrive/sql-lm-data/metrics.jsonl`. Each line is either:

```json
{"step": 100,  "kind": "train", "loss": 8.21, "grad_norm": 1.34, "lr": 5e-5}
{"step": 500,  "kind": "val",   "loss": 5.87}
{"step": 9999, "kind": "nan"}
```

This file persists across sessions and grows by ~50 KB per Colab session.
You can read and plot it from *any* notebook at any time — including while
training is running in another tab.

### Plot cell (run in any Colab notebook)

```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

path = '/drive/MyDrive/sql-lm-data/metrics.jsonl'
records = [json.loads(l) for l in Path(path).read_text().splitlines()]

train = [(r['step'], r['loss'], r['grad_norm']) for r in records if r['kind'] == 'train']
val   = [(r['step'], r['loss'])                  for r in records if r['kind'] == 'val']
nans  = [r['step'] for r in records if r['kind'] == 'nan']

fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

# Loss
ts, tl, _ = zip(*train) if train else ([], [], [])
vs, vl    = zip(*val)   if val   else ([], [])
ax[0].plot(ts, tl, label='train', linewidth=1)
ax[0].plot(vs, vl, 'o-', label='val', markersize=5)
ax[0].set_ylabel('loss'); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_yscale('log')

# Gradient norm
_, _, gn = zip(*train) if train else ([], [], [])
ax[1].plot(ts, gn, color='orange', linewidth=1)
ax[1].axhline(1.0, color='red', linestyle='--', label='GRAD_CLIP=1.0')
ax[1].set_ylabel('grad norm'); ax[1].legend(); ax[1].grid(alpha=0.3)

# Learning rate
lrs = [r['lr'] for r in records if r['kind'] == 'train']
ax[2].plot(ts, lrs, color='green', linewidth=1)
ax[2].set_ylabel('lr'); ax[2].set_xlabel('step'); ax[2].grid(alpha=0.3)

# Mark NaN events
for s in nans:
    for a in ax: a.axvline(s, color='red', alpha=0.4)

plt.tight_layout()
plt.show()
```

### What to look for on the plots

| Plot | Healthy | Trouble |
|---|---|---|
| **Train loss** | Smooth log-linear-ish decrease, ~9.4 → ~2.0 over 38k steps | Plateau, NaN spike, sudden jump up |
| **Val loss** | Tracks train loss within ~0.1-0.3, decreasing monotonically | Diverges from train loss → overfitting; flat → underfitting |
| **Grad norm** | Mostly 0.1-1.0; brief spikes to 2-3 OK in first 1000 steps | Constantly clipped at 1.0 → LR too high; under 0.01 → vanishing grads |
| **LR** | Linear warmup → flat plateau → cosine decay (the WSD shape) | If it's a different shape, the schedule wasn't wired right |

### Real-time monitoring during a session

The print logs in the running notebook show the same info per 100 steps.
For live curves while training, open a **second notebook** on the same Drive
(read-only is fine), run the plot cell, and re-execute it every few minutes —
each refresh reads the up-to-date JSONL.

TensorBoard works too if you prefer (`pip install tensorboard`, write summaries
in the train loop, `%tensorboard --logdir /drive/MyDrive/sql-lm-data/tb`) but adds
dependencies and a writer object to manage. JSONL + matplotlib is enough.

---

## Training Health Checks

### Warning Signs

| Symptom | Likely Cause | Fix |
|---|---|---|
| Loss stays at 9.4 after 500 steps | jit not applied, or LR too low | Verify @jax.jit on train_step |
| Loss spikes to 15+ then recovers | LR too high | Reduce PEAK_LR to 3e-4 |
| Loss NaN after ~1000 steps | Gradient explosion or BF16 overflow | Check GRAD_CLIP, check dtype |
| Loss oscillates, won't decrease | beta2=0.999 (wrong default) | Confirm beta2=0.95 in optimizer |
| Loss < 1.0 before step 5,000 | Repeated data in corpus | Check manifest for duplicates |
| Compile takes > 5 min | Python loop inside jit (not nn.scan) | Switch to nn.scan |
| Step 1 takes 30+ seconds | train_step not jitted | Add @jax.jit |

### Gradient Norm
Healthy range: 0.1 – 1.0.
- Consistently > 5.0: reduce LR or increase GRAD_CLIP
- Consistently < 0.01: LR too low or vanishing gradients

---

## Common Mistakes to Avoid

- **Do not use beta2=0.999.** It is the optax default. We set 0.95
  explicitly. Using 0.999 slows convergence and causes instability
  near LR schedule transitions.

- **Do not use a Python for loop for the 16 layers inside jit.**
  Use `nn.scan`. Unrolling 16 layers at compile time causes 5-10 min
  compilations and wastes HBM.

- **Do not forget @jax.jit on train_step.** Un-jitted TPU training
  is 100-1000× slower. If step 1 takes more than 5 seconds after
  the initial compilation, jit is not applied.

- **Do not apply weight decay to 1D parameters.** RMSNorm scale
  vectors must be excluded. The ndim >= 2 mask in the optimizer
  handles this — do not remove it.

- **Do not forget to cast int16 → int32.** The .npy corpus files use
  int16 to halve disk size. JAX embedding lookup requires int32.
  Always call `.astype(np.int32)` in the data loader.

- **Do not use dropout during pretraining.** dropout=0.0 for all
  pretraining. Relevant only for finetuning.

- **Do not hardcode batch size in the model.** Use `B, T = x.shape`
  everywhere. Eval uses different batch sizes than training.

- **Do not save checkpoints only at the end.** With 7-11 sessions,
  you WILL need mid-training checkpoints. The NaN guard in the loop
  prevents writing bad state over good state.

- **Do not reinstall jax[tpu] in Colab.** The runtime has the exact
  matching libtpu pre-installed. Overwriting it breaks TPU silently.

- **Do not skip the XLA compile cache.** Without it, every session
  wastes 60-90s on recompilation. The cache directory on Drive is
  ~10MB and reused automatically.