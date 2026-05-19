# -*- coding: utf-8 -*-
"""sql-lm — Fine-Tuning v2

Trains v2 of the fine-tuned model from the step_76500 pretrain checkpoint.

Changes vs v1:
  - dropout = 0.1 (was 0.0)
  - 5,000 steps (was 3,000)
  - BIRD 5× (was 3×), gretelai 0.25× (was 0.5×)
  - Dataset already built and saved to Drive as finetune_v2/

Session resume: if Colab disconnects, run cells 1–4 again then jump to
the "Resume" cell — finetune() auto-detects the latest local checkpoint.
"""

# ===========================================================================
# Cell 1 — Clone repo
# ===========================================================================

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

# ===========================================================================
# Cell 2 — Mount Drive + install deps
# ===========================================================================

from google.colab import drive
drive.mount('/content/drive')

get_ipython().system('pip install -q -U flax optax orbax-checkpoint tokenizers')

# ===========================================================================
# Cell 3 — XLA compile cache
# ===========================================================================

import jax
print(f"JAX: {jax.__version__}  |  devices: {jax.devices()}")

jax.config.update('jax_compilation_cache_dir',
                  '/content/drive/MyDrive/sql-lm-data/xla-cache')
jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1)

# ===========================================================================
# Cell 4 — Copy data from Drive to local /content
# ===========================================================================

import shutil, os, glob

# Tokenizer
os.makedirs('/content/tokenizer', exist_ok=True)
shutil.copy('/content/drive/MyDrive/sql-lm-data/tokenizer/tokenizer.json',
            '/content/tokenizer/tokenizer.json')
print("Tokenizer copied.")

# v2 fine-tune dataset (built last session — skip the 15-min rebuild)
os.makedirs('/content/data/finetune', exist_ok=True)
shutil.copytree('/content/drive/MyDrive/sql-lm-data/finetune_v2',
                '/content/data/finetune', dirs_exist_ok=True)
print("v2 finetune dataset copied.")

# Pretrain checkpoint (step_76500) — starting point for v2
os.makedirs('/content/checkpoints', exist_ok=True)
ckpt_src = '/content/drive/MyDrive/sql-lm-data/checkpoints/step_76500'
ckpt_dst = '/content/checkpoints/step_76500'
if not os.path.exists(ckpt_dst):
    print("Copying step_76500 checkpoint (~200MB)...")
    shutil.copytree(ckpt_src, ckpt_dst)
    print("Done.")
else:
    print("step_76500 already local.")

# ===========================================================================
# Cell 5 — Quick sanity check before training
# ===========================================================================

import jax.numpy as jnp, numpy as np
from tokenizers import Tokenizer
from scripts.model import SQLTransformer
from scripts.config import *
from scripts.finetune import FT_DROPOUT, FT_TOTAL_STEPS

tok = Tokenizer.from_file('/content/tokenizer/tokenizer.json')
assert tok.get_vocab_size() == 12288

model = SQLTransformer(
    vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS, head_dim=HEAD_DIM, intermediate_dim=INTERMEDIATE_DIM,
    context_length=CONTEXT_LENGTH, rope_base=ROPE_BASE,
    dtype=jnp.bfloat16, dropout_rate=FT_DROPOUT,
)
params = model.init(jax.random.PRNGKey(0),
                    jnp.ones((1, CONTEXT_LENGTH), dtype=jnp.int32))['params']
n = sum(x.size for x in jax.tree_util.tree_leaves(params))
assert 30_000_000 < n < 33_000_000, f"param count {n:,} unexpected"

tokens = np.load('/content/data/finetune/train_tokens.npy')
masks  = np.load('/content/data/finetune/train_mask.npy')

print(f"Model:   {n:,} params  |  dropout={FT_DROPOUT}  |  steps={FT_TOTAL_STEPS}")
print(f"Dataset: {len(tokens):,} train examples  |  avg mask={masks.mean():.3f}")
print(f"Checkpoint: {ckpt_dst}")
print("All checks passed — ready to train.")

# ===========================================================================
# Cell 6 — Run fine-tuning (Session 1: steps 1–2500)
# ===========================================================================

from scripts.finetune import finetune
state = finetune()

# ===========================================================================
# Cell 7 — RESUME (Session 2+): copy latest ft checkpoint from Drive, then run
#
# Run this cell instead of Cell 6 when resuming after a Colab disconnect.
# ===========================================================================

import shutil, os, glob

# Copy latest ft checkpoint from Drive back to local
ft_ckpts = sorted(glob.glob(
    '/content/drive/MyDrive/sql-lm-data/checkpoints/ft_step_*'))
if ft_ckpts:
    latest = ft_ckpts[-1]
    name   = os.path.basename(latest)
    dst    = f'/content/checkpoints/{name}'
    if not os.path.exists(dst):
        print(f"Copying {name} from Drive...")
        shutil.copytree(latest, dst)
    print(f"Resuming from {name}")
else:
    print("No ft checkpoint found on Drive — will start from step_76500.")

from scripts.finetune import finetune
state = finetune()

# ===========================================================================
# Cell 8 — Plot training curve (run any time during or after training)
# ===========================================================================

import json, matplotlib.pyplot as plt
from pathlib import Path

metrics_path = '/content/drive/MyDrive/sql-lm-data/ft_metrics.jsonl'
records = [json.loads(l) for l in Path(metrics_path).read_text().splitlines() if l.strip()]

train_r = [(r['step'], r['loss'], r['grad_norm']) for r in records if r['kind'] == 'train']
val_r   = [(r['step'], r['loss'])                 for r in records if r['kind'] == 'val']

ts, tl, gn = zip(*train_r)
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax[0].plot(ts, tl, label='train', linewidth=0.9)
if val_r:
    vs, vl = zip(*val_r)
    ax[0].plot(vs, vl, 'o-', label='val', markersize=4)
    best_step = vs[vl.index(min(vl))]
    ax[0].axvline(best_step, color='green', linestyle='--', alpha=0.5,
                  label=f'best val @ step {best_step}')
ax[0].set_ylabel('loss'); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_title('sql-lm v2 fine-tuning')

ax[1].plot(ts, gn, color='orange', linewidth=0.9)
ax[1].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='clip=1.0')
ax[1].set_ylabel('grad norm'); ax[1].set_xlabel('step')
ax[1].legend(); ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/content/ft_v2_curve.png', dpi=150)
plt.show()

if val_r:
    print(f"Best val loss: {min(vl):.4f} at step {best_step}")
print(f"Latest train loss: {tl[-1]:.4f} at step {ts[-1]}")

# ===========================================================================
# Cell 9 — Quick inference test (run after training completes)
# ===========================================================================

import orbax.checkpoint as ocp
import jax, jax.numpy as jnp
from tokenizers import Tokenizer
from scripts.model import SQLTransformer
from scripts.config import *
from scripts.generate import generate
from scripts.finetune import create_ft_optimizer, create_ft_schedule, FT_DROPOUT

# Load best checkpoint (change path if you want a specific step)
CKPT = '/content/drive/MyDrive/sql-lm-data/checkpoints/ft_step_02500'  # update after training

model = SQLTransformer(
    vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS, head_dim=HEAD_DIM, intermediate_dim=INTERMEDIATE_DIM,
    context_length=CONTEXT_LENGTH, rope_base=ROPE_BASE,
    dtype=jnp.bfloat16, dropout_rate=FT_DROPOUT,
)
dummy          = jnp.ones((1, CONTEXT_LENGTH), dtype=jnp.int32)
template_params = model.init(jax.random.PRNGKey(0), dummy)['params']
opt_template   = create_ft_optimizer(create_ft_schedule()).init(template_params)

ckptr = ocp.PyTreeCheckpointer()
restored = ckptr.restore(CKPT, item={
    'params': template_params, 'opt_state': opt_template, 'step': 0})
params = restored['params']

tok = Tokenizer.from_file('/content/tokenizer/tokenizer.json')
print(f"Loaded checkpoint: {CKPT}\n")

tests = [
    ("<schema>CREATE TABLE employees (id INT, name TEXT, salary FLOAT, dept TEXT)</schema>"
     "<question>What is the average salary by department?</question><sql>",
     "avg by group"),

    ("<schema>CREATE TABLE orders (id INT, customer_id INT, total FLOAT)\n"
     "CREATE TABLE customers (id INT, name TEXT, city TEXT)</schema>"
     "<question>List all customer names and their total order amounts</question><sql>",
     "join"),

    ("<schema>CREATE TABLE products (id INT, name TEXT, price FLOAT, category TEXT)</schema>"
     "<question>What is the most expensive product in each category?</question><sql>",
     "subquery"),

    ("<schema>CREATE TABLE students (id INT, name TEXT, grade INT, school TEXT)</schema>"
     "<question>Find students with a grade above 90 sorted by grade descending</question><sql>",
     "filter + sort"),
]

for prompt, label in tests:
    out = generate(params, model, tok, prompt, max_new_tokens=150, temperature=0.0)
    print(f"[{label}]\n  {out}\n")
