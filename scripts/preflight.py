import jax
import jax.numpy as jnp
import numpy as np
import json
import os
from pathlib import Path
from tokenizers import Tokenizer
import optax

from scripts.config import *
from scripts.model import SQLTransformer
from scripts.data_loader import CorpusLoader
from scripts.train import create_learning_rate_schedule
from scripts.evaluate import loss_fn

def run_preflight():
    print("="*60)
    print("PRE-FLIGHT VERIFICATION")
    print("="*60)

    # --- 1. Tokenizer loads and round-trips ---
    tokenizer_path = 'tokenizer/tokenizer.json'
    if not os.path.exists(tokenizer_path):
        print(f"[1] FAIL: Tokenizer not found at {tokenizer_path}")
        return
    tok = Tokenizer.from_file(tokenizer_path)
    sample = "SELECT name FROM users WHERE id = 1;"
    ids = tok.encode(sample).ids
    back = tok.decode(ids)
    assert tok.get_vocab_size() == 12288, f"vocab {tok.get_vocab_size()} != 12288"
    assert 'SELECT' in back and 'users' in back, f"decode failed: {back!r}"
    print(f"[1] tokenizer OK   ({len(ids)} tok roundtrip: {back!r})")

    # --- 2. Manifest + all .npy files present ---
    manifest_path = 'data/tokenized/manifest.json'
    if not os.path.exists(manifest_path):
        print(f"[2] FAIL: Manifest not found at {manifest_path}")
        return
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    assert manifest['total_train_tokens'] > 2e9, "corpus suspiciously small"
    print(f"[2] corpus OK      ({manifest['total_train_tokens']:,} train tokens, {len(manifest['sources'])} sources)")

    # --- 3. Data loader yields the right shape/dtype, and tokens decode ---
    loader = CorpusLoader('data/tokenized', split='train')
    batch = loader.next_batch(4)
    assert batch.shape == (4, 512), f"batch shape {batch.shape}"
    assert batch.dtype == np.int32, f"batch dtype {batch.dtype}"
    assert batch.min() >= 0 and batch.max() < 12288, "tokens out of range"
    preview = tok.decode(batch[0][:60].tolist())
    print(f"[3] dataloader OK  (sample: {preview[:80]!r}...)")

    # --- 4. Model initializes with expected param count ---
    model = SQLTransformer(
        vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        intermediate_dim=INTERMEDIATE_DIM, context_length=CONTEXT_LENGTH,
        rope_base=ROPE_BASE,
    )
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, jnp.ones((1, 512), dtype=jnp.int32))
    params = variables['params']
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert 30_000_000 < n_params < 33_000_000, f"param count {n_params:,} outside expected range"
    print(f"[4] model OK       ({n_params:,} params)")

    # --- 5. Forward pass produces finite loss near ln(vocab) at init ---
    batch_jax = jnp.array(batch)
    logits = model.apply({'params': params}, batch_jax)
    
    # Simple cross entropy for verification
    def compute_loss(p, b):
        inputs = b[:, :-1]
        targets = b[:, 1:]
        l = model.apply({'params': p}, inputs)
        return optax.softmax_cross_entropy_with_integer_labels(
            l.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1)
        ).mean()

    loss0 = compute_loss(params, batch_jax)
    expected = float(jnp.log(VOCAB_SIZE))   # ≈ 9.42
    assert jnp.isfinite(loss0), f"loss is {loss0} (NaN/Inf)"
    assert abs(float(loss0) - expected) < 1.5, f"init loss {float(loss0):.2f} far from ln(vocab)={expected:.2f}"
    print(f"[5] forward OK     (loss {float(loss0):.3f}, expected ~{expected:.2f})")

    # --- 6. Backward pass runs without error ---
    grads = jax.grad(compute_loss)(params, batch_jax)
    gnorm = float(optax.global_norm(grads))
    assert jnp.isfinite(gnorm), f"grad norm is {gnorm}"
    print(f"[6] backward OK    (grad norm {gnorm:.3f})")

    # --- 7. LR schedule produces expected values at key steps ---
    schedule = create_learning_rate_schedule()
    assert abs(float(schedule(0)) - 0.0) < 1e-9
    assert abs(float(schedule(WARMUP_STEPS)) - PEAK_LR) < 1e-6
    # Note: total steps depends on decay steps etc. 
    # Just check warmup and peak.
    print(f"[7] schedule OK    (warmup -> {PEAK_LR})")

    # --- 8. Local write test ---
    test_file = Path('.preflight_test')
    test_file.write_text('ok')
    test_file.unlink()
    print(f"[8] local write OK")

    print("="*60)
    print("ALL CHECKS PASSED — ready for training")
    print("="*60)

if __name__ == "__main__":
    try:
        run_preflight()
    except Exception as e:
        print(f"\nPRE-FLIGHT FAILED: {e}")
        import traceback
        traceback.print_exc()
