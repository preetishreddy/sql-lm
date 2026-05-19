import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any


def precompute_freqs(head_dim: int, max_seq_len: int,
                     base: int = 10_000) -> tuple:
    theta = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))
    positions = jnp.arange(max_seq_len)
    freqs = jnp.outer(positions, theta)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x, cos, sin):
    # x: [batch, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim // 2]
    # We expand cos/sin to [1, seq_len, 1, head_dim // 2] for robust broadcasting
    cos = jnp.expand_dims(cos, axis=(0, 2))
    sin = jnp.expand_dims(sin, axis=(0, 2))
    
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = jnp.concatenate(
        [x1 * cos - x2 * sin,
         x1 * sin + x2 * cos], axis=-1)
    return rotated


def make_causal_mask(seq_len: int) -> jnp.ndarray:
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    return mask[None, None, :, :]  # [1, 1, T, T]


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        gamma = self.param('scale', nn.initializers.ones, (self.dim,))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * gamma


class SwiGLU(nn.Module):
    hidden_dim: int
    intermediate_dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        gate = nn.silu(nn.Dense(self.intermediate_dim,
                                use_bias=False, dtype=self.dtype)(x))
        up   = nn.Dense(self.intermediate_dim,
                        use_bias=False, dtype=self.dtype)(x)
        return nn.Dense(self.hidden_dim,
                        use_bias=False, dtype=self.dtype)(gate * up)


class MultiHeadAttention(nn.Module):
    hidden_dim: int
    num_heads: int
    head_dim: int
    dtype: Any = jnp.bfloat16

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

        # Cast RoPE frequencies to match activation dtype
        cos = cos.astype(self.dtype)
        sin = sin.astype(self.dtype)

        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        # Revert to a highly controlled manual scaled dot-product attention
        # to guarantee shape compatibility and avoid opaque XLA broadcasting bugs.
        q = q.transpose(0, 2, 1, 3)  # [B, H, T, head_dim]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        # Compute attention scores: [B, H, T, head_dim] @ [B, H, head_dim, T] -> [B, H, T, T]
        attn = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        
        # Ensure mask is [1, 1, T, T] and broadcastable to [B, H, T, T]
        attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)

        # Apply attention to values: [B, H, T, T] @ [B, H, T, head_dim] -> [B, H, T, head_dim]
        out = jnp.einsum('bhij,bhjd->bhid', attn, v)

        # Transpose back to [batch, seq_len, num_heads, head_dim] before reshape
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return nn.Dense(self.hidden_dim, use_bias=False,
                        dtype=self.dtype)(out)


class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    head_dim: int
    intermediate_dim: int
    dtype: Any = jnp.bfloat16
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, cos, sin, mask):
        attn_out = MultiHeadAttention(
            self.hidden_dim, self.num_heads,
            self.head_dim, self.dtype)(
            RMSNorm(self.hidden_dim)(x), cos, sin, mask)
        no_drop = self.dropout_rate == 0.0
        x = x + nn.Dropout(rate=self.dropout_rate, deterministic=no_drop)(attn_out)

        mlp_out = SwiGLU(
            self.hidden_dim, self.intermediate_dim,
            self.dtype)(RMSNorm(self.hidden_dim)(x))
        x = x + nn.Dropout(rate=self.dropout_rate, deterministic=no_drop)(mlp_out)

        return x, None


# Defined at module level so nn.remat wraps the class once, not per forward pass.
RematTransformerBlock = nn.remat(TransformerBlock)


class SQLTransformer(nn.Module):
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_dim: int
    context_length: int
    rope_base: int
    dtype: Any = jnp.bfloat16
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, token_ids, train: bool = False):
        B, T = token_ids.shape

        embed = self.param('embedding',
                           nn.initializers.normal(stddev=0.02),
                           (self.vocab_size, self.hidden_dim))
        x = embed[token_ids].astype(self.dtype)

        cos, sin = precompute_freqs(self.head_dim, T, self.rope_base)
        mask = make_causal_mask(T)

        # Broadcast to [num_layers, ...] for nn.scan
        cos_scanned = jnp.broadcast_to(cos[None], (self.num_layers, *cos.shape))
        sin_scanned = jnp.broadcast_to(sin[None], (self.num_layers, *sin.shape))
        mask_scanned = jnp.broadcast_to(mask[None], (self.num_layers, *mask.shape))

        # Set dropout_rate=0 during inference so nn.Dropout is a no-op without
        # needing to pass `deterministic` through scan (scan ignores kwargs).
        effective_dropout = self.dropout_rate if train else 0.0

        # 16 Transformer blocks via nn.scan with gradient checkpointing
        ScanBlock = nn.scan(
            RematTransformerBlock,
            variable_axes={'params': 0},
            split_rngs={'params': True, 'dropout': True},
            length=self.num_layers,
        )(self.hidden_dim, self.num_heads, self.head_dim,
          self.intermediate_dim, self.dtype, effective_dropout)

        x, _ = ScanBlock(x, cos_scanned, sin_scanned, mask_scanned)

        x = RMSNorm(self.hidden_dim)(x)

        logits = x @ embed.T

        return logits
