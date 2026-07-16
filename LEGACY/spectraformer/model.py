import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState


class CustomTrainState(TrainState):
    """Extended TrainState with epoch tracking"""
    epoch: jax.Array


class LinearProjection(nn.Module):
    """Linear embedding projection"""

    embedding_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embedding_dim, use_bias=True)(x)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Continuous sinusoidal positional encoding for physical coordinates."""
    
    embedding_dim: int
    min_freq: float = 1.0
    max_freq: float = 1000.0

    @nn.compact
    def __call__(self, x):
        # x shape: expected [..., 1]
        half_dim = self.embedding_dim // 2
        
        freqs = jnp.geomspace(self.min_freq, self.max_freq, half_dim)
        angles = x * freqs
        
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        
        if self.embedding_dim % 2 != 0:
            pad_dims = [(0, 0)] * (emb.ndim - 1) + [(0, 1)]
            emb = jnp.pad(emb, pad_dims)
            
        return emb


def test_LinearProjection():
    """Test LinearProjection"""
    x = np.random.randn(1, 2, 3, 100)
    model = LinearProjection(128)
    variables = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(variables, x)
    chex.assert_shape(y, (1, 2, 3, 128))
    chex.assert_shape(variables["params"]["Dense_0"]["kernel"], (100, 128))


class FFBlock(nn.Module):
    """Feed-forward block for transformer model"""

    embedding_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(4 * self.embedding_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.embedding_dim)(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer"""

    embedding_dim: int
    num_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, attn_mask, training: bool):
        # Multi-head attention
        x_norm = nn.LayerNorm()(x)
        x_norm = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embedding_dim,
            deterministic=not training,
            dropout_rate=self.dropout_rate,
        )(x_norm, mask=attn_mask)
        x = x + nn.Dense(self.embedding_dim)(x_norm)  # Residual connection
        x_norm = nn.LayerNorm()(x)
        x_norm = FFBlock(self.embedding_dim, self.dropout_rate)(
            x_norm, training=training
        )
        x = x + nn.Dense(self.embedding_dim)(x_norm)
        return x


class SpectraFormer(nn.Module):
    embedding_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, counts, wave_number, wave_number_mask, training: bool = False):
        # wave_number_mask is a 1D boolean array of shape num_wave_numbers
        emb_counts = LinearProjection(self.embedding_dim)(
            counts
        )  # [batch_size, num_wave_numbers, 1]
        emb_wave_number = SinusoidalPositionalEncoding(self.embedding_dim)(
            wave_number
        )  # [num_wave_numbers, embedding_dim]
        x = (
            emb_counts + emb_wave_number
        )  # [batch_size, num_wave_numbers, embedding_dim]

        # Asymmetric attention mask: we only attend to the unmasked keys/values
        kv_mask = wave_number_mask
        query_mask = jnp.ones_like(kv_mask, dtype=bool)
        attn_mask = nn.make_attention_mask(
            query_mask, kv_mask, extra_batch_dims=x.ndim - 2
        )  # [1, num_wave_numbers, num_wave_numbers]
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                self.embedding_dim, self.num_heads, self.dropout_rate
            )(x, attn_mask, training=training)
        x = nn.LayerNorm()(x)
        x = nn.Dense(1)(x)
        x = jnp.exp(x)
        return x

if __name__ == "__main__":
    test_LinearProjection()