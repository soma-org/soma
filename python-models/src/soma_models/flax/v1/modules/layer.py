from jax import Array
from flax import nnx

from soma_models.v1.configs import LayerConfig, PositionWiseFeedForwardConfig
from soma_models.flax.v1.modules.pwff import PositionWiseFeedForward
from soma_models.flax.v1.modules.attention import MultiHeadAttention


class Layer(nnx.Module):
    def __init__(self, config: LayerConfig, rngs: nnx.Rngs):
        self.norm_1 = nnx.LayerNorm(
            num_features=config.embedding_dim,
            rngs=rngs,
            epsilon=1e-5,
            use_fast_variance=False,
        )
        self.attention = MultiHeadAttention(
            num_heads=config.num_heads,
            num_features=config.embedding_dim,
            dropout_rate=config.dropout_rate,
            max_wavelength=config.max_wavelength,
            scale_factor=config.scale_factor,
            rngs=rngs,
        )
        self.norm_2 = nnx.LayerNorm(
            num_features=config.embedding_dim,
            rngs=rngs,
            use_fast_variance=False,
            epsilon=1e-5,
        )
        self.pwff = PositionWiseFeedForward(
            PositionWiseFeedForwardConfig(
                dropout_rate=config.dropout_rate,
                embedding_dim=config.embedding_dim,
                pwff_hidden_dim=config.pwff_hidden_dim,
            ),
            rngs,
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, context: Array, positions: Array, attn_mask: Array):
        x = context
        residual_path = self.norm_1(x)
        residual_path = self.attention(
            residual_path, positions=positions, mask=attn_mask
        )
        residual_path = self.dropout(residual_path)
        x = x + residual_path
        residual_path = self.norm_2(x)
        residual_path = self.pwff(residual_path)
        residual_path = self.dropout(residual_path)
        x = x + residual_path
        return x
