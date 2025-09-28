from jax import Array
from flax import nnx
from soma_probes.config import (
    V1_EMBEDDING_DIM,
    V1_NUM_HEADS,
    V1_MAX_WAVELENGTH,
)
from soma_probes.flax.v1.modules.pwff import PositionWiseFeedForward
from soma_probes.flax.v1.modules.attention import MultiHeadAttention


class Layer(nnx.Module):
    def __init__(self, dropout_rate: float, rngs: nnx.Rngs):
        self.norm_1 = nnx.LayerNorm(
            num_features=V1_EMBEDDING_DIM,
            rngs=rngs,
            epsilon=1e-5,
            use_fast_variance=False,
        )
        self.attention = MultiHeadAttention(
            num_heads=V1_NUM_HEADS,
            in_features=V1_EMBEDDING_DIM,
            dropout_rate=dropout_rate,
            decode=False,
            max_wavelength=V1_MAX_WAVELENGTH,
            rngs=rngs,
        )
        self.norm_2 = nnx.LayerNorm(
            num_features=V1_EMBEDDING_DIM,
            rngs=rngs,
            use_fast_variance=False,
            epsilon=1e-5,
        )
        self.pwff = PositionWiseFeedForward(dropout_rate, rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(
        self,
        representations: Array,
        positions: Array,
    ):
        x = representations
        residual_path = self.norm_1(x)
        residual_path = self.attention(residual_path, positions)
        residual_path = self.dropout(residual_path)
        x = x + residual_path
        residual_path = self.norm_2(x)
        residual_path = self.pwff(residual_path)
        x = x + residual_path
        return x
