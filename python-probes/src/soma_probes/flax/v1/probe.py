from jax import Array
import jax.numpy as jnp
from flax import nnx


from soma_probes.config import (
    V1_EMBEDDING_DIM,
    V1_VOCAB_SIZE,
)
from soma_probes.flax.v1.modules.encoder import Encoder


class Probe(nnx.Module):
    def __init__(self, dropout_rate: float, rngs: nnx.Rngs):
        self.mask_token = nnx.Param(
            # using the same init as nnx.Embed
            nnx.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)(
                rngs.params(), (1, 1, V1_EMBEDDING_DIM)
            )
        )
        self.encoder = Encoder(dropout_rate, rngs)
        self.final_norm = nnx.LayerNorm(V1_EMBEDDING_DIM, rngs=rngs)
        self.predictor = nnx.Linear(V1_EMBEDDING_DIM, V1_VOCAB_SIZE, rngs=rngs)

    def __call__(
        self,
        target_byte_index: Array,  # [batch]
        context_embeddings: Array,  # [batch, seq_len, embedding_dim]
        context_byte_indices: Array,  # [batch, seq_len]
    ):
        relative_positions = jnp.subtract(context_byte_indices, target_byte_index)
        x = jnp.concatenate([self.mask_token, context_embeddings], axis=1)
        relative_positions = jnp.concatenate([0, relative_positions], axis=1)

        x = self.encoder(x, relative_positions)
        x = self.final_norm(x)
        mask_token = x[:, 0]  # Shape: [batch, embedding_dim]
        return self.predictor(mask_token)
