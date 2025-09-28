import os
from typing import Union
from jax import Array
import jax.numpy as jnp
from flax import nnx


from soma_probes.config import (
    V1_EMBEDDING_DIM,
    V1_VOCAB_SIZE,
    V1_MAX_SEQ_LEN,
)
from soma_probes.flax.v1.modules.encoder import Encoder
from soma_probes.flax.serde import Serde


class Probe(nnx.Module):
    def __init__(self, dropout_rate: float, rngs: nnx.Rngs) -> None:
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
    ) -> Array:
        tbi_shape = target_byte_index.shape
        emb_shape = context_embeddings.shape
        idx_shape = context_byte_indices.shape
        if not (
            len(tbi_shape) == 1
            and len(emb_shape) == 3
            and len(idx_shape) == 2
            and tbi_shape[:1] == emb_shape[:1] == idx_shape[:1]  # Batch size matches
            and emb_shape[:2] == idx_shape[:2]  # Batch and seq_len match
        ):
            raise ValueError(
                f"Shape mismatch: "
                f"target_byte_index.shape={tbi_shape}, "
                f"context_embeddings.shape={emb_shape}, "
                f"context_byte_indices.shape={idx_shape}, "
            )

        if emb_shape[1] + 1 <= V1_MAX_SEQ_LEN:
            raise ValueError(
                "seq len plus mask token is greater than the allowed sequence length"
            )

        relative_positions = jnp.subtract(context_byte_indices, target_byte_index)
        x = jnp.concatenate([self.mask_token, context_embeddings], axis=1)
        relative_positions = jnp.concatenate([0, relative_positions], axis=1)
        x = self.encoder(x, relative_positions)
        x = self.final_norm(x)
        mask_token = x[:, 0]  # Shape: [batch, embedding_dim]
        return self.predictor(mask_token)

    def serialize(self) -> bytes:
        return Serde(self).serialize()

    def serialize_to_file(self, filename: Union[str, os.PathLike]) -> None:
        return Serde(self).serialize_to_file(filename)

    def deserialize(self, data: bytes):
        self = Serde(self).deserialize(data)

    def deserialize_from_file(self, filename: Union[str, os.PathLike]):
        self = Serde(self).deserialize_from_file(filename)
