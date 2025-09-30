import os
from typing import Union
from jax import Array
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass

from soma_probes.config import (
    V1_EMBEDDING_DIM,
    V1_VOCAB_SIZE,
    V1_MAX_SEQ_LEN,
    V1_NUM_LAYERS,
    V1_PWFF_HIDDEN_DIM,
    V1_NUM_HEADS,
    V1_MAX_WAVELENGTH,
)
from soma_probes.flax.v1.modules.encoder import Encoder, EncoderConfig
from soma_probes.flax.serde import Serde


@dataclass
class ProbeConfig:
    dropout_rate: float
    embedding_dim: int = V1_EMBEDDING_DIM
    pwff_hidden_dim: int = V1_PWFF_HIDDEN_DIM
    num_layers: int = V1_NUM_LAYERS
    num_heads: int = V1_NUM_HEADS
    vocab_size: int = V1_VOCAB_SIZE
    max_wavelength: int = V1_MAX_WAVELENGTH
    max_seq_len: int = V1_MAX_SEQ_LEN


class Probe(nnx.Module):
    def __init__(self, config: ProbeConfig, rngs: nnx.Rngs) -> None:
        self.config = config
        self.mask_token = nnx.Param(
            # using the same init as nnx.Embed
            nnx.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)(
                rngs.params(), (1, 1, config.embedding_dim)
            )
        )
        self.encoder = Encoder(EncoderConfig(dropout_rate=config.dropout_rate), rngs)
        self.final_norm = nnx.LayerNorm(config.embedding_dim, rngs=rngs)
        self.predictor = nnx.Linear(config.embedding_dim, config.vocab_size, rngs=rngs)

    def __call__(
        self,
        representations: Array,  # [batch, seq_len, embedding_dim]
        positions: Array,  # [batch, seq_len]
    ) -> Array:
        rep_shape = representations.shape
        pos_shape = positions.shape
        if not (
            len(rep_shape) == 3
            and len(pos_shape) == 2
            and rep_shape[:2] == pos_shape[:2]  # Batch and seq_len match
        ):
            raise ValueError(
                f"Shape mismatch: "
                f"context_embeddings.shape={rep_shape}, "
                f"context_byte_indices.shape={pos_shape}, "
            )

        if rep_shape[1] + 1 <= self.config.max_seq_len:
            raise ValueError(
                "seq len plus mask token is greater than the allowed sequence length"
            )

        x = jnp.concatenate([self.mask_token, representations], axis=1)
        relative_positions = jnp.concatenate([0, positions], axis=1)
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
