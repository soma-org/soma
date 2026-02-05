import os
from typing import Union
from jax import Array
from flax import nnx
from dataclasses import dataclass

from soma_models.config import (
    V1_EMBEDDING_DIM,
    V1_NUM_LAYERS,
    V1_PWFF_HIDDEN_DIM,
    V1_NUM_HEADS,
    V1_VOCAB_SIZE,
    V1_MAX_WAVELENGTH,
    V1_SCALE_FACTOR,
)

from soma_models.flax.v1.modules.encoder import Encoder, EncoderConfig
from soma_models.flax.serde import Serde


@dataclass
class ProbeConfig:
    dropout_rate: float
    embedding_dim: int = V1_EMBEDDING_DIM
    pwff_hidden_dim: int = V1_PWFF_HIDDEN_DIM
    num_layers: int = V1_NUM_LAYERS
    num_heads: int = V1_NUM_HEADS
    vocab_size: int = V1_VOCAB_SIZE
    max_wavelength: float = V1_MAX_WAVELENGTH
    scale_factor: float = V1_SCALE_FACTOR


class Probe(nnx.Module):
    def __init__(self, config: ProbeConfig, rngs: nnx.Rngs) -> None:
        self.config = config
        self.encoder = Encoder(
            EncoderConfig(
                dropout_rate=config.dropout_rate,
                embedding_dim=config.embedding_dim,
                pwff_hidden_dim=config.pwff_hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                max_wavelength=config.max_wavelength,
                scale_factor=config.scale_factor,
            ),
            rngs,
        )
        self.final_norm = nnx.LayerNorm(
            config.embedding_dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs
        )
        self.predictor = nnx.Linear(config.embedding_dim, config.vocab_size, rngs=rngs)

    def __call__(
        self,
        input: Array,  # [batch, seq_len, embedding_dim]
        positions: Array,
        attn_mask: Array,
    ) -> Array:
        x = self.encoder(input, positions, attn_mask)
        x = self.final_norm(x)
        return x

    def predictor(
        self,
        embeddings: Array,
    ) -> Array:
        return self.predictor(embeddings)

    def serialize(self) -> bytes:
        return Serde(self).serialize()

    def serialize_to_file(self, filename: Union[str, os.PathLike]) -> None:
        return Serde(self).serialize_to_file(filename)

    def deserialize(self, data: bytes):
        self = Serde(self).deserialize(data)

    def deserialize_from_file(self, filename: Union[str, os.PathLike]):
        self = Serde(self).deserialize_from_file(filename)
