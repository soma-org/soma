import os
from typing import Union
from jax import Array
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass

from soma_probes.config import (
    V1_EMBEDDING_DIM,
    V1_NUM_LAYERS,
    V1_PWFF_HIDDEN_DIM,
    V1_NUM_HEADS,
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
            ),
            rngs,
        )

    def __call__(
        self,
        input: Array,  # [batch, seq_len, embedding_dim]
    ) -> Array:
        return self.encoder(input)

    def serialize(self) -> bytes:
        return Serde(self).serialize()

    def serialize_to_file(self, filename: Union[str, os.PathLike]) -> None:
        return Serde(self).serialize_to_file(filename)

    def deserialize(self, data: bytes):
        self = Serde(self).deserialize(data)

    def deserialize_from_file(self, filename: Union[str, os.PathLike]):
        self = Serde(self).deserialize_from_file(filename)
