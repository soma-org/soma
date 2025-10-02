from jax import Array
from flax import nnx
from soma_probes.config import (
    V1_NUM_LAYERS,
    V1_EMBEDDING_DIM,
    V1_PWFF_HIDDEN_DIM,
    V1_NUM_HEADS,
    V1_MAX_WAVELENGTH,
)
from soma_probes.flax.v1.modules.layer import Layer, LayerConfig
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    dropout_rate: float
    embedding_dim: int = V1_EMBEDDING_DIM
    pwff_hidden_dim: int = V1_PWFF_HIDDEN_DIM
    num_layers: int = V1_NUM_LAYERS
    num_heads: int = V1_NUM_HEADS
    max_wavelength: int = V1_MAX_WAVELENGTH


class Encoder(nnx.Module):
    def __init__(self, config: EncoderConfig, rngs: nnx.Rngs):
        self.layers = nnx.List(
            [
                Layer(
                    LayerConfig(
                        dropout_rate=config.dropout_rate,
                        embedding_dim=config.embedding_dim,
                        pwff_hidden_dim=config.pwff_hidden_dim,
                        num_heads=config.num_heads,
                        max_wavelength=config.max_wavelength,
                    ),
                    rngs=rngs,
                )
                for _ in range(config.num_layers)
            ]
        )

    def __call__(
        self,
        representations: Array,
        positions: Array,
    ):
        x = representations
        for layer in self.layers:
            x = layer(x, positions)
        return x
