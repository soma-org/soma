from jax import Array
from flax import nnx
from typing import List
from soma_probes.config import (
    V1_NUM_LAYERS,
)
from soma_probes.flax.v1.modules.layer import Layer


class Encoder(nnx.Module):
    def __init__(self, dropout_rate: float, rngs: nnx.Rngs):
        self.layers: List[Layer] = [
            Layer(dropout_rate, rngs=rngs) for _ in range(V1_NUM_LAYERS)
        ]

    def __call__(
        self,
        representations: Array,
        positions: Array,
    ):
        x = representations
        for layer in self.layers:
            x = layer(x, positions)
        return x
