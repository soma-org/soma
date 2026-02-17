from jax import Array
from flax import nnx

from soma_models.v1.configs import EncoderConfig, LayerConfig
from soma_models.v1.flax.modules.layer import Layer


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
                        scale_factor=config.scale_factor,
                    ),
                    rngs=rngs,
                )
                for _ in range(config.num_layers)
            ]
        )

    def __call__(
        self,
        input: Array,
        positions: Array,
    ):
        x = input
        for layer in self.layers:
            x = layer(x, positions)
        return x
