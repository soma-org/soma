from jax import Array
from flax import nnx

from soma_models.v1.configs import ModelConfig, EncoderConfig
from soma_models.flax.v1.modules.encoder import Encoder
from soma_models.flax.serde import Serializable


class Model(Serializable, nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs) -> None:
        self.config = config
        self.embedding = nnx.Embed(config.vocab_size, config.embedding_dim, rngs=rngs)
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
        input: Array,
        positions: Array,
        attn_mask: Array,
    ) -> Array:
        x = self.embedding(input)
        x = self.encoder(x, positions, attn_mask)
        x = self.final_norm(x)
        return x

    def predict(
        self,
        embeddings: Array,
    ) -> Array:
        return self.predictor(embeddings)
