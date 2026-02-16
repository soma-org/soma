from flax import nnx

from soma_models.v1.configs import PositionWiseFeedForwardConfig


class PositionWiseFeedForward(nnx.Module):
    def __init__(self, config: PositionWiseFeedForwardConfig, rngs: nnx.Rngs):
        self.linear_inner = nnx.Linear(
            in_features=config.embedding_dim,
            out_features=config.pwff_hidden_dim,
            rngs=rngs,
        )
        self.linear_outer = nnx.Linear(
            in_features=config.pwff_hidden_dim,
            out_features=config.embedding_dim,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, x):
        x = self.linear_inner(x)
        x = nnx.gelu(x, approximate=False)
        x = self.dropout(x)
        output = self.linear_outer(x)
        return output
