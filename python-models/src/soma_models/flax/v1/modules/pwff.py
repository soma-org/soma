from flax import nnx
from dataclasses import dataclass

from soma_models.config import (
    V1_EMBEDDING_DIM,
    V1_PWFF_HIDDEN_DIM,
)


@dataclass
class PositionWiseFeedForwardConfig:
    dropout_rate: float
    embedding_dim: int = V1_EMBEDDING_DIM
    pwff_hidden_dim: int = V1_PWFF_HIDDEN_DIM


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
