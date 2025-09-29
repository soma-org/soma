from flax import nnx

from soma_probes.config import (
    V1_EMBEDDING_DIM,
    V1_PWFF_HIDDEN_DIM,
)


class PositionWiseFeedForward(nnx.Module):
    def __init__(self, dropout_rate: float, rngs: nnx.Rngs):
        self.linear_inner = nnx.Linear(
            in_features=V1_EMBEDDING_DIM,
            out_features=V1_PWFF_HIDDEN_DIM,
            rngs=rngs,
        )
        self.linear_outer = nnx.Linear(
            in_features=V1_EMBEDDING_DIM,
            out_features=V1_PWFF_HIDDEN_DIM,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x):
        x = self.linear_inner(x)
        x = nnx.gelu(x, approximate=False)
        x = self.dropout(x)
        output = self.linear_outer(x)
        return output
