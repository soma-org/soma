from flax import nnx

from soma_probes.config import (
    V1_EMBEDDING_DIM,
    V1_PWFF_HIDDEN_DIM,
)


class PositionWiseFeedForward(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear_1 = nnx.Linear(
            in_features=V1_EMBEDDING_DIM,
            out_features=V1_PWFF_HIDDEN_DIM,
            rngs=rngs,
        )
        self.linear_2 = nnx.Linear(
            in_features=V1_EMBEDDING_DIM,
            out_features=V1_PWFF_HIDDEN_DIM,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.linear_1(x)
        x = nnx.gelu(x, approximate=False)
        output = self.linear_2(x)
        return output
