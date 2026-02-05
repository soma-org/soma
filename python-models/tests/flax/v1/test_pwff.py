from jax import numpy as jnp
from flax import nnx
from arrgen import (
    uniform_array,
    normal_array,
    constant_array,
)
from safetensors.numpy import save
from soma_models.flax.serde import Serde
from soma_models.flax.v1.modules.pwff import (
    PositionWiseFeedForward,
    PositionWiseFeedForwardConfig,
)


def test_v1_pwff_ones():
    embedding_dim = 4
    hidden_dim = 2
    seed = 42
    serde = Serde(
        PositionWiseFeedForward(
            PositionWiseFeedForwardConfig(
                dropout_rate=0.0,
                embedding_dim=embedding_dim,
                pwff_hidden_dim=hidden_dim,
            ),
            rngs=nnx.Rngs(0),
        )
    )
    generated_tensors = {
        "linear_inner.weight": normal_array(
            seed + 1, [embedding_dim, hidden_dim], mean=0.0, std_dev=1.0
        ),
        "linear_inner.bias": normal_array(
            seed + 2, [hidden_dim], mean=0.0, std_dev=1.0
        ),
        "linear_outer.weight": normal_array(
            seed + 3, [hidden_dim, embedding_dim], mean=0.0, std_dev=1.0
        ),
        "linear_outer.bias": normal_array(
            seed + 4, [embedding_dim], mean=0.0, std_dev=1.0
        ),
    }
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()

    expected = jnp.array([0.31442693, 0.70205802, -2.13397980, -1.71679294])
    input = jnp.array(constant_array([embedding_dim], 1.0))
    output = module(input)

    assert jnp.allclose(output, expected), "Arrays are not close enough!"
