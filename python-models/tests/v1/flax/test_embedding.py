from jax import numpy as jnp
from flax import nnx
from arrgen import normal_array
from safetensors.numpy import save
from soma_models.v1.flax.serde import Serde


class EmbedModule(nnx.Module):
    def __init__(self, num_embeddings, embedding_dim, rngs: nnx.Rngs):
        self.embedding = nnx.Embed(num_embeddings, embedding_dim, rngs=rngs)

    def __call__(self, x):
        return self.embedding(x)


def test_v1_embedding():
    seed = 42
    batch_size = 1
    embedding_dim = 2
    num_embeddings = 4

    serde = Serde(EmbedModule(num_embeddings, embedding_dim, rngs=nnx.Rngs(0)))
    generated_tensors = {
        "embedding.weight": normal_array(
            seed, [num_embeddings, embedding_dim], mean=0.0, std_dev=1.0
        ),
    }
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()

    inputs = jnp.arange(0, num_embeddings).reshape(batch_size, -1)
    outputs = module(inputs)

    expected = jnp.array(
        [[
            [0.069427915, 0.13293812],
            [0.26257637, -0.22530088],
            [-0.66422486, -0.2153902],
            [0.19392312, 1.4764173],
        ]],
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"
