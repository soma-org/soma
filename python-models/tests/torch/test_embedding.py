import torch
import torch.nn as nn
from arrgen import normal_array
from safetensors.numpy import save
from soma_models.torch.serde import load_safetensor_into


class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


def test_embedding():
    seed = 42
    batch_size = 1
    embedding_dim = 2
    num_embeddings = 4

    module = EmbeddingModule(num_embeddings, embedding_dim)
    generated_tensors = {
        "embedding.weight": normal_array(
            seed, [num_embeddings, embedding_dim], mean=0.0, std_dev=1.0
        ),
    }
    serialized_tensors = save(generated_tensors)
    load_safetensor_into(module, serialized_tensors)
    module.eval()

    inputs = torch.arange(0, num_embeddings).unsqueeze(0).repeat(batch_size, 1)
    outputs = module(inputs)
    expected = torch.tensor(
        [[
            [0.069427915, 0.13293812],
            [0.26257637, -0.22530088],
            [-0.66422486, -0.2153902],
            [0.19392312, 1.4764173],
        ]],
    )
    assert torch.allclose(outputs, expected), f"Arrays are not close enough! {outputs} vs {expected}"
