import torch
from arrgen import (
    uniform_array,
    normal_array,
    constant_array,
)
from safetensors.numpy import save
from soma_models.v1.torch.serde import load_safetensor_into


class LayerNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(4, eps=1e-5)

    def forward(self, x):
        return self.layer_norm(x)


def test_layer_norm_ones():
    seed = 42
    module = LayerNormModule()
    generated_tensors = {
        "layer_norm.gamma": normal_array(seed, [4], mean=0.0, std_dev=1.0),
        "layer_norm.beta": normal_array(seed + 1, [4], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    load_safetensor_into(module, serialized_tensors)
    module.eval()

    expected = torch.tensor([0.26803425, -0.30034754, -0.18579677, -0.37248048])
    input = torch.tensor(constant_array([4], 1.0))
    output = module(input)

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"


def test_layer_norm_uniform():
    seed = 44
    module = LayerNormModule()
    generated_tensors = {
        "layer_norm.gamma": normal_array(seed, [4], mean=0.0, std_dev=1.0),
        "layer_norm.beta": normal_array(seed + 1, [4], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    load_safetensor_into(module, serialized_tensors)
    module.eval()

    expected = torch.tensor([-0.74536324, -2.98460746, -0.31756663, -0.38157958])
    input = torch.tensor(uniform_array(seed + 2, [4], 0.0, 1.0))
    output = module(input)

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"
