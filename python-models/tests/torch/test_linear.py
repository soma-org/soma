import torch
from arrgen import (
    uniform_array,
    normal_array,
    constant_array,
)
from safetensors.numpy import save
from soma_models.torch.serde import load_safetensor_into


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)

    def forward(self, x):
        return self.linear(x)


def test_linear_ones():
    seed = 42
    input_dim = 2
    output_dim = 4
    module = LinearModule()
    generated_tensors = {
        "linear.weight": normal_array(
            seed + 1, [input_dim, output_dim], mean=0.0, std_dev=1.0
        ),
        "linear.bias": normal_array(seed, [output_dim], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    load_safetensor_into(module, serialized_tensors)
    module.eval()

    expected = torch.tensor([-1.77364016, 1.29809809, -0.31307063, -1.68842816])
    input = torch.tensor(constant_array([input_dim], 1.0))
    output = module(input)

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"


def test_linear_uniform():
    seed = 44
    input_dim = 2
    output_dim = 4
    module = LinearModule()
    generated_tensors = {
        "linear.weight": normal_array(
            seed + 1, [input_dim, output_dim], mean=0.0, std_dev=1.0
        ),
        "linear.bias": normal_array(seed, [output_dim], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    load_safetensor_into(module, serialized_tensors)
    module.eval()

    expected = torch.tensor([-0.53813028, -1.69855022, 0.92013592, 0.92915082])
    input = torch.tensor(uniform_array(seed + 2, [input_dim], 0.0, 1.0))
    output = module(input)

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"
