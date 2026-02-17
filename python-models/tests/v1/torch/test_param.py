import torch
from arrgen import normal_array
from safetensors.numpy import save
from safetensors.torch import load


class ParamModule1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(4))

    def forward(self):
        return self.param


class ParamModule3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1, 1, 4))

    def forward(self):
        return self.param


def test_1d_param():
    seed = 42
    embedding_dim = 4
    module = ParamModule1D()
    generated_tensors = {
        "param": normal_array(seed, [embedding_dim], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    tensors = load(serialized_tensors)
    module.load_state_dict(tensors)
    module.eval()

    expected = torch.tensor(
        [0.06942791, 0.13293812, 0.26257637, -0.22530088],
    )
    output = module()

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"


def test_3d_param():
    seed = 42
    embedding_dim = 4
    module = ParamModule3D()
    generated_tensors = {
        "param": normal_array(seed, [1, 1, embedding_dim], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    tensors = load(serialized_tensors)
    module.load_state_dict(tensors)
    module.eval()

    expected = torch.tensor(
        [[[0.06942791, 0.13293812, 0.26257637, -0.22530088]]],
    )
    output = module()

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"
