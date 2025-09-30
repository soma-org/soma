from arrgen import (
    uniform_array,
    normal_array,
    constant_array,
)
from safetensors.numpy import save

SEED = 42
SHAPE = [2, 2]


def test_generate():
    tensors = {
        "uniform_tensor": uniform_array(SEED, SHAPE, min=0.0, max=1.0),
        "normal_tensor": normal_array(SEED, SHAPE, mean=0.0, std_dev=1.0),
        "constant_tensor": constant_array(SHAPE, value=0.0),
    }

    for name, arr in tensors.items():
        print(f"{name} shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"{name} values:\n{arr}")

    st = save(tensors)
    print(list(st))
