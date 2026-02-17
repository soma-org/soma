import torch
from soma_models.torch.v1.model import Model, ModelConfig
from soma_models.torch.serde import Serde


def test_torch_serde():
    torch.manual_seed(0)
    original = Model(ModelConfig(dropout_rate=0.0))
    serde = Serde(original)
    serialized = serde.serialize()

    torch.manual_seed(0)
    restored = Model(ModelConfig(dropout_rate=0.0))
    Serde(restored).deserialize(serialized)

    for (k1, p1), (k2, p2) in zip(
        original.state_dict().items(), restored.state_dict().items()
    ):
        assert k1 == k2, f"Key mismatch: {k1} vs {k2}"
        assert torch.equal(p1, p2), f"Tensor mismatch at {k1}"


def test_torch_serializable_roundtrip():
    torch.manual_seed(0)
    original = Model(ModelConfig(dropout_rate=0.0))
    data = original.save_bytes()

    torch.manual_seed(0)
    restored = Model.load_bytes(data, ModelConfig(dropout_rate=0.0))

    for (k1, p1), (k2, p2) in zip(
        original.state_dict().items(), restored.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(p1, p2), f"Tensor mismatch at {k1}"


def test_torch_serde_file(tmp_path):
    torch.manual_seed(0)
    original = Model(ModelConfig(dropout_rate=0.0))
    path = tmp_path / "model.safetensors"
    original.save(path)

    torch.manual_seed(0)
    restored = Model.load(path, ModelConfig(dropout_rate=0.0))

    for (k1, p1), (k2, p2) in zip(
        original.state_dict().items(), restored.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(p1, p2), f"Tensor mismatch at {k1}"
