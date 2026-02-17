import torch
from soma_models.v1.torch.modules.model import Model, ModelConfig
from soma_models.v1.torch.serde import Serde

_SMALL = dict(
    dropout_rate=0.0,
    embedding_dim=64,
    pwff_hidden_dim=256,
    num_layers=1,
    num_heads=4,
    vocab_size=32,
)


def test_torch_serde():
    torch.manual_seed(0)
    original = Model(ModelConfig(**_SMALL))
    serde = Serde(original)
    serialized = serde.serialize()

    torch.manual_seed(0)
    restored = Model(ModelConfig(**_SMALL))
    Serde(restored).deserialize(serialized)

    for (k1, p1), (k2, p2) in zip(
        original.state_dict().items(), restored.state_dict().items()
    ):
        assert k1 == k2, f"Key mismatch: {k1} vs {k2}"
        assert torch.equal(p1, p2), f"Tensor mismatch at {k1}"


def test_torch_serializable_roundtrip():
    torch.manual_seed(0)
    original = Model(ModelConfig(**_SMALL))
    data = original.save_bytes()

    torch.manual_seed(0)
    restored = Model.load_bytes(data, ModelConfig(**_SMALL))

    for (k1, p1), (k2, p2) in zip(
        original.state_dict().items(), restored.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(p1, p2), f"Tensor mismatch at {k1}"


def test_torch_serde_file(tmp_path):
    torch.manual_seed(0)
    original = Model(ModelConfig(**_SMALL))
    path = tmp_path / "model.safetensors"
    original.save(path)

    torch.manual_seed(0)
    restored = Model.load(path, ModelConfig(**_SMALL))

    for (k1, p1), (k2, p2) in zip(
        original.state_dict().items(), restored.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(p1, p2), f"Tensor mismatch at {k1}"
