from flax import nnx
from soma_models.v1.flax.modules.model import Model, ModelConfig
from soma_models.v1.flax.serde import Serde

_SMALL = dict(
    dropout_rate=0.0,
    embedding_dim=64,
    pwff_hidden_dim=256,
    num_layers=1,
    num_heads=4,
    vocab_size=32,
)


def test_flax_serde():
    original = Model(ModelConfig(**_SMALL), rngs=nnx.Rngs(0))
    serde = Serde(original)
    serialized = serde.serialize()
    deserialized = serde.deserialize(serialized)
    assert original == deserialized
