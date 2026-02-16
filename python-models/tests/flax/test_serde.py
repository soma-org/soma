from flax import nnx
from soma_models.flax.v1.model import Model, ModelConfig
from soma_models.flax.serde import Serde


def test_flax_serde():
    original = Model(ModelConfig(dropout_rate=0.0), rngs=nnx.Rngs(0))
    serde = Serde(original)
    serialized = serde.serialize()
    print(len(serialized))
    deserialized = serde.deserialize(serialized)
    assert original == deserialized
