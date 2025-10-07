from flax import nnx
from soma_probes.flax.v1.probe import Probe, ProbeConfig
from soma_probes.flax.serde import Serde


def test_flax_serde():
    original = Probe(ProbeConfig(dropout_rate=0.0), rngs=nnx.Rngs(0))
    serde = Serde(original)
    serialized = serde.serialize()
    print(len(serialized))
    deserialized = serde.deserialize(serialized)
    assert original == deserialized
