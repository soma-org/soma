import pytest
from flax import nnx
from soma_probes.flax.v1.modules.pwff import PositionWiseFeedForward
from soma_probes.flax.serde import Serde


def test_flax_serde():
    original_pwff = PositionWiseFeedForward(rngs=nnx.Rngs(0))
    pwff_serde = Serde(original_pwff)
    serialized_pwff = pwff_serde.serialize()
    deserialized_pwff = pwff_serde.deserialize(serialized_pwff)
    assert original_pwff == deserialized_pwff
