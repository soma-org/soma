"""Tests for v1 data preparation, verifying parity with Rust dataset.rs + batcher.rs."""

import numpy as np
from soma_models.config import V1_PAD_TOKEN_ID, V1_EOS_TOKEN_ID
from soma_models.v1.data import prepare_batches


def test_empty_data():
    assert prepare_batches(b"", seq_len=8, batch_size=2) == []


def test_exact_multiple_no_eos():
    """Data length is exact multiple of seq_len — no EOS, no padding."""
    data = bytes(range(10))
    batches = prepare_batches(data, seq_len=5, batch_size=4)

    assert len(batches) == 1
    b = batches[0]
    assert b["token_ids"].shape == (2, 5)
    assert b["positions"].shape == (2, 5)
    assert b["targets"].shape == (2, 5)

    # First chunk: [0,1,2,3,4]
    np.testing.assert_array_equal(b["token_ids"][0], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(b["positions"][0], [0, 1, 2, 3, 4])
    # Targets = shifted left by 1, PAD appended
    np.testing.assert_array_equal(b["targets"][0], [1, 2, 3, 4, V1_PAD_TOKEN_ID])

    # Second chunk: [5,6,7,8,9] — no EOS because exact multiple
    np.testing.assert_array_equal(b["token_ids"][1], [5, 6, 7, 8, 9])
    np.testing.assert_array_equal(b["positions"][1], [5, 6, 7, 8, 9])
    np.testing.assert_array_equal(b["targets"][1], [6, 7, 8, 9, V1_PAD_TOKEN_ID])
    assert V1_EOS_TOKEN_ID not in b["token_ids"]


def test_final_chunk_eos_and_padding():
    """Last chunk has room — gets EOS then PAD, positions clamped."""
    data = bytes(range(10))
    batches = prepare_batches(data, seq_len=6, batch_size=4)

    assert len(batches) == 1
    b = batches[0]
    assert b["token_ids"].shape == (2, 6)

    # Second chunk: 4 data bytes [6,7,8,9], EOS, PAD
    np.testing.assert_array_equal(
        b["token_ids"][1],
        [6, 7, 8, 9, V1_EOS_TOKEN_ID, V1_PAD_TOKEN_ID],
    )
    # Positions: global offsets for data, clamped for EOS/PAD
    np.testing.assert_array_equal(b["positions"][1], [6, 7, 8, 9, 10, 10])


def test_single_byte():
    """One byte of data — gets EOS + PAD fill."""
    data = b"\x00"
    batches = prepare_batches(data, seq_len=4, batch_size=2)

    assert len(batches) == 1
    b = batches[0]
    assert b["token_ids"].shape == (1, 4)
    np.testing.assert_array_equal(
        b["token_ids"][0],
        [0, V1_EOS_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID],
    )
    np.testing.assert_array_equal(b["positions"][0], [0, 1, 1, 1])


def test_batching_splits_correctly():
    """Multiple batches with a partial last batch."""
    data = bytes(30)  # 30 bytes, seq_len=10 → 3 chunks, batch_size=2 → 2 batches
    batches = prepare_batches(data, seq_len=10, batch_size=2)

    assert len(batches) == 2
    assert batches[0]["token_ids"].shape == (2, 10)
    assert batches[1]["token_ids"].shape == (1, 10)


def test_causal_mask_shape_and_values():
    data = bytes(4)
    batches = prepare_batches(data, seq_len=4, batch_size=1)
    mask = batches[0]["attn_mask"]

    assert mask.shape == (1, 1, 4, 4)
    assert mask.dtype == bool
    # Lower triangle should be True
    expected = np.tril(np.ones((4, 4), dtype=bool))
    np.testing.assert_array_equal(mask[0, 0], expected)


def test_targets_match_rust_batcher():
    """Targets are token_ids[1:] + [PAD], matching Rust ByteSequenceBatcher."""
    data = bytes([10, 20, 30, 40, 50])
    batches = prepare_batches(data, seq_len=5, batch_size=1)
    b = batches[0]

    # token_ids = [10, 20, 30, 40, 50] (exact multiple, no EOS)
    np.testing.assert_array_equal(b["token_ids"][0], [10, 20, 30, 40, 50])
    # targets = [20, 30, 40, 50, PAD]
    np.testing.assert_array_equal(b["targets"][0], [20, 30, 40, 50, V1_PAD_TOKEN_ID])
