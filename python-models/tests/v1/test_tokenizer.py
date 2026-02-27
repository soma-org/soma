"""Tests for the V1 byte-level tokenizer.

Test cases mirror the Rust tests in models/src/v1/data/dataset.rs and
models/src/v1/data/batcher.rs to ensure exact parity.
"""

from soma_models.v1.configs import V1_EOS_TOKEN_ID, V1_PAD_TOKEN_ID
from soma_models.v1.tokenizer import tokenize


def _test_buffer(size):
    return bytes(i % 256 for i in range(size))


def test_empty_buffer():
    assert tokenize(b"", max_seq_len=16) == []


def test_single_byte():
    sequences = tokenize(_test_buffer(1), max_seq_len=4)
    assert len(sequences) == 1
    seq = sequences[0]

    assert seq.token_ids == [0, V1_EOS_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID]
    assert seq.pos_ids == [0, 1, 1, 1]
    assert seq.targets == [V1_EOS_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID]


def test_full_chunk_no_eos():
    seq_len = 8
    sequences = tokenize(_test_buffer(16), max_seq_len=seq_len)
    assert len(sequences) == 2

    # First chunk: bytes 0..8
    for i in range(seq_len):
        assert sequences[0].token_ids[i] == i
        assert sequences[0].pos_ids[i] == i
    assert V1_EOS_TOKEN_ID not in sequences[0].token_ids

    # Second chunk: bytes 8..16
    for i in range(seq_len):
        assert sequences[1].token_ids[i] == 8 + i
        assert sequences[1].pos_ids[i] == 8 + i
    assert V1_EOS_TOKEN_ID not in sequences[1].token_ids


def test_last_chunk_with_eos_and_padding():
    seq_len = 6
    sequences = tokenize(_test_buffer(10), max_seq_len=seq_len)
    assert len(sequences) == 2

    last = sequences[1]

    # 4 data bytes (6..10)
    for i in range(4):
        assert last.token_ids[i] == 6 + i
        assert last.pos_ids[i] == 6 + i

    # EOS at position 4
    assert last.token_ids[4] == V1_EOS_TOKEN_ID
    assert last.pos_ids[4] == 10

    # PAD at position 5
    assert last.token_ids[5] == V1_PAD_TOKEN_ID
    assert last.pos_ids[5] == 10


def test_exact_multiple_no_eos():
    seq_len = 5
    sequences = tokenize(_test_buffer(10), max_seq_len=seq_len)
    assert len(sequences) == 2

    last = sequences[1]

    for i in range(5):
        assert last.token_ids[i] == 5 + i
        assert last.pos_ids[i] == 5 + i

    assert V1_EOS_TOKEN_ID not in last.token_ids


def test_targets_shift_left():
    """Targets are token_ids shifted left by 1 with PAD appended (matches Rust batcher)."""
    seq_len = 5
    sequences = tokenize(_test_buffer(10), max_seq_len=seq_len)

    # token_ids: [[0,1,2,3,4], [5,6,7,8,9]]
    # targets:   [[1,2,3,4,PAD], [6,7,8,9,PAD]]
    assert sequences[0].token_ids == [0, 1, 2, 3, 4]
    assert sequences[0].targets == [1, 2, 3, 4, V1_PAD_TOKEN_ID]
    assert sequences[1].token_ids == [5, 6, 7, 8, 9]
    assert sequences[1].targets == [6, 7, 8, 9, V1_PAD_TOKEN_ID]


def test_chunk_count():
    """Number of chunks matches Rust div_ceil logic."""
    cases = [(0, 0), (1, 1), (9, 1), (10, 1), (11, 2), (20, 2), (21, 3)]
    for size, expected in cases:
        data = _test_buffer(size)
        if size == 0:
            assert tokenize(data, max_seq_len=10) == []
            continue
        sequences = tokenize(data, max_seq_len=10)
        assert len(sequences) == expected, f"size {size}: expected {expected} chunks, got {len(sequences)}"
