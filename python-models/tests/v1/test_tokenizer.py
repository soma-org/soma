"""Tests for the V1 byte-level tokenizer.

Test cases mirror the Rust tests in models/src/v1/data/dataset.rs and
models/src/v1/data/batcher.rs to ensure exact parity.
"""

from soma_models.v1.configs import V1_EOS_TOKEN_ID, V1_PAD_TOKEN_ID
from soma_models.v1.tokenizer import tokenize


def _test_buffer(size):
    return bytes(i % 256 for i in range(size))


def test_empty_buffer():
    assert tokenize(b"", max_seq_len=16, batch_size=2) == []


def test_single_byte():
    batches = tokenize(_test_buffer(1), max_seq_len=4, batch_size=32)
    assert len(batches) == 1
    batch = batches[0]
    assert len(batch.token_ids) == 1

    assert batch.token_ids[0] == [0, V1_EOS_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID]
    assert batch.pos_ids[0] == [0, 1, 1, 1]
    assert batch.targets[0] == [V1_EOS_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID]


def test_full_chunk_no_eos():
    seq = 8
    batches = tokenize(_test_buffer(16), max_seq_len=seq, batch_size=32)
    assert len(batches) == 1
    batch = batches[0]
    assert len(batch.token_ids) == 2

    # First chunk: bytes 0..8
    for i in range(seq):
        assert batch.token_ids[0][i] == i
        assert batch.pos_ids[0][i] == i
    assert V1_EOS_TOKEN_ID not in batch.token_ids[0]

    # Second chunk: bytes 8..16
    for i in range(seq):
        assert batch.token_ids[1][i] == 8 + i
        assert batch.pos_ids[1][i] == 8 + i
    assert V1_EOS_TOKEN_ID not in batch.token_ids[1]


def test_last_chunk_with_eos_and_padding():
    seq = 6
    batches = tokenize(_test_buffer(10), max_seq_len=seq, batch_size=32)
    assert len(batches) == 1
    batch = batches[0]
    assert len(batch.token_ids) == 2

    last = batch.token_ids[1]
    last_pos = batch.pos_ids[1]

    # 4 data bytes (6..10)
    for i in range(4):
        assert last[i] == 6 + i
        assert last_pos[i] == 6 + i

    # EOS at position 4
    assert last[4] == V1_EOS_TOKEN_ID
    assert last_pos[4] == 10

    # PAD at position 5
    assert last[5] == V1_PAD_TOKEN_ID
    assert last_pos[5] == 10


def test_exact_multiple_no_eos():
    seq = 5
    batches = tokenize(_test_buffer(10), max_seq_len=seq, batch_size=32)
    assert len(batches) == 1
    batch = batches[0]

    last = batch.token_ids[1]
    last_pos = batch.pos_ids[1]

    for i in range(5):
        assert last[i] == 5 + i
        assert last_pos[i] == 5 + i

    assert V1_EOS_TOKEN_ID not in last


def test_targets_shift_left():
    """Targets are token_ids shifted left by 1 with PAD appended (matches Rust batcher)."""
    seq = 5
    batches = tokenize(_test_buffer(10), max_seq_len=seq, batch_size=32)
    batch = batches[0]

    # token_ids: [[0,1,2,3,4], [5,6,7,8,9]]
    # targets:   [[1,2,3,4,PAD], [6,7,8,9,PAD]]
    assert batch.token_ids[0] == [0, 1, 2, 3, 4]
    assert batch.targets[0] == [1, 2, 3, 4, V1_PAD_TOKEN_ID]
    assert batch.token_ids[1] == [5, 6, 7, 8, 9]
    assert batch.targets[1] == [6, 7, 8, 9, V1_PAD_TOKEN_ID]


def test_batching():
    """Multiple batches are produced when items exceed batch_size."""
    seq = 10
    batches = tokenize(_test_buffer(30), max_seq_len=seq, batch_size=2)

    # 30 bytes / seq 10 = 3 items, batch_size 2 → two batches: [2, 10] then [1, 10]
    assert len(batches) == 2
    assert len(batches[0].token_ids) == 2
    assert len(batches[1].token_ids) == 1

    # Shape consistency within each batch
    for batch in batches:
        for row in batch.token_ids:
            assert len(row) == seq
        for row in batch.targets:
            assert len(row) == seq
        for row in batch.pos_ids:
            assert len(row) == seq


def test_parity_with_rust_batcher():
    """End-to-end match with the Rust batcher test in batcher.rs."""
    seq = 5
    batches = tokenize(_test_buffer(10), max_seq_len=seq, batch_size=2)
    assert len(batches) == 1
    batch = batches[0]

    # Flatten for comparison (matches Rust test assertions)
    flat_tokens = [t for row in batch.token_ids for t in row]
    flat_pos = [p for row in batch.pos_ids for p in row]
    flat_targets = [t for row in batch.targets for t in row]

    assert flat_tokens == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert flat_pos == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert flat_targets == [1, 2, 3, 4, 256, 6, 7, 8, 9, 256]


def test_chunk_count():
    """Number of chunks matches Rust div_ceil logic."""
    cases = [(0, 0), (1, 1), (9, 1), (10, 1), (11, 2), (20, 2), (21, 3)]
    for size, expected in cases:
        data = _test_buffer(size)
        if size == 0:
            assert tokenize(data, max_seq_len=10, batch_size=100) == []
            continue
        batches = tokenize(data, max_seq_len=10, batch_size=100)
        total_items = sum(len(b.token_ids) for b in batches)
        assert total_items == expected, f"size {size}: expected {expected} chunks, got {total_items}"
