"""Exact parity tests against Rust data pipeline.

Each test reproduces a specific Rust test from models/src/v1/data/dataset.rs
and models/src/v1/data/batcher.rs, asserting identical token_ids, pos_ids,
and targets values.
"""

from soma_models.v1.configs import V1_EOS_TOKEN_ID, V1_PAD_TOKEN_ID
from soma_models.v1.tokenizer import tokenize

PAD = V1_PAD_TOKEN_ID  # 256
EOS = V1_EOS_TOKEN_ID  # 257


def _test_buffer(size):
    return bytes(i % 256 for i in range(size))


# ── dataset.rs::test_dataset_length ──────────────────────────────────────

def test_rust_dataset_length():
    """Mirrors dataset.rs::test_dataset_length — chunk counts for various sizes."""
    cases = [(0, 0), (1, 1), (9, 1), (10, 1), (11, 2), (20, 2), (21, 3)]
    for size, expected in cases:
        sequences = tokenize(_test_buffer(size), max_seq_len=10)
        assert len(sequences) == expected, f"size {size}: expected {expected} chunks, got {len(sequences)}"


# ── dataset.rs::test_full_chunk_no_eos ───────────────────────────────────

def test_rust_full_chunk_no_eos():
    """Mirrors dataset.rs::test_full_chunk_no_eos — 16 bytes, seq=8, 2 full chunks."""
    sequences = tokenize(_test_buffer(16), max_seq_len=8)
    assert len(sequences) == 2

    # Chunk 0: bytes 0..8
    assert sequences[0].token_ids == [0, 1, 2, 3, 4, 5, 6, 7]
    assert sequences[0].pos_ids == [0, 1, 2, 3, 4, 5, 6, 7]

    # Chunk 1: bytes 8..16
    assert sequences[1].token_ids == [8, 9, 10, 11, 12, 13, 14, 15]
    assert sequences[1].pos_ids == [8, 9, 10, 11, 12, 13, 14, 15]

    # No EOS in either chunk (exact multiple)
    assert EOS not in sequences[0].token_ids
    assert EOS not in sequences[1].token_ids


# ── dataset.rs::test_last_chunk_with_eos_and_padding ─────────────────────

def test_rust_last_chunk_with_eos_and_padding():
    """Mirrors dataset.rs::test_last_chunk_with_eos_and_padding — 10 bytes, seq=6."""
    sequences = tokenize(_test_buffer(10), max_seq_len=6)
    assert len(sequences) == 2

    seq = sequences[1]

    # 4 data bytes at positions 6..10
    assert seq.token_ids[:4] == [6, 7, 8, 9]
    assert seq.pos_ids[:4] == [6, 7, 8, 9]

    # EOS at index 4, pos clamped to 10
    assert seq.token_ids[4] == EOS
    assert seq.pos_ids[4] == 10

    # PAD at index 5, pos clamped to 10
    assert seq.token_ids[5] == PAD
    assert seq.pos_ids[5] == 10


# ── dataset.rs::test_exact_multiple_no_eos ───────────────────────────────

def test_rust_exact_multiple_no_eos():
    """Mirrors dataset.rs::test_exact_multiple_no_eos — 10 bytes, seq=5."""
    sequences = tokenize(_test_buffer(10), max_seq_len=5)

    seq = sequences[1]

    assert seq.token_ids == [5, 6, 7, 8, 9]
    assert seq.pos_ids == [5, 6, 7, 8, 9]
    assert EOS not in seq.token_ids


# ── dataset.rs::test_empty_buffer ────────────────────────────────────────

def test_rust_empty_buffer():
    """Mirrors dataset.rs::test_empty_buffer — 0 bytes produces no items."""
    assert tokenize(b"", max_seq_len=16) == []


# ── dataset.rs::test_single_byte ─────────────────────────────────────────

def test_rust_single_byte():
    """Mirrors dataset.rs::test_single_byte — 1 byte, seq=4."""
    sequences = tokenize(_test_buffer(1), max_seq_len=4)
    assert len(sequences) == 1

    seq = sequences[0]

    assert seq.token_ids == [0, EOS, PAD, PAD]
    assert seq.pos_ids == [0, 1, 1, 1]


# ── batcher.rs::test_dataset_length ──────────────────────────────────────

def test_rust_batcher_full():
    """Mirrors batcher.rs::test_dataset_length — 10 bytes, seq=5.

    Asserts exact flattened values for token_ids, pos_ids, and targets
    matching the Rust test's i64 vectors.
    """
    sequences = tokenize(_test_buffer(10), max_seq_len=5)
    assert len(sequences) == 2

    assert len(sequences[0].token_ids) == 5
    assert len(sequences[1].token_ids) == 5

    # Exact flattened values from Rust
    flat_tokens = [t for s in sequences for t in s.token_ids]
    flat_pos = [p for s in sequences for p in s.pos_ids]
    flat_targets = [t for s in sequences for t in s.targets]

    assert flat_tokens == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert flat_pos == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert flat_targets == [1, 2, 3, 4, 256, 6, 7, 8, 9, 256]
