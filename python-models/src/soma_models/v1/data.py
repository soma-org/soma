"""Data preparation matching the Rust on-chain evaluation pipeline.

This replicates models/src/v1/data/dataset.rs and batcher.rs exactly:
bytes are chunked into fixed-length sequences, EOS is placed on the final
chunk (if it fits), remaining positions are PAD, targets are token_ids
shifted left by 1, and position IDs use global byte offsets with clamping
for non-data positions.
"""

import numpy as np
from soma_models.config import (
    V1_PAD_TOKEN_ID,
    V1_EOS_TOKEN_ID,
    V1_MAX_SEQ_LEN,
    V1_BATCH_SIZE,
)


def prepare_batches(
    data: bytes,
    seq_len: int = V1_MAX_SEQ_LEN,
    batch_size: int = V1_BATCH_SIZE,
) -> list[dict[str, np.ndarray]]:
    """Chunk raw bytes into evaluation batches matching the Rust runtime.

    Replicates ByteSequenceDataset + ByteSequenceBatcher from the Rust
    models crate. Each returned dict has:
        token_ids:  int32 [batch, seq_len]
        positions:  int32 [batch, seq_len]
        targets:    int32 [batch, seq_len]
        attn_mask:  bool  [batch, 1, seq_len, seq_len]  (causal)

    Args:
        data: Raw bytes to evaluate.
        seq_len: Sequence length per chunk (default: V1_MAX_SEQ_LEN).
        batch_size: Batch size (default: V1_BATCH_SIZE).

    Returns:
        List of batch dicts. The last batch may have fewer than batch_size
        sequences. Returns an empty list if data is empty.
    """
    if len(data) == 0:
        return []

    num_chunks = -(-len(data) // seq_len)  # ceil division

    # Build all sequence items (dataset.rs logic)
    all_token_ids = []
    all_pos_ids = []
    for idx in range(num_chunks):
        start = idx * seq_len
        remaining = len(data) - start
        data_len = min(remaining, seq_len)

        is_final = idx + 1 == num_chunks
        has_room = data_len < seq_len
        eos_slot = data_len if (is_final and has_room) else None

        token_ids = np.full(seq_len, V1_PAD_TOKEN_ID, dtype=np.int32)
        pos_ids = np.empty(seq_len, dtype=np.int32)
        pos_after_last = start + data_len

        for i in range(seq_len):
            if i < data_len:
                token_ids[i] = data[start + i]
                pos_ids[i] = start + i
            else:
                if eos_slot is not None and i == eos_slot:
                    token_ids[i] = V1_EOS_TOKEN_ID
                pos_ids[i] = pos_after_last

        all_token_ids.append(token_ids)
        all_pos_ids.append(pos_ids)

    # Batch items and build targets (batcher.rs logic)
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))[np.newaxis, :, :]

    batches = []
    for batch_start in range(0, num_chunks, batch_size):
        batch_end = min(batch_start + batch_size, num_chunks)
        b = batch_end - batch_start

        token_ids = np.stack(all_token_ids[batch_start:batch_end])
        pos_ids = np.stack(all_pos_ids[batch_start:batch_end])

        # targets = token_ids shifted left by 1, PAD appended
        targets = np.concatenate(
            [token_ids[:, 1:], np.full((b, 1), V1_PAD_TOKEN_ID, dtype=np.int32)],
            axis=1,
        )

        attn_mask = np.broadcast_to(causal_mask, (b, seq_len, seq_len)).copy()
        attn_mask = attn_mask[:, np.newaxis, :, :]  # [b, 1, seq, seq]

        batches.append({
            "token_ids": token_ids,
            "positions": pos_ids,
            "targets": targets,
            "attn_mask": attn_mask,
        })

    return batches
