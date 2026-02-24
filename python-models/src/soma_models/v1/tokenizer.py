"""Framework-agnostic byte-level tokenizer matching the on-chain V1 data contract.

Converts raw bytes into (token_ids, targets, pos_ids) batches that can be
consumed by any ML framework — wrap the returned lists with ``torch.tensor()``,
``jnp.array()``, ``tf.constant()``, etc.

The logic mirrors the Rust ``ByteSequenceDataset`` / ``ByteSequenceBatcher``
in ``models/src/v1/data/`` exactly.
"""

from __future__ import annotations

from soma_models.v1.configs import (
    V1_BATCH_SIZE,
    V1_EOS_TOKEN_ID,
    V1_MAX_SEQ_LEN,
    V1_PAD_TOKEN_ID,
)


class ByteSequenceBatch:
    """A single batch of tokenized byte sequences.

    Attributes:
        token_ids: ``[batch, seq_len]`` nested list of ints.
        targets: ``[batch, seq_len]`` nested list of ints (token_ids shifted
            left by 1, with PAD appended).
        pos_ids: ``[batch, seq_len]`` nested list of ints (global byte
            offsets; PAD/EOS positions clamped to last-data-byte + 1).
    """

    __slots__ = ("token_ids", "targets", "pos_ids")

    def __init__(
        self,
        token_ids: list[list[int]],
        targets: list[list[int]],
        pos_ids: list[list[int]],
    ) -> None:
        self.token_ids = token_ids
        self.targets = targets
        self.pos_ids = pos_ids


def tokenize(
    data: bytes | bytearray,
    max_seq_len: int = V1_MAX_SEQ_LEN,
    batch_size: int = V1_BATCH_SIZE,
) -> list[ByteSequenceBatch]:
    """Tokenize raw bytes into batches matching the on-chain V1 data contract.
    Args:
        data: Raw byte data to tokenize.
        max_seq_len: Maximum sequence length per chunk.
        batch_size: Number of sequences per batch.

    Returns:
        A list of ``ByteSequenceBatch`` instances.  The final batch may
        contain fewer than ``batch_size`` sequences (matching the Rust
        DataLoader behaviour).
    """
    if len(data) == 0:
        return []

    num_chunks = -(-len(data) // max_seq_len)  # ceil division

    items: list[tuple[list[int], list[int], list[int]]] = []
    for index in range(num_chunks):
        start = index * max_seq_len
        remaining = len(data) - start
        data_len = min(remaining, max_seq_len)

        is_final = index + 1 == num_chunks
        has_room = data_len < max_seq_len
        eos_pos = data_len if (is_final and has_room) else -1

        token_ids: list[int] = []
        pos_ids: list[int] = []
        pos_after_last = start + data_len

        for i in range(max_seq_len):
            if i < data_len:
                token_ids.append(data[start + i])
                pos_ids.append(start + i)
            elif i == eos_pos:
                token_ids.append(V1_EOS_TOKEN_ID)
                pos_ids.append(pos_after_last)
            else:
                token_ids.append(V1_PAD_TOKEN_ID)
                pos_ids.append(pos_after_last)

        targets = token_ids[1:] + [V1_PAD_TOKEN_ID]
        items.append((token_ids, targets, pos_ids))

    batches: list[ByteSequenceBatch] = []
    for i in range(0, len(items), batch_size):
        batch_items = items[i : i + batch_size]
        batch_token_ids = [item[0] for item in batch_items]
        batch_targets = [item[1] for item in batch_items]
        batch_pos_ids = [item[2] for item in batch_items]
        batches.append(ByteSequenceBatch(batch_token_ids, batch_targets, batch_pos_ids))

    return batches
