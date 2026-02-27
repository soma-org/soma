"""Framework-agnostic byte-level tokenizer matching the on-chain V1 data contract.

Converts raw bytes into a list of ``TokenizedSequence`` items (token_ids,
targets, pos_ids) that can be consumed by any ML framework — wrap the
returned lists with ``torch.tensor()``, ``jnp.array()``, ``tf.constant()``,
etc.

The logic mirrors the Rust ``ByteSequenceDataset`` in
``models/src/v1/data/`` exactly.
"""

from __future__ import annotations

from typing import NamedTuple

from soma_models.v1.configs import (
    V1_EOS_TOKEN_ID,
    V1_MAX_SEQ_LEN,
    V1_PAD_TOKEN_ID,
)


class TokenizedSequence(NamedTuple):
    """A single tokenized byte sequence.

    Attributes:
        token_ids: ``[seq_len]`` list of ints.
        targets: ``[seq_len]`` list of ints (token_ids shifted left by 1,
            with PAD appended).
        pos_ids: ``[seq_len]`` list of ints (global byte offsets; PAD/EOS
            positions clamped to last-data-byte + 1).
    """

    token_ids: list[int]
    targets: list[int]
    pos_ids: list[int]


def tokenize(
    data: bytes | bytearray,
    max_seq_len: int = V1_MAX_SEQ_LEN,
) -> list[TokenizedSequence]:
    """Tokenize raw bytes into sequences matching the on-chain V1 data contract.

    Args:
        data: Raw byte data to tokenize.
        max_seq_len: Maximum sequence length per chunk.

    Returns:
        A list of ``TokenizedSequence`` instances, one per chunk.
    """
    if len(data) == 0:
        return []

    num_chunks = -(-len(data) // max_seq_len)  # ceil division

    sequences: list[TokenizedSequence] = []
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
        sequences.append(TokenizedSequence(token_ids, targets, pos_ids))

    return sequences
