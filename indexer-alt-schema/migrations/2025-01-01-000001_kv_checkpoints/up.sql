-- Copyright (c) Mysten Labs, Inc.
-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS kv_checkpoints
(
    sequence_number                     BIGINT       PRIMARY KEY,
    checkpoint_contents                 BYTEA        NOT NULL,
    checkpoint_summary                  BYTEA        NOT NULL,
    validator_signatures                BYTEA        NOT NULL
);
