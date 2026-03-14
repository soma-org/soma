-- Copyright (c) Mysten Labs, Inc.
-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS tx_affected_objects
(
    tx_sequence_number          BIGINT       NOT NULL,
    affected                    BYTEA        NOT NULL,
    sender                      BYTEA        NOT NULL,
    PRIMARY KEY(affected, tx_sequence_number)
);

CREATE INDEX IF NOT EXISTS tx_affected_objects_tx_sequence_number
ON tx_affected_objects (tx_sequence_number);

CREATE INDEX IF NOT EXISTS tx_affected_objects_sender
ON tx_affected_objects (sender, affected, tx_sequence_number);
