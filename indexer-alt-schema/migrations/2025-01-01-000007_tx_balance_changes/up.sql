-- Copyright (c) Mysten Labs, Inc.
-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS tx_balance_changes
(
    tx_sequence_number          BIGINT        PRIMARY KEY,
    -- BCS serialized array of BalanceChanges
    balance_changes             BYTEA         NOT NULL
);
