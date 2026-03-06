-- Copyright (c) Mysten Labs, Inc.
-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS cp_sequence_numbers
(
    cp_sequence_number                  BIGINT       PRIMARY KEY,
    tx_lo                               BIGINT       NOT NULL,
    epoch                               BIGINT       NOT NULL
);
