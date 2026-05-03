-- Copyright (c) Mysten Labs, Inc.
-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS obj_info_deletion_reference
(
    object_id                   BYTEA         NOT NULL,
    cp_sequence_number          BIGINT        NOT NULL,
    PRIMARY KEY (cp_sequence_number, object_id)
);

-- Stage 13i: coin_balance_buckets_deletion_reference removed
-- alongside the coin_balance_buckets table.
