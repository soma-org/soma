-- Copyright (c) Mysten Labs, Inc.
-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS coin_balance_buckets
(
    object_id                   BYTEA         NOT NULL,
    cp_sequence_number          BIGINT        NOT NULL,
    owner_kind                  SMALLINT,
    owner_id                    BYTEA,
    coin_type                   BYTEA,
    coin_balance_bucket         SMALLINT,
    PRIMARY KEY (object_id, cp_sequence_number)
);

CREATE INDEX IF NOT EXISTS coin_balances_buckets_object_id_desc
ON coin_balance_buckets (owner_kind, owner_id, coin_type, coin_balance_bucket DESC, cp_sequence_number DESC, object_id DESC);
