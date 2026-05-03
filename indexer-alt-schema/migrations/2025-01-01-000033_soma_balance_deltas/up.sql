-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Stage 13m: per-checkpoint per-(owner, coin_type) signed balance
-- delta. Sourced from `TransactionEffects.balance_events` by the
-- `soma_balance_deltas` indexer handler — every checkpoint that
-- touches a balance produces one row per (owner, coin_type) the
-- checkpoint affected, with the net signed delta across all txs in
-- that checkpoint.
--
-- Current balance is `SUM(delta) WHERE owner = ? AND coin_type = ?`.
-- The (owner) index makes per-address scans cheap; the primary key
-- gives us the per-(owner, coin_type, cp) ordering needed for
-- pruning and replay.
CREATE TABLE soma_balance_deltas (
    owner              BYTEA  NOT NULL,
    coin_type          TEXT   NOT NULL,
    cp_sequence_number BIGINT NOT NULL,
    -- Signed; negative = net withdraw, positive = net deposit.
    -- `BIGINT` (i64) is the safe domain type — total supply fits
    -- well under 2^63 even though the on-chain accumulator stores u64.
    delta              BIGINT NOT NULL,
    PRIMARY KEY (owner, coin_type, cp_sequence_number)
);

CREATE INDEX soma_balance_deltas_owner
    ON soma_balance_deltas (owner, coin_type);
