-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS soma_rewards
(
    target_id                       BYTEA        NOT NULL,
    cp_sequence_number              BIGINT       NOT NULL,
    epoch                           BIGINT       NOT NULL,
    tx_digest                       BYTEA        NOT NULL,
    balance_changes_bcs             BYTEA        NOT NULL,
    PRIMARY KEY (target_id)
);

CREATE INDEX IF NOT EXISTS soma_rewards_epoch
ON soma_rewards (epoch);
