-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS soma_targets
(
    target_id                       BYTEA        NOT NULL,
    cp_sequence_number              BIGINT       NOT NULL,
    epoch                           BIGINT       NOT NULL,
    status                          TEXT         NOT NULL,
    submitter                       BYTEA,
    winning_model_id                BYTEA,
    reward_pool                     BIGINT       NOT NULL,
    bond_amount                     BIGINT       NOT NULL,
    report_count                    INT          NOT NULL DEFAULT 0,
    state_bcs                       BYTEA        NOT NULL,
    PRIMARY KEY (target_id, cp_sequence_number)
);

CREATE INDEX IF NOT EXISTS soma_targets_epoch
ON soma_targets (epoch);

CREATE INDEX IF NOT EXISTS soma_targets_status
ON soma_targets (status, epoch);
