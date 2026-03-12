-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS soma_models
(
    model_id                        BYTEA        NOT NULL,
    epoch                           BIGINT       NOT NULL,
    status                          TEXT         NOT NULL,
    owner                           BYTEA        NOT NULL,
    architecture_version            BIGINT       NOT NULL,
    commit_epoch                    BIGINT       NOT NULL,
    stake                           BIGINT       NOT NULL,
    commission_rate                 BIGINT       NOT NULL,
    has_embedding                   BOOLEAN      NOT NULL,
    state_bcs                       BYTEA        NOT NULL,
    PRIMARY KEY (model_id, epoch)
);

CREATE INDEX IF NOT EXISTS soma_models_status
ON soma_models (status, epoch);

CREATE INDEX IF NOT EXISTS soma_models_owner
ON soma_models (owner);
