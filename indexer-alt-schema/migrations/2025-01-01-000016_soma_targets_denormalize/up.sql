-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Fully denormalize TargetV1 fields so GraphQL never needs to decode state_bcs.

ALTER TABLE soma_targets
    ADD COLUMN winning_distance_score   DOUBLE PRECISION,
    ADD COLUMN winning_loss_score       DOUBLE PRECISION,
    ADD COLUMN winning_model_owner      BYTEA,
    ADD COLUMN fill_epoch               BIGINT,
    ADD COLUMN distance_threshold       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ADD COLUMN model_ids_json           TEXT NOT NULL DEFAULT '[]',
    ADD COLUMN winning_data_url         TEXT,
    ADD COLUMN winning_data_checksum    BYTEA,
    ADD COLUMN winning_data_size        BIGINT;

-- Remove the DEFAULT after backfill (new rows always supply values).
ALTER TABLE soma_targets ALTER COLUMN distance_threshold DROP DEFAULT;
ALTER TABLE soma_targets ALTER COLUMN model_ids_json DROP DEFAULT;

-- Support filtering by submitter ("all targets filled by this address")
CREATE INDEX IF NOT EXISTS soma_targets_submitter
ON soma_targets (submitter) WHERE submitter IS NOT NULL;

-- Support filtering by winning_model_id ("all targets won by this model")
CREATE INDEX IF NOT EXISTS soma_targets_winning_model_id
ON soma_targets (winning_model_id) WHERE winning_model_id IS NOT NULL;

-- Support filtering by fill_epoch ("targets filled during epoch X")
CREATE INDEX IF NOT EXISTS soma_targets_fill_epoch
ON soma_targets (fill_epoch) WHERE fill_epoch IS NOT NULL;

-- Support filtering by winning_model_owner ("rewards going to this owner")
CREATE INDEX IF NOT EXISTS soma_targets_winning_model_owner
ON soma_targets (winning_model_owner) WHERE winning_model_owner IS NOT NULL;
