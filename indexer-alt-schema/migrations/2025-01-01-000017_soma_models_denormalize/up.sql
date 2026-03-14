-- Denormalize soma_models: extract all useful fields from state_bcs into proper columns.

ALTER TABLE soma_models
    ADD COLUMN next_epoch_commission_rate  BIGINT NOT NULL DEFAULT 0,
    -- StakingPool fields
    ADD COLUMN staking_pool_id            BYTEA  NOT NULL DEFAULT '\x',
    ADD COLUMN activation_epoch           BIGINT,
    ADD COLUMN deactivation_epoch         BIGINT,
    ADD COLUMN rewards_pool               BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN pool_token_balance         BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN pending_stake              BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN pending_total_soma_withdraw BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN pending_pool_token_withdraw BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN exchange_rates_json        TEXT   NOT NULL DEFAULT '{}',
    -- Manifest fields (only Pending/Active/Inactive)
    ADD COLUMN manifest_url               TEXT,
    ADD COLUMN manifest_checksum          BYTEA,
    ADD COLUMN manifest_size              BIGINT,
    -- Commitment digests (only Pending/Active/Inactive)
    ADD COLUMN weights_commitment         BYTEA,
    ADD COLUMN embedding_commitment       BYTEA,
    ADD COLUMN decryption_key_commitment  BYTEA,
    -- Active/Inactive only
    ADD COLUMN decryption_key             BYTEA,
    -- Active only
    ADD COLUMN has_pending_update         BOOLEAN NOT NULL DEFAULT FALSE;

-- Drop temporary defaults for NOT NULL columns
ALTER TABLE soma_models ALTER COLUMN next_epoch_commission_rate DROP DEFAULT;
ALTER TABLE soma_models ALTER COLUMN staking_pool_id DROP DEFAULT;
ALTER TABLE soma_models ALTER COLUMN rewards_pool DROP DEFAULT;
ALTER TABLE soma_models ALTER COLUMN pool_token_balance DROP DEFAULT;
ALTER TABLE soma_models ALTER COLUMN pending_stake DROP DEFAULT;
ALTER TABLE soma_models ALTER COLUMN pending_total_soma_withdraw DROP DEFAULT;
ALTER TABLE soma_models ALTER COLUMN pending_pool_token_withdraw DROP DEFAULT;
ALTER TABLE soma_models ALTER COLUMN exchange_rates_json DROP DEFAULT;
ALTER TABLE soma_models ALTER COLUMN has_pending_update DROP DEFAULT;

-- Drop the opaque BCS blob
ALTER TABLE soma_models DROP COLUMN state_bcs;

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS soma_models_stake
    ON soma_models (stake, epoch);

CREATE INDEX IF NOT EXISTS soma_models_has_embedding
    ON soma_models (model_id) WHERE has_embedding = TRUE;
