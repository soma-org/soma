-- Reverse model denormalization: restore state_bcs, drop new columns and indexes.

DROP INDEX IF EXISTS soma_models_has_embedding;
DROP INDEX IF EXISTS soma_models_stake;

ALTER TABLE soma_models ADD COLUMN state_bcs BYTEA NOT NULL DEFAULT '\x';

ALTER TABLE soma_models
    DROP COLUMN IF EXISTS next_epoch_commission_rate,
    DROP COLUMN IF EXISTS staking_pool_id,
    DROP COLUMN IF EXISTS activation_epoch,
    DROP COLUMN IF EXISTS deactivation_epoch,
    DROP COLUMN IF EXISTS rewards_pool,
    DROP COLUMN IF EXISTS pool_token_balance,
    DROP COLUMN IF EXISTS pending_stake,
    DROP COLUMN IF EXISTS pending_total_soma_withdraw,
    DROP COLUMN IF EXISTS pending_pool_token_withdraw,
    DROP COLUMN IF EXISTS exchange_rates_json,
    DROP COLUMN IF EXISTS manifest_url,
    DROP COLUMN IF EXISTS manifest_checksum,
    DROP COLUMN IF EXISTS manifest_size,
    DROP COLUMN IF EXISTS weights_commitment,
    DROP COLUMN IF EXISTS embedding_commitment,
    DROP COLUMN IF EXISTS decryption_key_commitment,
    DROP COLUMN IF EXISTS decryption_key,
    DROP COLUMN IF EXISTS has_pending_update;
