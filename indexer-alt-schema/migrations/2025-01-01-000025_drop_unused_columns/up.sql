-- Drop legacy `state_bcs` from soma_targets if the table still exists.
-- soma_targets / soma_models were removed in later migrations; this
-- migration is a no-op on databases that never had them.
ALTER TABLE IF EXISTS soma_targets DROP COLUMN IF EXISTS state_bcs;

-- Drop embedding/decryption columns from soma_models if present.
DROP INDEX IF EXISTS soma_models_has_embedding;
ALTER TABLE IF EXISTS soma_models
    DROP COLUMN IF EXISTS has_embedding,
    DROP COLUMN IF EXISTS embedding_commitment,
    DROP COLUMN IF EXISTS decryption_key_commitment,
    DROP COLUMN IF EXISTS decryption_key,
    DROP COLUMN IF EXISTS pending_embedding_commitment,
    DROP COLUMN IF EXISTS pending_decryption_key_commitment;

-- Drop soma_target_models table (redundant with model_ids_json on soma_targets).
DROP TABLE IF EXISTS soma_target_models;
