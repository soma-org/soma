-- Drop state_bcs from soma_targets (all fields already denormalized).
ALTER TABLE soma_targets DROP COLUMN state_bcs;

-- Drop embedding/decryption columns from soma_models (not indexed).
DROP INDEX IF EXISTS soma_models_has_embedding;
ALTER TABLE soma_models
    DROP COLUMN has_embedding,
    DROP COLUMN embedding_commitment,
    DROP COLUMN decryption_key_commitment,
    DROP COLUMN decryption_key,
    DROP COLUMN pending_embedding_commitment,
    DROP COLUMN pending_decryption_key_commitment;

-- Drop soma_target_models table (redundant with model_ids_json on soma_targets).
DROP TABLE IF EXISTS soma_target_models;
