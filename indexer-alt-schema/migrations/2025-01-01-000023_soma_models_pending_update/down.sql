ALTER TABLE soma_models
    DROP COLUMN IF EXISTS pending_manifest_url,
    DROP COLUMN IF EXISTS pending_manifest_checksum,
    DROP COLUMN IF EXISTS pending_manifest_size,
    DROP COLUMN IF EXISTS pending_weights_commitment,
    DROP COLUMN IF EXISTS pending_embedding_commitment,
    DROP COLUMN IF EXISTS pending_decryption_key_commitment,
    DROP COLUMN IF EXISTS pending_commit_epoch;
