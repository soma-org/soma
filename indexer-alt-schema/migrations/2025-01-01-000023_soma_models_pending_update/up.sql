ALTER TABLE soma_models
    ADD COLUMN pending_manifest_url              TEXT,
    ADD COLUMN pending_manifest_checksum         BYTEA,
    ADD COLUMN pending_manifest_size             BIGINT,
    ADD COLUMN pending_weights_commitment        BYTEA,
    ADD COLUMN pending_embedding_commitment      BYTEA,
    ADD COLUMN pending_decryption_key_commitment BYTEA,
    ADD COLUMN pending_commit_epoch              BIGINT;
