-- Restore soma_target_models table.
CREATE TABLE IF NOT EXISTS soma_target_models
(
    target_id           BYTEA   NOT NULL,
    cp_sequence_number  BIGINT  NOT NULL,
    model_id            BYTEA   NOT NULL,
    PRIMARY KEY (target_id, cp_sequence_number, model_id)
);
CREATE INDEX IF NOT EXISTS soma_target_models_model_id
    ON soma_target_models (model_id, cp_sequence_number DESC);

-- Restore embedding/decryption columns on soma_models.
ALTER TABLE soma_models
    ADD COLUMN has_embedding                   BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN embedding_commitment            BYTEA,
    ADD COLUMN decryption_key_commitment       BYTEA,
    ADD COLUMN decryption_key                  BYTEA,
    ADD COLUMN pending_embedding_commitment    BYTEA,
    ADD COLUMN pending_decryption_key_commitment BYTEA;
ALTER TABLE soma_models ALTER COLUMN has_embedding DROP DEFAULT;
CREATE INDEX IF NOT EXISTS soma_models_has_embedding
    ON soma_models (model_id) WHERE has_embedding = TRUE;

-- Restore state_bcs on soma_targets.
ALTER TABLE soma_targets ADD COLUMN state_bcs BYTEA NOT NULL DEFAULT '\x';
ALTER TABLE soma_targets ALTER COLUMN state_bcs DROP DEFAULT;
