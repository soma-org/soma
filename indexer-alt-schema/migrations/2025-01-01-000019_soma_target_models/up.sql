-- Join table for target <-> model relationships.
-- Enables efficient "find all targets assigned to model X" queries.

CREATE TABLE IF NOT EXISTS soma_target_models
(
    target_id           BYTEA   NOT NULL,
    cp_sequence_number  BIGINT  NOT NULL,
    model_id            BYTEA   NOT NULL,
    PRIMARY KEY (target_id, cp_sequence_number, model_id)
);

CREATE INDEX IF NOT EXISTS soma_target_models_model_id
    ON soma_target_models (model_id, cp_sequence_number DESC);
