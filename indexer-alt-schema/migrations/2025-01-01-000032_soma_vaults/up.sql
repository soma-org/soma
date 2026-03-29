CREATE TABLE soma_vaults (
    vault_id                BYTEA   NOT NULL,
    cp_sequence_number      BIGINT  NOT NULL,
    owner                   BYTEA   NOT NULL,
    balance                 BIGINT  NOT NULL,
    PRIMARY KEY (vault_id, cp_sequence_number)
);

CREATE INDEX idx_vaults_owner ON soma_vaults (owner);
