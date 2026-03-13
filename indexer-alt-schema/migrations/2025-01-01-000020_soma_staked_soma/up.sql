CREATE TABLE soma_staked_soma (
    staked_soma_id     BYTEA  NOT NULL,
    cp_sequence_number BIGINT NOT NULL,
    owner              BYTEA,
    pool_id            BYTEA,
    stake_activation_epoch BIGINT,
    principal          BIGINT,
    PRIMARY KEY (staked_soma_id, cp_sequence_number)
);

CREATE INDEX soma_staked_soma_owner ON soma_staked_soma (owner, cp_sequence_number DESC)
    WHERE owner IS NOT NULL;
CREATE INDEX soma_staked_soma_pool ON soma_staked_soma (pool_id, cp_sequence_number DESC)
    WHERE pool_id IS NOT NULL;
