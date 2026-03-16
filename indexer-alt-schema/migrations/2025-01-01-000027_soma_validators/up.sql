CREATE TABLE IF NOT EXISTS soma_validators
(
    address                     BYTEA        NOT NULL,
    epoch                       BIGINT       NOT NULL,
    voting_power                BIGINT       NOT NULL,
    commission_rate             BIGINT       NOT NULL,
    next_epoch_commission_rate  BIGINT       NOT NULL,
    staking_pool_id             BYTEA        NOT NULL,
    stake                       BIGINT       NOT NULL,
    pending_stake               BIGINT       NOT NULL,
    name                        TEXT,
    network_address             TEXT,
    proxy_address               TEXT,
    protocol_pubkey             BYTEA,
    PRIMARY KEY (address, epoch)
);

CREATE INDEX IF NOT EXISTS soma_validators_epoch
ON soma_validators (epoch);

CREATE INDEX IF NOT EXISTS soma_validators_stake
ON soma_validators (epoch, stake DESC);
