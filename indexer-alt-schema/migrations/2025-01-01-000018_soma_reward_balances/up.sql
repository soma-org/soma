-- Denormalize reward balance changes into per-recipient rows.

CREATE TABLE IF NOT EXISTS soma_reward_balances
(
    target_id           BYTEA   NOT NULL,
    cp_sequence_number  BIGINT  NOT NULL,
    epoch               BIGINT  NOT NULL,
    tx_digest           BYTEA   NOT NULL,
    recipient           BYTEA   NOT NULL,
    amount              BIGINT  NOT NULL,
    PRIMARY KEY (target_id, recipient)
);

CREATE INDEX IF NOT EXISTS soma_reward_balances_epoch
    ON soma_reward_balances (epoch);

CREATE INDEX IF NOT EXISTS soma_reward_balances_recipient
    ON soma_reward_balances (recipient, epoch);

-- Drop the opaque BCS blob from soma_rewards
ALTER TABLE soma_rewards DROP COLUMN balance_changes_bcs;
