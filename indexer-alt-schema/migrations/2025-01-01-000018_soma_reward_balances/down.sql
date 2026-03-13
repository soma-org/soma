DROP TABLE IF EXISTS soma_reward_balances;

ALTER TABLE soma_rewards ADD COLUMN balance_changes_bcs BYTEA NOT NULL DEFAULT '\x';
