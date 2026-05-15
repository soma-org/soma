-- Copyright (c) Mysten Labs, Inc.
-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0
--
-- Consolidated schema for the off-chain Soma indexer.
--
-- Tiers (see indexer-alt/src/lib.rs for pruning policy):
--   Tier A (KV content): kv_*. Prunable with BigTable watermark floor.
--   Tier B (Indexes):    tx_*. Prunable independently.
--   Tier B+ (Object):    obj_*. Prunable; latest version sufficient.
--   Tier C (Core/Soma):  cp_sequence_numbers, soma_*. Never pruned.

-- ---------------------------------------------------------------------------
-- Core mapping
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS cp_sequence_numbers
(
    cp_sequence_number      BIGINT       PRIMARY KEY,
    tx_lo                   BIGINT       NOT NULL,
    epoch                   BIGINT       NOT NULL
);

-- ---------------------------------------------------------------------------
-- Tier A: KV content (raw checkpoint / tx / object / epoch payloads)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kv_checkpoints
(
    sequence_number         BIGINT       PRIMARY KEY,
    checkpoint_contents     BYTEA        NOT NULL,
    checkpoint_summary      BYTEA        NOT NULL,
    validator_signatures    BYTEA        NOT NULL
);

CREATE TABLE IF NOT EXISTS kv_objects
(
    object_id               BYTEA        NOT NULL,
    object_version          BIGINT       NOT NULL,
    serialized_object       BYTEA,
    cp_sequence_number      BIGINT       NOT NULL DEFAULT 0,
    PRIMARY KEY (object_id, object_version)
);

CREATE INDEX IF NOT EXISTS kv_objects_cp_seq
    ON kv_objects (cp_sequence_number);

CREATE TABLE IF NOT EXISTS kv_transactions
(
    tx_digest               BYTEA        PRIMARY KEY,
    cp_sequence_number      BIGINT       NOT NULL,
    timestamp_ms            BIGINT       NOT NULL,
    -- BCS serialized TransactionData
    raw_transaction         BYTEA        NOT NULL,
    -- BCS serialized TransactionEffects
    raw_effects             BYTEA        NOT NULL,
    -- BCS serialized array of Events (always empty for Soma; kept for layout
    -- compatibility with the Sui-derived schema).
    events                  BYTEA        NOT NULL,
    -- BCS serialized array of user signatures
    user_signatures         BYTEA        NOT NULL
);

CREATE INDEX IF NOT EXISTS kv_transactions_cp_sequence_number
    ON kv_transactions (cp_sequence_number);

-- Information about an epoch that is available when it starts.
CREATE TABLE IF NOT EXISTS kv_epoch_starts
(
    epoch                   BIGINT       PRIMARY KEY,
    protocol_version        BIGINT       NOT NULL,
    -- Inclusive checkpoint lower bound of the epoch.
    cp_lo                   BIGINT       NOT NULL,
    start_timestamp_ms      BIGINT       NOT NULL,
    reference_gas_price     BIGINT       NOT NULL,
    -- BCS serialized SystemState.
    system_state            BYTEA        NOT NULL
);

CREATE INDEX IF NOT EXISTS kv_epoch_starts_cp_lo
    ON kv_epoch_starts (cp_lo);

-- Information about an epoch that is available when it ends. Several columns
-- are Sui-legacy storage-fund fields preserved for layout compatibility; in
-- Soma they are NULL when the epoch advancement entered safe mode (and even
-- when not, several do not apply).
CREATE TABLE IF NOT EXISTS kv_epoch_ends
(
    epoch                   BIGINT       PRIMARY KEY,
    -- Exclusive checkpoint upper bound of the epoch.
    cp_hi                   BIGINT       NOT NULL,
    -- Exclusive transaction upper bound of the epoch.
    tx_hi                   BIGINT       NOT NULL,
    end_timestamp_ms        BIGINT       NOT NULL,
    safe_mode               BOOLEAN      NOT NULL,
    total_stake             BIGINT,
    storage_fund_balance    BIGINT,
    storage_fund_reinvestment BIGINT,
    storage_charge          BIGINT,
    storage_rebate          BIGINT,
    stake_subsidy_amount    BIGINT,
    total_gas_fees          BIGINT,
    total_stake_rewards_distributed BIGINT,
    leftover_storage_fund_inflow BIGINT,
    -- BCS serialized `Vec<EpochCommitment>`.
    epoch_commitments       BYTEA        NOT NULL
);

-- ---------------------------------------------------------------------------
-- Tier B: Transaction indexes
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tx_affected_addresses
(
    affected                BYTEA        NOT NULL,
    tx_sequence_number      BIGINT       NOT NULL,
    sender                  BYTEA        NOT NULL,
    PRIMARY KEY (affected, tx_sequence_number)
);

CREATE INDEX IF NOT EXISTS tx_affected_addresses_tx_sequence_number
    ON tx_affected_addresses (tx_sequence_number);

CREATE INDEX IF NOT EXISTS tx_affected_addresses_sender
    ON tx_affected_addresses (sender, affected, tx_sequence_number);

CREATE TABLE IF NOT EXISTS tx_affected_objects
(
    tx_sequence_number      BIGINT       NOT NULL,
    affected                BYTEA        NOT NULL,
    sender                  BYTEA        NOT NULL,
    PRIMARY KEY (affected, tx_sequence_number)
);

CREATE INDEX IF NOT EXISTS tx_affected_objects_tx_sequence_number
    ON tx_affected_objects (tx_sequence_number);

CREATE INDEX IF NOT EXISTS tx_affected_objects_sender
    ON tx_affected_objects (sender, affected, tx_sequence_number);

CREATE TABLE IF NOT EXISTS tx_balance_changes
(
    tx_sequence_number      BIGINT       PRIMARY KEY,
    -- BCS serialized array of BalanceChanges
    balance_changes         BYTEA        NOT NULL
);

CREATE TABLE IF NOT EXISTS tx_digests
(
    tx_sequence_number      BIGINT       PRIMARY KEY,
    tx_digest               BYTEA        NOT NULL
);

CREATE INDEX IF NOT EXISTS tx_digests_tx_digest
    ON tx_digests (tx_digest);

CREATE TABLE IF NOT EXISTS tx_kinds
(
    tx_kind                 SMALLINT     NOT NULL,
    tx_sequence_number      BIGINT       NOT NULL,
    PRIMARY KEY (tx_kind, tx_sequence_number)
);

CREATE INDEX IF NOT EXISTS tx_kinds_tx_sequence_number
    ON tx_kinds (tx_sequence_number);

-- ---------------------------------------------------------------------------
-- Tier B+: Object state
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS obj_versions
(
    object_id               BYTEA        NOT NULL,
    object_version          BIGINT       NOT NULL,
    object_digest           BYTEA,
    cp_sequence_number      BIGINT       NOT NULL,
    PRIMARY KEY (object_id, object_version)
);

CREATE INDEX IF NOT EXISTS obj_versions_cp_sequence_number
    ON obj_versions (cp_sequence_number);

CREATE INDEX IF NOT EXISTS obj_versions_id_cp_version
    ON obj_versions (object_id, cp_sequence_number DESC, object_version DESC);

CREATE TABLE IF NOT EXISTS obj_info
(
    object_id               BYTEA        NOT NULL,
    cp_sequence_number      BIGINT       NOT NULL,
    owner_kind              SMALLINT,
    owner_id                BYTEA,
    package                 BYTEA,
    module                  TEXT,
    name                    TEXT,
    instantiation           BYTEA,
    PRIMARY KEY (object_id, cp_sequence_number)
);

CREATE INDEX IF NOT EXISTS obj_info_owner_object_id_desc
    ON obj_info (owner_kind, owner_id, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_pkg_object_id_desc
    ON obj_info (package, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_mod_object_id_desc
    ON obj_info (package, module, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_name_object_id_desc
    ON obj_info (package, module, name, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_inst_object_id_desc
    ON obj_info (package, module, name, instantiation, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_owner_pkg_object_id_desc
    ON obj_info (owner_kind, owner_id, package, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_owner_mod_object_id_desc
    ON obj_info (owner_kind, owner_id, package, module, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_owner_name_object_id_desc
    ON obj_info (owner_kind, owner_id, package, module, name, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_owner_inst_object_id_desc
    ON obj_info (owner_kind, owner_id, package, module, name, instantiation, cp_sequence_number DESC, object_id DESC);

CREATE TABLE IF NOT EXISTS obj_info_deletion_reference
(
    object_id               BYTEA        NOT NULL,
    cp_sequence_number      BIGINT       NOT NULL,
    PRIMARY KEY (cp_sequence_number, object_id)
);

-- ---------------------------------------------------------------------------
-- Tier C: Soma-specific tables (never pruned)
-- ---------------------------------------------------------------------------

-- Per-checkpoint per-(owner, coin_type) signed balance delta. Sourced from
-- `TransactionEffects.balance_events` by the soma_balance_deltas indexer
-- handler. Current balance is `SUM(delta) WHERE owner = ? AND coin_type = ?`.
CREATE TABLE IF NOT EXISTS soma_balance_deltas
(
    owner                   BYTEA        NOT NULL,
    coin_type               TEXT         NOT NULL,
    cp_sequence_number      BIGINT       NOT NULL,
    -- Signed; negative = net withdraw, positive = net deposit. BIGINT (i64)
    -- is the safe domain type — total supply fits well under 2^63.
    delta                   BIGINT       NOT NULL,
    PRIMARY KEY (owner, coin_type, cp_sequence_number)
);

CREATE INDEX IF NOT EXISTS soma_balance_deltas_owner
    ON soma_balance_deltas (owner, coin_type);

CREATE TABLE IF NOT EXISTS soma_epoch_state
(
    epoch                           BIGINT PRIMARY KEY,
    emission_balance                BIGINT NOT NULL,
    emission_per_epoch              BIGINT NOT NULL,
    distribution_counter            BIGINT NOT NULL,
    period_length                   BIGINT NOT NULL,
    decrease_rate                   INT    NOT NULL,
    protocol_fund_balance           BIGINT NOT NULL,
    safe_mode                       BOOL   NOT NULL,
    safe_mode_accumulated_fees      BIGINT NOT NULL,
    safe_mode_accumulated_emissions BIGINT NOT NULL
);

-- Stage 9d-C5: StakedSomaV1 was deleted as the on-chain stake record (F1
-- delegation rows replaced it). The handler that fed this table is gone;
-- legacy rows from before that change are retained for historical queries.
CREATE TABLE IF NOT EXISTS soma_staked_soma
(
    staked_soma_id          BYTEA        NOT NULL,
    cp_sequence_number      BIGINT       NOT NULL,
    owner                   BYTEA,
    pool_id                 BYTEA,
    stake_activation_epoch  BIGINT,
    principal               BIGINT,
    PRIMARY KEY (staked_soma_id, cp_sequence_number)
);

CREATE INDEX IF NOT EXISTS soma_staked_soma_owner
    ON soma_staked_soma (owner, cp_sequence_number DESC)
    WHERE owner IS NOT NULL;

CREATE INDEX IF NOT EXISTS soma_staked_soma_pool
    ON soma_staked_soma (pool_id, cp_sequence_number DESC)
    WHERE pool_id IS NOT NULL;

-- Detailed transaction information: kind label, sender, epoch, and optional
-- kind-specific metadata as JSON. Enables labeled tx feeds and kind-filtered
-- queries.
CREATE TABLE IF NOT EXISTS soma_tx_details
(
    tx_sequence_number      BIGINT       NOT NULL PRIMARY KEY,
    tx_digest               BYTEA        NOT NULL,
    kind                    TEXT         NOT NULL,
    sender                  BYTEA        NOT NULL,
    epoch                   BIGINT       NOT NULL,
    timestamp_ms            BIGINT       NOT NULL,
    metadata_json           TEXT
);

CREATE INDEX IF NOT EXISTS soma_tx_details_kind
    ON soma_tx_details (kind);

CREATE INDEX IF NOT EXISTS soma_tx_details_sender
    ON soma_tx_details (sender);

CREATE INDEX IF NOT EXISTS soma_tx_details_epoch
    ON soma_tx_details (epoch);

CREATE TABLE IF NOT EXISTS soma_validators
(
    address                     BYTEA    NOT NULL,
    epoch                       BIGINT   NOT NULL,
    voting_power                BIGINT   NOT NULL,
    commission_rate             BIGINT   NOT NULL,
    next_epoch_commission_rate  BIGINT   NOT NULL,
    staking_pool_id             BYTEA    NOT NULL,
    stake                       BIGINT   NOT NULL,
    pending_stake               BIGINT   NOT NULL,
    name                        TEXT,
    network_address             TEXT,
    protocol_pubkey             BYTEA,
    PRIMARY KEY (address, epoch)
);

CREATE INDEX IF NOT EXISTS soma_validators_epoch
    ON soma_validators (epoch);

CREATE INDEX IF NOT EXISTS soma_validators_stake
    ON soma_validators (epoch, stake DESC);

-- ---------------------------------------------------------------------------
-- NOTIFY triggers for GraphQL subscriptions
-- ---------------------------------------------------------------------------
-- The GraphQL server LISTENs on these channels and broadcasts to WebSocket
-- clients (see soma-graphql/src/subscriptions.rs).

CREATE OR REPLACE FUNCTION notify_new_transaction() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('new_transaction', json_build_object(
        'tx_sequence_number', NEW.tx_sequence_number,
        'kind', NEW.kind,
        'sender', encode(NEW.sender, 'hex'),
        'epoch', NEW.epoch,
        'timestamp_ms', NEW.timestamp_ms
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER soma_tx_details_notify
    AFTER INSERT ON soma_tx_details
    FOR EACH ROW EXECUTE FUNCTION notify_new_transaction();

CREATE OR REPLACE FUNCTION notify_new_checkpoint() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('new_checkpoint', json_build_object(
        'cp_sequence_number', NEW.cp_sequence_number,
        'tx_lo', NEW.tx_lo,
        'epoch', NEW.epoch
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER cp_sequence_numbers_notify
    AFTER INSERT ON cp_sequence_numbers
    FOR EACH ROW EXECUTE FUNCTION notify_new_checkpoint();

CREATE OR REPLACE FUNCTION notify_new_epoch() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('new_epoch', json_build_object(
        'epoch', NEW.epoch,
        'start_timestamp_ms', NEW.start_timestamp_ms,
        'protocol_version', NEW.protocol_version
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER kv_epoch_starts_notify
    AFTER INSERT ON kv_epoch_starts
    FOR EACH ROW EXECUTE FUNCTION notify_new_epoch();
