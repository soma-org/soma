-- Copyright (c) Mysten Labs, Inc.
-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Triggers + functions
DROP TRIGGER IF EXISTS kv_epoch_starts_notify ON kv_epoch_starts;
DROP FUNCTION IF EXISTS notify_new_epoch();

DROP TRIGGER IF EXISTS cp_sequence_numbers_notify ON cp_sequence_numbers;
DROP FUNCTION IF EXISTS notify_new_checkpoint();

DROP TRIGGER IF EXISTS soma_tx_details_notify ON soma_tx_details;
DROP FUNCTION IF EXISTS notify_new_transaction();

-- Soma-specific tables
DROP TABLE IF EXISTS soma_validators;
DROP TABLE IF EXISTS soma_tx_details;
DROP TABLE IF EXISTS soma_staked_soma;
DROP TABLE IF EXISTS soma_epoch_state;
DROP TABLE IF EXISTS soma_balance_deltas;

-- Object state
DROP TABLE IF EXISTS obj_info_deletion_reference;
DROP TABLE IF EXISTS obj_info;
DROP TABLE IF EXISTS obj_versions;

-- Transaction indexes
DROP TABLE IF EXISTS tx_kinds;
DROP TABLE IF EXISTS tx_digests;
DROP TABLE IF EXISTS tx_balance_changes;
DROP TABLE IF EXISTS tx_affected_objects;
DROP TABLE IF EXISTS tx_affected_addresses;

-- KV content
DROP TABLE IF EXISTS kv_epoch_ends;
DROP TABLE IF EXISTS kv_epoch_starts;
DROP TABLE IF EXISTS kv_transactions;
DROP TABLE IF EXISTS kv_objects;
DROP TABLE IF EXISTS kv_checkpoints;

-- Core mapping
DROP TABLE IF EXISTS cp_sequence_numbers;
