// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

// @generated automatically by Diesel CLI.

// Stage 13i: coin_balance_buckets and coin_balance_buckets_deletion_reference
// tables removed. The accumulator is the sole source of truth for fungible
// balances; the indexer no longer tracks Coin objects.

diesel::table! {
    cp_sequence_numbers (cp_sequence_number) {
        cp_sequence_number -> Int8,
        tx_lo -> Int8,
        epoch -> Int8,
    }
}

diesel::table! {
    kv_checkpoints (sequence_number) {
        sequence_number -> Int8,
        checkpoint_contents -> Bytea,
        checkpoint_summary -> Bytea,
        validator_signatures -> Bytea,
    }
}

diesel::table! {
    kv_epoch_ends (epoch) {
        epoch -> Int8,
        cp_hi -> Int8,
        tx_hi -> Int8,
        end_timestamp_ms -> Int8,
        safe_mode -> Bool,
        total_stake -> Nullable<Int8>,
        storage_fund_balance -> Nullable<Int8>,
        storage_fund_reinvestment -> Nullable<Int8>,
        storage_charge -> Nullable<Int8>,
        storage_rebate -> Nullable<Int8>,
        stake_subsidy_amount -> Nullable<Int8>,
        total_gas_fees -> Nullable<Int8>,
        total_stake_rewards_distributed -> Nullable<Int8>,
        leftover_storage_fund_inflow -> Nullable<Int8>,
        epoch_commitments -> Bytea,
    }
}

diesel::table! {
    kv_epoch_starts (epoch) {
        epoch -> Int8,
        protocol_version -> Int8,
        cp_lo -> Int8,
        start_timestamp_ms -> Int8,
        reference_gas_price -> Int8,
        system_state -> Bytea,
    }
}

diesel::table! {
    kv_objects (object_id, object_version) {
        object_id -> Bytea,
        object_version -> Int8,
        serialized_object -> Nullable<Bytea>,
        cp_sequence_number -> Int8,
    }
}

diesel::table! {
    kv_transactions (tx_digest) {
        tx_digest -> Bytea,
        cp_sequence_number -> Int8,
        timestamp_ms -> Int8,
        raw_transaction -> Bytea,
        raw_effects -> Bytea,
        events -> Bytea,
        user_signatures -> Bytea,
    }
}

diesel::table! {
    obj_info (object_id, cp_sequence_number) {
        object_id -> Bytea,
        cp_sequence_number -> Int8,
        owner_kind -> Nullable<Int2>,
        owner_id -> Nullable<Bytea>,
        package -> Nullable<Bytea>,
        module -> Nullable<Text>,
        name -> Nullable<Text>,
        instantiation -> Nullable<Bytea>,
    }
}

diesel::table! {
    obj_info_deletion_reference (cp_sequence_number, object_id) {
        object_id -> Bytea,
        cp_sequence_number -> Int8,
    }
}

diesel::table! {
    obj_versions (object_id, object_version) {
        object_id -> Bytea,
        object_version -> Int8,
        object_digest -> Nullable<Bytea>,
        cp_sequence_number -> Int8,
    }
}

diesel::table! {
    tx_affected_addresses (affected, tx_sequence_number) {
        affected -> Bytea,
        tx_sequence_number -> Int8,
        sender -> Bytea,
    }
}

diesel::table! {
    tx_affected_objects (affected, tx_sequence_number) {
        tx_sequence_number -> Int8,
        affected -> Bytea,
        sender -> Bytea,
    }
}

diesel::table! {
    tx_balance_changes (tx_sequence_number) {
        tx_sequence_number -> Int8,
        balance_changes -> Bytea,
    }
}

diesel::table! {
    tx_calls (package, module, function, tx_sequence_number) {
        package -> Bytea,
        module -> Text,
        function -> Text,
        tx_sequence_number -> Int8,
        sender -> Bytea,
    }
}

diesel::table! {
    tx_digests (tx_sequence_number) {
        tx_sequence_number -> Int8,
        tx_digest -> Bytea,
    }
}

diesel::table! {
    tx_kinds (tx_kind, tx_sequence_number) {
        tx_kind -> Int2,
        tx_sequence_number -> Int8,
    }
}

diesel::table! {
    soma_epoch_state (epoch) {
        epoch -> Int8,
        emission_balance -> Int8,
        emission_per_epoch -> Int8,
        distribution_counter -> Int8,
        period_length -> Int8,
        decrease_rate -> Int4,
        protocol_fund_balance -> Int8,
        safe_mode -> Bool,
        safe_mode_accumulated_fees -> Int8,
        safe_mode_accumulated_emissions -> Int8,
    }
}

diesel::table! {
    soma_staked_soma (staked_soma_id, cp_sequence_number) {
        staked_soma_id -> Bytea,
        cp_sequence_number -> Int8,
        owner -> Nullable<Bytea>,
        pool_id -> Nullable<Bytea>,
        stake_activation_epoch -> Nullable<Int8>,
        principal -> Nullable<Int8>,
    }
}

diesel::table! {
    soma_balance_deltas (owner, coin_type, cp_sequence_number) {
        owner -> Bytea,
        coin_type -> Text,
        cp_sequence_number -> Int8,
        delta -> Int8,
    }
}

diesel::table! {
    soma_validators (address, epoch) {
        address -> Bytea,
        epoch -> Int8,
        voting_power -> Int8,
        commission_rate -> Int8,
        next_epoch_commission_rate -> Int8,
        staking_pool_id -> Bytea,
        stake -> Int8,
        pending_stake -> Int8,
        name -> Nullable<Text>,
        network_address -> Nullable<Text>,
        proxy_address -> Nullable<Text>,
        protocol_pubkey -> Nullable<Bytea>,
    }
}

diesel::table! {
    soma_tx_details (tx_sequence_number) {
        tx_sequence_number -> Int8,
        tx_digest -> Bytea,
        kind -> Text,
        sender -> Bytea,
        epoch -> Int8,
        timestamp_ms -> Int8,
        metadata_json -> Nullable<Text>,
    }
}

diesel::table! {
    soma_asks (ask_id, cp_sequence_number) {
        ask_id -> Bytea,
        cp_sequence_number -> Int8,
        buyer -> Bytea,
        task_digest -> Bytea,
        max_price_per_bid -> Int8,
        num_bids_wanted -> Int4,
        timeout_ms -> Int8,
        created_at_ms -> Int8,
        status -> Text,
        accepted_bid_count -> Int4,
    }
}

diesel::table! {
    soma_bids (bid_id, cp_sequence_number) {
        bid_id -> Bytea,
        cp_sequence_number -> Int8,
        ask_id -> Bytea,
        seller -> Bytea,
        price -> Int8,
        response_digest -> Bytea,
        created_at_ms -> Int8,
        status -> Text,
    }
}

diesel::table! {
    soma_settlements (settlement_id, cp_sequence_number) {
        settlement_id -> Bytea,
        cp_sequence_number -> Int8,
        ask_id -> Bytea,
        bid_id -> Bytea,
        buyer -> Bytea,
        seller -> Bytea,
        amount -> Int8,
        task_digest -> Bytea,
        response_digest -> Bytea,
        settled_at_ms -> Int8,
        seller_rating -> Text,
        rating_deadline_ms -> Int8,
    }
}

diesel::table! {
    soma_vaults (vault_id, cp_sequence_number) {
        vault_id -> Bytea,
        cp_sequence_number -> Int8,
        owner -> Bytea,
        balance -> Int8,
    }
}

diesel::table! {
    watermarks (pipeline) {
        pipeline -> Text,
        epoch_hi_inclusive -> Int8,
        checkpoint_hi_inclusive -> Int8,
        tx_hi -> Int8,
        timestamp_ms_hi_inclusive -> Int8,
        reader_lo -> Int8,
        pruner_timestamp -> Timestamp,
        pruner_hi -> Int8,
    }
}

diesel::allow_tables_to_appear_in_same_query!(
    cp_sequence_numbers,
    kv_checkpoints,
    kv_epoch_ends,
    kv_epoch_starts,
    kv_objects,
    kv_transactions,
    obj_info,
    obj_info_deletion_reference,
    obj_versions,
    soma_asks,
    soma_balance_deltas,
    soma_bids,
    soma_epoch_state,
    soma_settlements,
    soma_staked_soma,
    soma_validators,
    soma_vaults,
    soma_tx_details,
    tx_affected_addresses,
    tx_affected_objects,
    tx_balance_changes,
    tx_calls,
    tx_digests,
    tx_kinds,
    watermarks,
);
