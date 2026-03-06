// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

// @generated automatically by Diesel CLI.

diesel::table! {
    coin_balance_buckets (object_id, cp_sequence_number) {
        object_id -> Bytea,
        cp_sequence_number -> Int8,
        owner_kind -> Nullable<Int2>,
        owner_id -> Nullable<Bytea>,
        coin_type -> Nullable<Bytea>,
        coin_balance_bucket -> Nullable<Int2>,
    }
}

diesel::table! {
    coin_balance_buckets_deletion_reference (cp_sequence_number, object_id) {
        object_id -> Bytea,
        cp_sequence_number -> Int8,
    }
}

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
    coin_balance_buckets,
    coin_balance_buckets_deletion_reference,
    cp_sequence_numbers,
    kv_checkpoints,
    kv_epoch_ends,
    kv_epoch_starts,
    kv_objects,
    kv_transactions,
    obj_info,
    obj_info_deletion_reference,
    obj_versions,
    tx_affected_addresses,
    tx_affected_objects,
    tx_balance_changes,
    tx_calls,
    tx_digests,
    tx_kinds,
    watermarks,
);
