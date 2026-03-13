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
        distance_threshold -> Float8,
        targets_generated_this_epoch -> Int8,
        hits_this_epoch -> Int8,
        hits_ema -> Int8,
        reward_per_target -> Int8,
        safe_mode -> Bool,
        safe_mode_accumulated_fees -> Int8,
        safe_mode_accumulated_emissions -> Int8,
    }
}

diesel::table! {
    soma_models (model_id, epoch) {
        model_id -> Bytea,
        epoch -> Int8,
        status -> Text,
        owner -> Bytea,
        architecture_version -> Int8,
        commit_epoch -> Int8,
        stake -> Int8,
        commission_rate -> Int8,
        has_embedding -> Bool,
        next_epoch_commission_rate -> Int8,
        staking_pool_id -> Bytea,
        activation_epoch -> Nullable<Int8>,
        deactivation_epoch -> Nullable<Int8>,
        rewards_pool -> Int8,
        pool_token_balance -> Int8,
        pending_stake -> Int8,
        pending_total_soma_withdraw -> Int8,
        pending_pool_token_withdraw -> Int8,
        exchange_rates_json -> Text,
        manifest_url -> Nullable<Text>,
        manifest_checksum -> Nullable<Bytea>,
        manifest_size -> Nullable<Int8>,
        weights_commitment -> Nullable<Bytea>,
        embedding_commitment -> Nullable<Bytea>,
        decryption_key_commitment -> Nullable<Bytea>,
        decryption_key -> Nullable<Bytea>,
        has_pending_update -> Bool,
        pending_manifest_url -> Nullable<Text>,
        pending_manifest_checksum -> Nullable<Bytea>,
        pending_manifest_size -> Nullable<Int8>,
        pending_weights_commitment -> Nullable<Bytea>,
        pending_embedding_commitment -> Nullable<Bytea>,
        pending_decryption_key_commitment -> Nullable<Bytea>,
        pending_commit_epoch -> Nullable<Int8>,
    }
}

diesel::table! {
    soma_reward_balances (target_id, recipient) {
        target_id -> Bytea,
        cp_sequence_number -> Int8,
        epoch -> Int8,
        tx_digest -> Bytea,
        recipient -> Bytea,
        amount -> Int8,
    }
}

diesel::table! {
    soma_rewards (target_id) {
        target_id -> Bytea,
        cp_sequence_number -> Int8,
        epoch -> Int8,
        tx_digest -> Bytea,
    }
}

diesel::table! {
    soma_target_models (target_id, cp_sequence_number, model_id) {
        target_id -> Bytea,
        cp_sequence_number -> Int8,
        model_id -> Bytea,
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
    soma_target_reports (target_id, cp_sequence_number, reporter) {
        target_id -> Bytea,
        cp_sequence_number -> Int8,
        reporter -> Bytea,
    }
}

diesel::table! {
    soma_targets (target_id, cp_sequence_number) {
        target_id -> Bytea,
        cp_sequence_number -> Int8,
        epoch -> Int8,
        status -> Text,
        submitter -> Nullable<Bytea>,
        winning_model_id -> Nullable<Bytea>,
        reward_pool -> Int8,
        bond_amount -> Int8,
        report_count -> Int4,
        state_bcs -> Bytea,
        winning_distance_score -> Nullable<Float8>,
        winning_loss_score -> Nullable<Float8>,
        winning_model_owner -> Nullable<Bytea>,
        fill_epoch -> Nullable<Int8>,
        distance_threshold -> Float8,
        model_ids_json -> Text,
        winning_data_url -> Nullable<Text>,
        winning_data_checksum -> Nullable<Bytea>,
        winning_data_size -> Nullable<Int8>,
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
    soma_epoch_state,
    soma_models,
    soma_reward_balances,
    soma_rewards,
    soma_staked_soma,
    soma_target_models,
    soma_target_reports,
    soma_targets,
    tx_affected_addresses,
    tx_affected_objects,
    tx_balance_changes,
    tx_calls,
    tx_digests,
    tx_kinds,
    watermarks,
);
