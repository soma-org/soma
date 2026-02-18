//! Library-level integration tests for CLI commands.
//!
//! These tests import CLI command types directly and verify typed return values.
//! Response display/serialization tests run without a network.
//!
//! Run: cargo test -p cli --test cli_integration_tests

use cli::response::{BalanceOutput, ClientCommandResponse};

// =============================================================================
// Response formatting tests (no network needed)
// =============================================================================

#[test]
fn test_balance_output_display() {
    let output = BalanceOutput {
        address: types::base::SomaAddress::ZERO,
        total_balance: 5_000_000_000,
        coin_count: 1,
        coins: None,
    };

    let display = format!("{}", output);
    assert!(display.contains("SOMA"), "Balance output should contain SOMA: {display}");
    assert!(
        display.contains(&types::base::SomaAddress::ZERO.to_string()[..10]),
        "Balance output should contain address: {display}"
    );
}

#[test]
fn test_balance_output_json() {
    let output = BalanceOutput {
        address: types::base::SomaAddress::ZERO,
        total_balance: 5_000_000_000,
        coin_count: 1,
        coins: None,
    };

    let json = serde_json::to_string(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed["total_balance"], 5_000_000_000u64);
    assert_eq!(parsed["coin_count"], 1);
}

#[test]
fn test_transaction_response_display_success() {
    use cli::response::{OwnedObjectRef, OwnerDisplay, TransactionResponse, TransactionStatus};

    let response = TransactionResponse {
        digest: types::digests::TransactionDigest::default(),
        status: TransactionStatus::Success,
        executed_epoch: 1,
        checkpoint: Some(100),
        fee: types::tx_fee::TransactionFee {
            base_fee: 1000,
            operation_fee: 500,
            value_fee: 0,
            total_fee: 1500,
        },
        created: vec![],
        mutated: vec![],
        deleted: vec![],
        gas_object: OwnedObjectRef {
            object_id: types::object::ObjectID::ZERO,
            version: types::object::Version::from_u64(1),
            digest: types::digests::ObjectDigest::new([0; 32]),
            owner: OwnerDisplay::AddressOwner { address: types::base::SomaAddress::ZERO },
        },
        balance_changes: vec![],
    };

    let display = format!("{}", response);
    assert!(display.contains("Succeeded"), "Should show success: {display}");
    assert!(display.contains("Gas Summary"), "Should show gas summary: {display}");
}

#[test]
fn test_transaction_response_display_failure() {
    use cli::response::{OwnedObjectRef, OwnerDisplay, TransactionResponse, TransactionStatus};

    let response = TransactionResponse {
        digest: types::digests::TransactionDigest::default(),
        status: TransactionStatus::Failure { error: "InsufficientGas".to_string() },
        executed_epoch: 1,
        checkpoint: None,
        fee: types::tx_fee::TransactionFee {
            base_fee: 0,
            operation_fee: 0,
            value_fee: 0,
            total_fee: 0,
        },
        created: vec![],
        mutated: vec![],
        deleted: vec![],
        gas_object: OwnedObjectRef {
            object_id: types::object::ObjectID::ZERO,
            version: types::object::Version::from_u64(1),
            digest: types::digests::ObjectDigest::new([0; 32]),
            owner: OwnerDisplay::AddressOwner { address: types::base::SomaAddress::ZERO },
        },
        balance_changes: vec![],
    };

    let display = format!("{}", response);
    assert!(display.contains("Failed"), "Should show failure: {display}");
    assert!(display.contains("InsufficientGas"), "Should show error: {display}");
}

// =============================================================================
// Address/Environment output tests
// =============================================================================

#[test]
fn test_addresses_output_display() {
    use cli::response::AddressesOutput;

    let output = AddressesOutput {
        active_address: types::base::SomaAddress::ZERO,
        addresses: vec![
            ("alice".to_string(), types::base::SomaAddress::ZERO),
            ("bob".to_string(), types::base::SomaAddress::random()),
        ],
    };

    let display = format!("{}", output);
    assert!(display.contains("alice"), "Should contain alias: {display}");
    assert!(display.contains("bob"), "Should contain alias: {display}");
}

#[test]
fn test_envs_output_display() {
    use cli::response::EnvsOutput;
    use sdk::client_config::SomaEnv;

    let output = EnvsOutput {
        envs: vec![
            SomaEnv {
                alias: "localnet".to_string(),
                rpc: "http://127.0.0.1:9000".to_string(),
                basic_auth: None,
                chain_id: None,
            },
            SomaEnv {
                alias: "testnet".to_string(),
                rpc: "https://fullnode.testnet.soma.org:443".to_string(),
                basic_auth: None,
                chain_id: None,
            },
        ],
        active: Some("localnet".to_string()),
    };

    let display = format!("{}", output);
    assert!(display.contains("localnet"), "Should contain localnet: {display}");
    assert!(display.contains("testnet"), "Should contain testnet: {display}");
}

#[test]
fn test_envs_output_empty() {
    use cli::response::EnvsOutput;

    let output = EnvsOutput { envs: vec![], active: None };
    let display = format!("{}", output);
    assert!(
        display.contains("No environments"),
        "Empty envs should say no environments: {display}"
    );
}

// =============================================================================
// Model response tests
// =============================================================================

#[test]
fn test_model_list_empty() {
    use cli::commands::model::ModelListOutput;

    let output = ModelListOutput { models: vec![] };
    let display = format!("{}", output);
    assert!(display.contains("No models"), "Empty list should say no models: {display}");
}

#[test]
fn test_model_info_display() {
    use cli::commands::model::{ModelInfoOutput, ModelStatus, ModelSummary};

    let output = ModelInfoOutput {
        model_id: types::object::ObjectID::ZERO,
        status: ModelStatus::Active,
        summary: ModelSummary {
            model_id: types::object::ObjectID::ZERO,
            owner: types::base::SomaAddress::ZERO,
            status: ModelStatus::Active,
            architecture_version: 1,
            commission_rate: 500,
            commit_epoch: 10,
            stake_balance: 1_000_000_000,
            has_pending_update: false,
        },
    };

    let display = format!("{}", output);
    assert!(display.contains("Model Information"), "Should contain header: {display}");
    assert!(display.contains("Active"), "Should show active status: {display}");
    assert!(display.contains("5.00%"), "Should show commission rate: {display}");
}

#[test]
fn test_model_status_display() {
    use cli::commands::model::ModelStatus;

    let active = format!("{}", ModelStatus::Active);
    let pending = format!("{}", ModelStatus::Pending);
    let inactive = format!("{}", ModelStatus::Inactive);

    assert!(active.contains("Active"));
    assert!(pending.contains("Pending"));
    assert!(inactive.contains("Inactive"));
}

// =============================================================================
// Format helpers tests
// =============================================================================

#[test]
fn test_format_soma_public() {
    assert_eq!(cli::response::format_soma_public(0), "0 SOMA");
    assert_eq!(cli::response::format_soma_public(1_000_000_000), "1 SOMA");
    assert_eq!(cli::response::format_soma_public(1_500_000_000), "1.50 SOMA");
    assert_eq!(cli::response::format_soma_public(1_000_000_000_000), "1.00K SOMA");
    assert_eq!(cli::response::format_soma_public(500_000_000), "0.5 SOMA");
}

// =============================================================================
// JSON serialization round-trip tests
// =============================================================================

#[test]
fn test_balance_json_roundtrip() {
    let output = BalanceOutput {
        address: types::base::SomaAddress::ZERO,
        total_balance: 42_000_000_000,
        coin_count: 3,
        coins: Some(vec![
            (types::object::ObjectID::ZERO, 20_000_000_000),
            (types::object::ObjectID::random(), 22_000_000_000),
        ]),
    };

    let json = serde_json::to_string_pretty(&output).unwrap();
    let _: serde_json::Value = serde_json::from_str(&json).unwrap();
}

#[test]
fn test_client_command_response_no_output() {
    let no_output = ClientCommandResponse::NoOutput;
    assert!(matches!(no_output, ClientCommandResponse::NoOutput));
}

// =============================================================================
// Validator output tests
// =============================================================================

#[test]
fn test_validator_summary_display() {
    use cli::response::{ValidatorStatus, ValidatorSummary};

    let summary = ValidatorSummary {
        address: types::base::SomaAddress::ZERO,
        status: ValidatorStatus::Active,
        voting_power: 10000,
        commission_rate: 200,
        network_address: "/dns/localhost/tcp/8080".to_string(),
        p2p_address: "/dns/localhost/tcp/8081".to_string(),
        primary_address: "/dns/localhost/tcp/8082".to_string(),
        protocol_pubkey: "abcdef1234567890abcdef1234567890".to_string(),
        network_pubkey: "1234567890abcdef1234567890abcdef".to_string(),
        worker_pubkey: "fedcba0987654321fedcba0987654321".to_string(),
    };

    let display = format!("{}", summary);
    assert!(display.contains("Validator Information"), "Should contain header: {display}");
    assert!(display.contains("Active"), "Should show active status: {display}");
    assert!(display.contains("2.00%"), "Should show commission rate: {display}");
}

// =============================================================================
// New address output tests
// =============================================================================

#[test]
fn test_new_address_output_display() {
    use cli::response::NewAddressOutput;
    use types::crypto::SignatureScheme;

    let output = NewAddressOutput {
        alias: "my-wallet".to_string(),
        address: types::base::SomaAddress::ZERO,
        key_scheme: SignatureScheme::ED25519,
        recovery_phrase: "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12".to_string(),
    };

    let display = format!("{}", output);
    assert!(display.contains("New Address Created"), "Should show creation success: {display}");
    assert!(display.contains("my-wallet"), "Should show alias: {display}");
    assert!(
        display.contains("recovery phrase"),
        "Should warn about recovery phrase: {display}"
    );
}
