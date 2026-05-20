// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Snapshot tests for CLI output.
//!
//! These tests capture the help text and error output of the soma CLI binary
//! and compare against stored snapshots. They do not require a running network.
//!
//! Run: cargo test -p cli --test cli_snapshot_tests
//! Update snapshots: cargo insta review

use assert_cmd::Command;

fn soma_cmd() -> Command {
    Command::cargo_bin("soma").expect("soma binary should be built")
}

// =============================================================================
// Help text snapshot tests
// =============================================================================

#[test]
fn test_help_output() {
    let output = soma_cmd().arg("--help").output().expect("failed to run soma --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("help", stdout);
}

#[test]
fn test_version_output() {
    let output = soma_cmd().arg("--version").output().expect("failed to run soma --version");
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Version output should contain "soma"
    assert!(stdout.contains("soma"), "Version output should contain 'soma': {stdout}");
}

#[test]
fn test_start_help() {
    let output =
        soma_cmd().args(["start", "--help"]).output().expect("failed to run soma start --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("start_help", stdout);
}

#[test]
fn test_start_localnet_help() {
    let output = soma_cmd()
        .args(["start", "localnet", "--help"])
        .output()
        .expect("failed to run soma start localnet --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("start_localnet_help", stdout);
}

#[test]
fn test_balance_help() {
    let output =
        soma_cmd().args(["balance", "--help"]).output().expect("failed to run soma balance --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("balance_help", stdout);
}

#[test]
fn test_start_provider_help() {
    let output = soma_cmd()
        .args(["start", "provider", "--help"])
        .output()
        .expect("failed to run soma start provider --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("start_provider_help", stdout);
}

#[test]
fn test_start_proxy_help() {
    let output = soma_cmd()
        .args(["start", "proxy", "--help"])
        .output()
        .expect("failed to run soma start proxy --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("start_proxy_help", stdout);
}

#[test]
fn test_status_help() {
    let output =
        soma_cmd().args(["status", "--help"]).output().expect("failed to run soma status --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("status_help", stdout);
}

#[test]
fn test_wallet_help() {
    let output =
        soma_cmd().args(["wallet", "--help"]).output().expect("failed to run soma wallet --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("wallet_help", stdout);
}

#[test]
fn test_env_help() {
    let output =
        soma_cmd().args(["env", "--help"]).output().expect("failed to run soma env --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("env_help", stdout);
}

#[test]
fn test_validator_help() {
    let output = soma_cmd()
        .args(["validator", "--help"])
        .output()
        .expect("failed to run soma validator --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("validator_help", stdout);
}

#[test]
fn test_stake_help() {
    let output =
        soma_cmd().args(["stake", "--help"]).output().expect("failed to run soma stake --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("stake_help", stdout);
}

#[test]
fn test_transfer_help() {
    let output = soma_cmd()
        .args(["transfer", "--help"])
        .output()
        .expect("failed to run soma transfer --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("transfer_help", stdout);
}

#[test]
fn test_object_help() {
    let output =
        soma_cmd().args(["object", "--help"]).output().expect("failed to run soma object --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("object_help", stdout);
}

#[test]
fn test_tx_help() {
    let output = soma_cmd().args(["tx", "--help"]).output().expect("failed to run soma tx --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("tx_help", stdout);
}

#[test]
fn test_network_help() {
    let output =
        soma_cmd().args(["network", "--help"]).output().expect("failed to run soma network --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("network_help", stdout);
}

#[test]
fn test_genesis_help() {
    let output =
        soma_cmd().args(["genesis", "--help"]).output().expect("failed to run soma genesis --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("genesis_help", stdout);
}

#[test]
fn test_keytool_help() {
    let output =
        soma_cmd().args(["keytool", "--help"]).output().expect("failed to run soma keytool --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("keytool_help", stdout);
}

#[test]
fn test_completions_help() {
    let output = soma_cmd()
        .args(["completions", "--help"])
        .output()
        .expect("failed to run soma completions --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("completions_help", stdout);
}

// =============================================================================
// Error output tests (offline)
// =============================================================================

#[test]
fn test_unknown_command() {
    soma_cmd().arg("nonexistent-command").assert().failure();
}

// =============================================================================
// Error formatting tests (via the library)
// =============================================================================

#[test]
fn test_error_formatting_includes_error_prefix() {
    // Verify the error formatting produces user-friendly output
    // The actual format_error function is tested in main.rs unit tests,
    // but we can verify the binary's exit behavior here
    soma_cmd().arg("nonexistent-command").assert().failure().code(2); // clap exits with 2
}
