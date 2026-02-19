//! Snapshot tests for CLI output.
//!
//! These tests capture the help text and error output of the soma CLI binary
//! and compare against stored snapshots. They do not require a running network.
//!
//! Run: cargo test -p cli --test cli_snapshot_tests
//! Update snapshots: cargo insta review

use assert_cmd::Command;
use predicates::prelude::*;

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
    let output = soma_cmd()
        .args(["start", "--help"])
        .output()
        .expect("failed to run soma start --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("start_help", stdout);
}

#[test]
fn test_balance_help() {
    let output = soma_cmd()
        .args(["balance", "--help"])
        .output()
        .expect("failed to run soma balance --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("balance_help", stdout);
}

#[test]
fn test_send_help() {
    let output =
        soma_cmd().args(["send", "--help"]).output().expect("failed to run soma send --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("send_help", stdout);
}

#[test]
fn test_model_help() {
    let output = soma_cmd()
        .args(["model", "--help"])
        .output()
        .expect("failed to run soma model --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("model_help", stdout);
}

#[test]
fn test_model_commit_help() {
    let output = soma_cmd()
        .args(["model", "commit", "--help"])
        .output()
        .expect("failed to run soma model commit --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("model_commit_help", stdout);
}

#[test]
fn test_claim_help() {
    let output = soma_cmd()
        .args(["claim", "--help"])
        .output()
        .expect("failed to run soma claim --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("claim_help", stdout);
}

#[test]
fn test_status_help() {
    let output = soma_cmd()
        .args(["status", "--help"])
        .output()
        .expect("failed to run soma status --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("status_help", stdout);
}

#[test]
fn test_wallet_help() {
    let output = soma_cmd()
        .args(["wallet", "--help"])
        .output()
        .expect("failed to run soma wallet --help");
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
    let output = soma_cmd()
        .args(["stake", "--help"])
        .output()
        .expect("failed to run soma stake --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("stake_help", stdout);
}

#[test]
fn test_pay_help() {
    let output =
        soma_cmd().args(["pay", "--help"]).output().expect("failed to run soma pay --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("pay_help", stdout);
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
fn test_unstake_help() {
    let output = soma_cmd()
        .args(["unstake", "--help"])
        .output()
        .expect("failed to run soma unstake --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("unstake_help", stdout);
}

#[test]
fn test_objects_help() {
    let output = soma_cmd()
        .args(["objects", "--help"])
        .output()
        .expect("failed to run soma objects --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("objects_help", stdout);
}

#[test]
fn test_tx_help() {
    let output =
        soma_cmd().args(["tx", "--help"]).output().expect("failed to run soma tx --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("tx_help", stdout);
}

#[test]
fn test_data_help() {
    let output =
        soma_cmd().args(["data", "--help"]).output().expect("failed to run soma data --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("data_help", stdout);
}

#[test]
fn test_submit_help() {
    let output = soma_cmd()
        .args(["submit", "--help"])
        .output()
        .expect("failed to run soma submit --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("submit_help", stdout);
}

#[test]
fn test_challenge_help() {
    let output = soma_cmd()
        .args(["challenge", "--help"])
        .output()
        .expect("failed to run soma challenge --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("challenge_help", stdout);
}

#[test]
fn test_target_help() {
    let output = soma_cmd()
        .args(["target", "--help"])
        .output()
        .expect("failed to run soma target --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("target_help", stdout);
}

#[test]
fn test_client_help() {
    let output = soma_cmd()
        .args(["client", "--help"])
        .output()
        .expect("failed to run soma client --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("client_help", stdout);
}

#[test]
fn test_network_help() {
    let output = soma_cmd()
        .args(["network", "--help"])
        .output()
        .expect("failed to run soma network --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("network_help", stdout);
}

#[test]
fn test_genesis_help() {
    let output = soma_cmd()
        .args(["genesis", "--help"])
        .output()
        .expect("failed to run soma genesis --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("genesis_help", stdout);
}

#[test]
fn test_keytool_help() {
    let output = soma_cmd()
        .args(["keytool", "--help"])
        .output()
        .expect("failed to run soma keytool --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("keytool_help", stdout);
}

#[test]
fn test_faucet_help() {
    let output = soma_cmd()
        .args(["faucet", "--help"])
        .output()
        .expect("failed to run soma faucet --help");
    let stdout = String::from_utf8_lossy(&output.stdout);

    insta::assert_snapshot!("faucet_help", stdout);
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

#[test]
fn test_send_missing_required_args() {
    soma_cmd()
        .args(["send"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--to"));
}

#[test]
fn test_model_commit_missing_args() {
    soma_cmd()
        .args(["model", "commit"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("<MODEL_ID>"));
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
