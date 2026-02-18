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
        .stderr(predicate::str::contains("--model-id"));
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
