// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Indexer integration tests.
//!
//! Tests:
//! 1. test_checkpoint_binpb_zst_format — Verify checkpoints are written in .binpb.zst format
//! 2. test_checkpoint_roundtrip — Verify checkpoint encode->decode roundtrip preserves data
//! 3. test_checkpoint_transactions_included — Verify user txs appear in checkpoint data

use std::path::PathBuf;
use std::time::Duration;

use rpc::utils::checkpoint_blob;
use test_cluster::TestClusterBuilder;
use tokio::time::sleep;
use tracing::info;
use types::effects::TransactionEffectsAPI;
use types::full_checkpoint_content::Checkpoint;
use types::transaction::{TransactionData, TransactionKind};
use utils::logging::init_tracing;

/// Wait for a .binpb.zst checkpoint file to appear in the given directory.
async fn wait_for_checkpoint_file(
    dir: &PathBuf,
    min_seq: u64,
    timeout: Duration,
) -> (PathBuf, u64) {
    let start = tokio::time::Instant::now();
    while start.elapsed() < timeout {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(seq_str) = name.strip_suffix(".binpb.zst") {
                    if let Ok(seq) = seq_str.parse::<u64>() {
                        if seq >= min_seq {
                            return (entry.path(), seq);
                        }
                    }
                }
            }
        }
        sleep(Duration::from_millis(100)).await;
    }
    panic!(
        "Timeout waiting for .binpb.zst checkpoint file with seq >= {} in {:?}",
        min_seq, dir
    );
}

/// Read and decode a checkpoint from a .binpb.zst file.
fn read_checkpoint(path: &PathBuf) -> Checkpoint {
    let bytes = std::fs::read(path).expect("Failed to read checkpoint file");
    checkpoint_blob::decode_checkpoint(&bytes).expect("Failed to decode checkpoint")
}

/// Verify checkpoints are written in .binpb.zst format and can be decoded.
#[cfg(msim)]
#[msim::sim_test]
async fn test_checkpoint_binpb_zst_format() {
    init_tracing();

    let ingestion_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let ingestion_path = ingestion_dir.path().to_path_buf();

    let test_cluster = TestClusterBuilder::new()
        .with_data_ingestion_dir(ingestion_path.clone())
        .build()
        .await;

    // Execute a transaction to generate checkpoint data
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .unwrap();

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: gas,
            amount: Some(1000),
            recipient,
        },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    // Wait for checkpoint file to appear
    let (file_path, seq) =
        wait_for_checkpoint_file(&ingestion_path, 0, Duration::from_secs(30)).await;
    info!("Found checkpoint file: {:?} (seq {})", file_path, seq);

    // Verify the file is a valid .binpb.zst checkpoint
    let checkpoint = read_checkpoint(&file_path);
    assert_eq!(checkpoint.summary.sequence_number, seq);

    // Verify no .chk files were written (only .binpb.zst)
    let chk_count = std::fs::read_dir(&ingestion_path)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().ends_with(".chk"))
        .count();
    assert_eq!(chk_count, 0, "No .chk files should be written");

    info!(
        "Checkpoint {} decoded successfully with {} transactions",
        seq,
        checkpoint.transactions.len()
    );
}

/// Verify the encode -> decode roundtrip preserves checkpoint data.
#[cfg(msim)]
#[msim::sim_test]
async fn test_checkpoint_roundtrip() {
    init_tracing();

    let ingestion_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let ingestion_path = ingestion_dir.path().to_path_buf();

    let _test_cluster = TestClusterBuilder::new()
        .with_data_ingestion_dir(ingestion_path.clone())
        .build()
        .await;

    // Wait for genesis checkpoint
    let (file_path, seq) =
        wait_for_checkpoint_file(&ingestion_path, 0, Duration::from_secs(30)).await;

    // Read the raw bytes and decode
    let raw_bytes = std::fs::read(&file_path).expect("Failed to read file");
    let checkpoint = checkpoint_blob::decode_checkpoint(&raw_bytes).expect("Failed to decode");

    // Re-encode and decode again
    let re_encoded =
        checkpoint_blob::encode_checkpoint(&checkpoint).expect("Failed to re-encode");
    let checkpoint2 =
        checkpoint_blob::decode_checkpoint(&re_encoded).expect("Failed to decode roundtrip");

    // Verify key fields match
    assert_eq!(
        checkpoint.summary.sequence_number,
        checkpoint2.summary.sequence_number
    );
    assert_eq!(checkpoint.summary.epoch, checkpoint2.summary.epoch);
    assert_eq!(
        checkpoint.transactions.len(),
        checkpoint2.transactions.len()
    );
    assert_eq!(
        checkpoint.summary.network_total_transactions,
        checkpoint2.summary.network_total_transactions
    );

    info!("Roundtrip encode/decode successful for checkpoint {}", seq);
}

/// Verify that user transactions appear in checkpoint files.
#[cfg(msim)]
#[msim::sim_test]
async fn test_checkpoint_transactions_included() {
    init_tracing();

    let ingestion_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let ingestion_path = ingestion_dir.path().to_path_buf();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(10_000)
        .with_data_ingestion_dir(ingestion_path.clone())
        .build()
        .await;

    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    // Use AddStake which is the most reliable test workload
    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .unwrap();

    let tx_data = TransactionData::new(
        TransactionKind::AddStake {
            coin: gas,
            amount: Some(1000),
            validator: test_cluster.swarm.validator_addresses()[0],
        },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());
    let expected_digest = *response.effects.transaction_digest();

    // Poll checkpoint files until we find the transaction (up to 30 seconds)
    let mut found_tx = false;
    let start = tokio::time::Instant::now();
    let timeout = Duration::from_secs(30);

    while start.elapsed() < timeout && !found_tx {
        if let Ok(entries) = std::fs::read_dir(&ingestion_path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.ends_with(".binpb.zst") {
                    continue;
                }

                let checkpoint = read_checkpoint(&entry.path().to_path_buf());
                for tx in &checkpoint.transactions {
                    if tx.transaction.digest() == expected_digest {
                        found_tx = true;

                        // Verify the transaction effects are present and successful
                        assert!(
                            tx.effects.status().is_ok(),
                            "Transaction effects should indicate success"
                        );

                        info!(
                            "Found transaction {} in checkpoint {}",
                            expected_digest, checkpoint.summary.sequence_number
                        );
                        break;
                    }
                }
                if found_tx {
                    break;
                }
            }
        }
        if !found_tx {
            sleep(Duration::from_millis(200)).await;
        }
    }

    if !found_tx {
        // Debug: list all checkpoint files and their transaction counts
        let mut debug_info = String::new();
        let mut file_count = 0;
        let mut tx_count = 0;
        if let Ok(entries) = std::fs::read_dir(&ingestion_path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".binpb.zst") {
                    let cp = read_checkpoint(&entry.path().to_path_buf());
                    debug_info.push_str(&format!(
                        "\n  cp {} (epoch {}): {} txs, digests: {:?}",
                        cp.summary.sequence_number,
                        cp.summary.epoch,
                        cp.transactions.len(),
                        cp.transactions.iter().map(|t| t.transaction.digest()).collect::<Vec<_>>(),
                    ));
                    tx_count += cp.transactions.len();
                    file_count += 1;
                }
            }
        }
        panic!(
            "Expected to find transaction {} in {} checkpoint files ({} total txs){}",
            expected_digest, file_count, tx_count, debug_info
        );
    }
}
