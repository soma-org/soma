use std::str::FromStr;
use std::time::Duration;
use test_cluster::{TestCluster, TestClusterBuilder};
use test_encoder_cluster::{TestEncoderCluster, TestEncoderClusterBuilder};
use types::{
    base::SomaAddress,
    config::genesis_config::{AccountConfig, DEFAULT_GAS_AMOUNT},
    effects::object_change::IDOperation,
};

use anyhow::{anyhow, Result};
use futures::TryStreamExt;
use rpc::proto::soma::{transaction_kind::Kind, SubscribeCheckpointsRequest};
use rpc::utils::field::{FieldMask, FieldMaskUtil};
use sdk::SomaClient;
use tracing::info;
use types::digests::TransactionDigest;
use types::effects::{TransactionEffects, TransactionEffectsAPI};
use types::full_checkpoint_content::ObjectSet;
use types::object::{ObjectID, ObjectType};

/// Sets up an integrated test environment with validators and encoders.
pub async fn setup_integrated_encoder_validator_test(
    num_validators: usize,
    num_encoders: usize,
    encoder_candidates: impl IntoIterator<Item = SomaAddress>,
) -> (TestCluster, TestEncoderCluster) {
    // Create accounts for encoders with appropriate gas amounts
    let encoder_candidate_accounts = encoder_candidates
        .into_iter()
        .map(|address| AccountConfig {
            address: Some(address),
            gas_amounts: vec![DEFAULT_GAS_AMOUNT * 10], // Enough gas for transactions
        })
        .collect::<Vec<_>>();

    // Step 1: Build TestCluster with genesis encoders
    let test_cluster_builder = TestClusterBuilder::new()
        .with_num_validators(num_validators)
        .with_num_encoders(num_encoders)
        .with_accounts(encoder_candidate_accounts)
        .with_epoch_duration_ms(10 * 1000); // 10s

    // Build and start TestCluster
    let test_cluster = test_cluster_builder.build().await;

    // Step 2: Get the formatted encoder configs from TestCluster
    let encoder_configs = test_cluster.get_encoder_configs_for_encoder_cluster();

    // Step 3: Build TestEncoderCluster with the configs from TestCluster
    let encoder_cluster = TestEncoderClusterBuilder::new()
        .with_encoders(encoder_configs)
        .with_shared_object_store(test_cluster.object_store())
        .build()
        .await;

    (test_cluster, encoder_cluster)
}

/// Information about a completed shard encoding round
#[derive(Debug, Clone)]
pub struct ShardCompletionInfo {
    /// The digest of the ReportWinner transaction
    pub winner_tx_digest: String,
    /// The checkpoint sequence number where the ReportWinner was included
    pub checkpoint_sequence: u64,
    /// The encoder public keys that signed the report
    pub signers: Vec<String>,
}

/// Extract the ShardInput object ID from EmbedData transaction effects.
///
/// The EmbedData transaction creates a ShardInput object which is later
/// referenced by the ReportWinner transaction.
pub fn extract_shard_input_id(effects: &TransactionEffects) -> Result<ObjectID> {
    // Look through changed_objects for newly created objects
    // EmbedData should create exactly one new object: the ShardInput
    let created_objects: Vec<_> = effects
        .changed_objects
        .iter()
        .filter(|(_, change)| change.id_operation == IDOperation::Created)
        .collect();

    match created_objects.len() {
        0 => Err(anyhow!("No created objects found in transaction effects")),
        1 => {
            let (object_id, _) = created_objects[0];
            info!(object_id = %object_id, "Found created ShardInput object");
            Ok(*object_id)
        }
        n => {
            // If there are multiple created objects, log them all and return the first
            // In practice, EmbedData should only create one ShardInput
            info!(
                count = n,
                objects = ?created_objects.iter().map(|(id, _)| id.to_string()).collect::<Vec<_>>(),
                "Multiple objects created, using first one as ShardInput"
            );
            let (object_id, _) = created_objects[0];
            Ok(*object_id)
        }
    }
}

/// Waits for a ReportWinner transaction for a specific shard input object.
///
/// This function subscribes to the checkpoint stream and watches for a
/// ReportWinner transaction that references the given ShardInput object ID.
///
/// # Arguments
/// * `client` - The SomaClient to use for subscription
/// * `shard_input_object_id` - The ObjectID of the ShardInput created by EmbedData
/// * `timeout` - Maximum time to wait for the ReportWinner transaction
///
/// # Returns
/// Information about the shard completion including the winner transaction digest
pub async fn wait_for_shard_completion(
    client: &SomaClient,
    shard_input_object_id: &ObjectID,
    timeout: Duration,
) -> Result<ShardCompletionInfo> {
    // Create subscription request with the fields we need to inspect transactions
    let mut request = SubscribeCheckpointsRequest::default();
    request.read_mask = Some(FieldMask::from_paths([
        "sequence_number",
        "transactions.digest",
        "transactions.transaction.kind",
    ]));

    info!(
        shard_input = %shard_input_object_id,
        "Subscribing to checkpoints, waiting for ReportWinner"
    );

    let mut stream = client
        .subscribe_checkpoints(request)
        .await
        .map_err(|e| anyhow!("Failed to subscribe to checkpoints: {}", e))?;

    let expected_object_id_hex = shard_input_object_id.to_hex();

    let wait_future = async {
        while let Some(response) = stream
            .try_next()
            .await
            .map_err(|e| anyhow!("Checkpoint stream error: {}", e))?
        {
            let checkpoint = response
                .checkpoint
                .ok_or_else(|| anyhow!("Missing checkpoint in response"))?;

            let seq_num = response.cursor.unwrap_or(0);

            for executed_tx in checkpoint.transactions.iter() {
                // Get the transaction kind
                let tx_kind = executed_tx
                    .transaction
                    .as_ref()
                    .and_then(|tx| tx.kind.as_ref())
                    .and_then(|k| k.kind.as_ref());

                if let Some(Kind::ReportWinner(report)) = tx_kind {
                    // Check if this ReportWinner references our shard
                    let matches = report
                        .shard_ref
                        .as_ref()
                        .and_then(|r| r.object_id.as_ref())
                        .map(|id| id == &expected_object_id_hex)
                        .unwrap_or(false);

                    if matches {
                        let digest = executed_tx
                            .digest
                            .clone()
                            .ok_or_else(|| anyhow!("Missing transaction digest"))?;

                        info!(
                            tx_digest = %digest,
                            checkpoint = seq_num,
                            signers_count = report.signers.len(),
                            "Found ReportWinner for shard"
                        );

                        return Ok(ShardCompletionInfo {
                            winner_tx_digest: digest,
                            checkpoint_sequence: seq_num,
                            signers: report.signers.clone(),
                        });
                    }
                }
            }
        }

        Err(anyhow!("Checkpoint stream ended unexpectedly"))
    };

    tokio::select! {
        result = wait_future => result,
        _ = tokio::time::sleep(timeout) => {
            Err(anyhow!(
                "Timeout after {:?} waiting for ReportWinner transaction for shard {}",
                timeout,
                shard_input_object_id
            ))
        }
    }
}

/// Convenience function that combines EmbedData execution with waiting for completion.
///
/// This is useful for tests that want to submit data and wait for the full
/// encoding round to complete.
pub async fn embed_data_and_wait_for_completion(
    client: &SomaClient,
    effects: &TransactionEffects,
    objects: &ObjectSet,
    timeout: Duration,
) -> Result<ShardCompletionInfo> {
    let shard_input_id = extract_shard_input_id(effects)?;

    info!(
        shard_input = %shard_input_id,
        "EmbedData created ShardInput, waiting for encoding completion"
    );

    wait_for_shard_completion(client, &shard_input_id, timeout).await
}
