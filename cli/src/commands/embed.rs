use anyhow::{anyhow, Result};
use fastcrypto::hash::HashFunction as _;
use sdk::wallet_context::WalletContext;
use std::time::Duration;
use types::checksum::Checksum;
use types::crypto::DefaultHash;
use types::metadata::{
    DefaultDownloadMetadata, DefaultDownloadMetadataV1, DownloadMetadata, Metadata, MetadataV1,
};
use types::object::ObjectID;
use types::transaction::TransactionKind;
use url::Url;

use crate::client_commands::TxProcessingArgs;
use crate::response::{EmbedCommandResponse, EmbedCompletedOutput};

/// Execute the embed command (embed data on the Soma network)
///
/// Downloads data from the URL to compute its checksum, then submits
/// an embedding transaction and waits for completion.
pub async fn execute(
    context: &mut WalletContext,
    url: String,
    target: Option<ObjectID>,
    coin: Option<ObjectID>,
    timeout: u64,
    tx_args: TxProcessingArgs,
) -> Result<EmbedCommandResponse> {
    let sender = context.active_address()?;
    let client = context.get_client().await?;

    // Parse URL
    let parsed_url = Url::parse(&url).map_err(|e| anyhow!("Invalid URL: {}", e))?;

    // Download data to compute checksum
    println!("Downloading data from {}...", url);
    let data = download_data(&url).await?;
    let size = data.len();
    println!("Downloaded {} bytes", size);

    // Compute checksum
    let checksum = compute_checksum(&data);

    // Build download metadata
    let metadata = Metadata::V1(MetadataV1::new(checksum, size));
    let download_metadata = DownloadMetadata::Default(DefaultDownloadMetadata::V1(
        DefaultDownloadMetadataV1::new(parsed_url, metadata),
    ));

    // Get coin reference
    let coin_ref = match coin {
        Some(coin_id) => {
            let obj = client
                .get_object(coin_id)
                .await
                .map_err(|e| anyhow!("Failed to get coin: {}", e))?;
            obj.compute_object_reference()
        }
        None => context
            .get_one_gas_object_owned_by_address(sender)
            .await?
            .ok_or_else(|| anyhow!("No coins found for address {}", sender))?,
    };

    // Get target reference if specified
    let target_ref = match target {
        Some(target_id) => {
            let obj = client
                .get_object(target_id)
                .await
                .map_err(|e| anyhow!("Failed to get target: {}", e))?;
            Some(obj.compute_object_reference())
        }
        None => None,
    };

    let kind = TransactionKind::EmbedData {
        download_metadata,
        coin_ref,
        target_ref,
    };

    // Handle serialization
    if tx_args.serialize_unsigned_transaction {
        use fastcrypto::encoding::{Base64, Encoding};
        use sdk::transaction_builder::TransactionBuilder;

        let builder = TransactionBuilder::new(context);
        let tx_data = builder
            .build_transaction_data(sender, kind, Some(coin_ref))
            .await?;
        let bytes = bcs::to_bytes(&tx_data)?;
        let encoded = Base64::encode(&bytes);

        return Ok(EmbedCommandResponse::SerializedTransaction {
            serialized_unsigned_transaction: encoded,
        });
    }

    // Build and sign transaction
    use sdk::transaction_builder::TransactionBuilder;
    let builder = TransactionBuilder::new(context);
    let tx_data = builder
        .build_transaction_data(sender, kind, Some(coin_ref))
        .await?;
    drop(builder);

    let tx = context.sign_transaction(&tx_data).await;

    // Execute and wait for completion
    println!("Submitting embedding transaction...");
    let (exec_response, completion) = client
        .execute_embed_data_and_wait_for_completion(&tx, Duration::from_secs(timeout))
        .await
        .map_err(|e| anyhow!("Embed failed: {:?}", e))?;

    // Fetch the shard to get embedding URL
    let shard = client
        .get_shard(completion.shard_id)
        .await
        .map_err(|e| anyhow!("Failed to fetch shard: {}", e))?;

    let embedding_url = shard
        .embeddings_download_metadata
        .map(|m| m.url().to_string());

    Ok(EmbedCommandResponse::Completed(EmbedCompletedOutput {
        shard_id: completion.shard_id,
        tx_digest: *tx.digest(),
        checkpoint: exec_response.checkpoint_sequence_number,
        winner_tx_digest: completion.winner_tx_digest,
        embedding_url,
    }))
}

/// Download data from a URL
async fn download_data(url: &str) -> Result<Vec<u8>> {
    let response = reqwest::get(url)
        .await
        .map_err(|e| anyhow!("Failed to download data: {}", e))?;

    if !response.status().is_success() {
        return Err(anyhow!("HTTP error: {}", response.status()));
    }

    response
        .bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| anyhow!("Failed to read response body: {}", e))
}

/// Compute checksum of data
fn compute_checksum(data: &[u8]) -> Checksum {
    let mut hasher = DefaultHash::default();
    hasher.update(data);
    Checksum::new_from_hash(hasher.finalize().into())
}
