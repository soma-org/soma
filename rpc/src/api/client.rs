use std::pin::Pin;
use std::time::Duration;

use bytes::Bytes;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use tap::Pipe;
use tonic::metadata::MetadataMap;
use tracing::info;
use types::effects::TransactionEffectsAPI as _;
use types::effects::object_change::IDOperation;

use crate::api::ServerVersion;
use crate::api::rpc_client;
use crate::api::rpc_client::HeadersInterceptor;
use crate::proto::TryFromProtoError;
use crate::proto::soma as proto;
use crate::proto::soma::InitiateShardWorkRequest;
use crate::proto::soma::InitiateShardWorkResponse;
use crate::proto::soma::ListOwnedObjectsRequest;
use crate::proto::soma::SubscribeCheckpointsRequest;
use crate::proto::soma::transaction_kind::Kind;
use crate::proto::soma::{
    GetClaimableEscrowsRequest, GetClaimableEscrowsResponse, GetClaimableRewardsRequest,
    GetClaimableRewardsResponse, GetShardsByEncoderRequest, GetShardsByEncoderResponse,
    GetShardsByEpochRequest, GetShardsByEpochResponse, GetShardsBySubmitterRequest,
    GetShardsBySubmitterResponse, GetValidTargetsRequest, GetValidTargetsResponse, ShardInfo,
    TargetInfo,
};
use crate::utils::field::FieldMaskUtil;
use crate::utils::types_conversions::SdkTypeConversionError;
use prost_types::FieldMask;
use types::checkpoints::{CertifiedCheckpointSummary, CheckpointSequenceNumber};
use types::effects::TransactionEffects;
use types::full_checkpoint_content::CheckpointData;
use types::full_checkpoint_content::ObjectSet;
use types::object::Object;
use types::object::{ObjectID, Version};
use types::transaction::Transaction;
pub type Result<T, E = tonic::Status> = std::result::Result<T, E>;
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

use tonic::Status;

#[derive(Clone)]
pub struct Client(rpc_client::Client);

impl Client {
    pub fn new<T>(uri: T) -> Result<Self>
    where
        T: TryInto<http::Uri>,
        T::Error: Into<BoxError>,
    {
        rpc_client::Client::new(uri).map(Self)
    }

    pub fn with_headers(self, headers: HeadersInterceptor) -> Self {
        Self(self.0.with_headers(headers))
    }

    pub fn list_owned_objects(
        &self,
        request: impl tonic::IntoRequest<ListOwnedObjectsRequest>,
    ) -> Pin<Box<dyn Stream<Item = Result<Object>> + Send + 'static>> {
        Box::pin(self.0.clone().list_owned_objects(request).map(|o| {
            let object = o?;
            info!("Getting object (gas) : {object:?}");
            object_try_from_proto(&object).map_err(|e| Status::from_error(e.into()))
        }))
    }

    pub async fn get_latest_checkpoint(&mut self) -> Result<CertifiedCheckpointSummary> {
        self.get_checkpoint_internal(None).await
    }

    pub async fn get_checkpoint_summary(
        &mut self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Result<CertifiedCheckpointSummary> {
        self.get_checkpoint_internal(Some(sequence_number)).await
    }

    async fn get_checkpoint_internal(
        &mut self,
        sequence_number: Option<CheckpointSequenceNumber>,
    ) -> Result<CertifiedCheckpointSummary> {
        let mut request = crate::proto::soma::GetCheckpointRequest::default()
            .with_read_mask(FieldMask::from_paths(["summary", "signature"]));
        request.checkpoint_id = sequence_number.map(|sequence_number| {
            proto::get_checkpoint_request::CheckpointId::SequenceNumber(sequence_number)
        });

        let (metadata, checkpoint, _extentions) = self
            .0
            .ledger_client()
            .get_checkpoint(request)
            .await?
            .into_parts();

        let checkpoint = checkpoint
            .checkpoint
            .ok_or_else(|| tonic::Status::not_found("no checkpoint returned"))?;
        certified_checkpoint_summary_try_from_proto(&checkpoint)
            .map_err(|e| status_from_error_with_metadata(e, metadata))
    }

    pub async fn get_full_checkpoint(
        &mut self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Result<CheckpointData> {
        let request = crate::proto::soma::GetCheckpointRequest::by_sequence_number(sequence_number)
            .with_read_mask(checkpoint_data_field_mask());

        let (metadata, response, _extentions) = self
            .0
            .ledger_client()
            .max_decoding_message_size(128 * 1024 * 1024)
            .get_checkpoint(request)
            .await?
            .into_parts();

        let checkpoint = response
            .checkpoint
            .ok_or_else(|| tonic::Status::not_found("no checkpoint returned"))?;
        types::full_checkpoint_content::Checkpoint::try_from(&checkpoint)
            .map(Into::into)
            .map_err(|e| status_from_error_with_metadata(e, metadata))
    }

    pub async fn get_object(&mut self, object_id: ObjectID) -> Result<Object> {
        self.get_object_internal(object_id, None).await
    }

    pub async fn get_object_with_version(
        &mut self,
        object_id: ObjectID,
        version: Version,
    ) -> Result<Object> {
        self.get_object_internal(object_id, Some(version.value()))
            .await
    }

    async fn get_object_internal(
        &mut self,
        object_id: ObjectID,
        version: Option<u64>,
    ) -> Result<Object> {
        let mut request = proto::GetObjectRequest::new(&object_id.into());
        request.version = version;

        request.read_mask = Some(FieldMask::from_paths([
            "object_id",
            "version",
            "digest",
            "object_type",
            "owner",
            "contents",
            "previous_transaction",
        ]));

        let (metadata, object, _extentions) = self
            .0
            .ledger_client()
            .get_object(request)
            .await?
            .into_parts();

        let object = object
            .object
            .ok_or_else(|| tonic::Status::not_found("no object returned"))?;
        object_try_from_proto(&object).map_err(|e| status_from_error_with_metadata(e, metadata))
    }

    pub async fn execute_transaction(
        &mut self,
        transaction: &Transaction,
    ) -> Result<TransactionExecutionResponse> {
        let tx_data = transaction.inner().intent_message.value.clone();
        let proto_transaction: proto::Transaction = tx_data.into();

        let signatures: Vec<proto::UserSignature> = transaction
            .inner()
            .tx_signatures
            .clone()
            .into_iter()
            .map(|signature| signature.into())
            .collect();

        let request = proto::ExecuteTransactionRequest::new(proto_transaction)
            .with_signatures(signatures)
            .with_read_mask(FieldMask::from_paths([
                "effects",
                "balance_changes",
                "objects",
            ]));

        let (metadata, response, _extentions) = self
            .0
            .execution_client()
            .execute_transaction(request)
            .await?
            .into_parts();

        execute_transaction_response_try_from_proto(&response)
            .map_err(|e| status_from_error_with_metadata(e, metadata))
    }

    /// Execute transaction and wait for it to be checkpointed (indexes updated)
    pub async fn execute_transaction_and_wait_for_checkpoint(
        &mut self,
        transaction: &Transaction,
        timeout: Duration,
    ) -> Result<TransactionExecutionResponseWithCheckpoint> {
        let tx_data = transaction.inner().intent_message.value.clone();
        let proto_transaction: proto::Transaction = tx_data.into();

        let signatures: Vec<proto::UserSignature> = transaction
            .inner()
            .tx_signatures
            .clone()
            .into_iter()
            .map(|signature| signature.into())
            .collect();

        let request = proto::ExecuteTransactionRequest::new(proto_transaction)
            .with_signatures(signatures)
            .with_read_mask(FieldMask::from_paths([
                "effects",
                "balance_changes",
                "objects",
            ]));

        let execute_and_wait_response = self
            .0
            .clone()
            .execute_transaction_and_wait_for_checkpoint(request, timeout)
            .await
            .map_err(|e| tonic::Status::internal(e.to_string()))?;

        let (metadata, response, _) = execute_and_wait_response.response.into_parts();
        let base_response = execute_transaction_response_try_from_proto(&response)
            .map_err(|e| status_from_error_with_metadata(e, metadata))?;

        Ok(TransactionExecutionResponseWithCheckpoint {
            effects: base_response.effects,
            balance_changes: base_response.balance_changes,
            objects: base_response.objects,
            checkpoint_sequence_number: execute_and_wait_response.checkpoint_sequence_number,
        })
    }

    /// Subscribe to checkpoints stream
    pub async fn subscribe_checkpoints(
        &mut self,
        request: impl tonic::IntoRequest<crate::proto::soma::SubscribeCheckpointsRequest>,
    ) -> Result<tonic::Streaming<crate::proto::soma::SubscribeCheckpointsResponse>> {
        self.0
            .subscription_client()
            .subscribe_checkpoints(request)
            .await
            .map(|r| r.into_inner())
    }

    pub async fn get_chain_identifier(&mut self) -> Result<String> {
        let request = crate::proto::soma::GetServiceInfoRequest::default();
        let response = self
            .0
            .ledger_client()
            .get_service_info(request)
            .await?
            .into_inner();

        response
            .chain_id
            .ok_or_else(|| tonic::Status::not_found("chain_id not found in service info response"))
    }

    pub async fn get_server_version(&mut self) -> Result<String> {
        let request = crate::proto::soma::GetServiceInfoRequest::default();
        let response = self
            .0
            .ledger_client()
            .get_service_info(request)
            .await?
            .into_inner();

        response.server.ok_or_else(|| {
            tonic::Status::not_found("server_version not found in service info response")
        })
    }

    /// Initiate shard work for a given shard input
    pub async fn initiate_shard_work(
        &mut self,
        request: impl tonic::IntoRequest<InitiateShardWorkRequest>,
    ) -> Result<InitiateShardWorkResponse> {
        let (metadata, response, _extensions) = self
            .0
            .execution_client()
            .initiate_shard_work(request)
            .await?
            .into_parts();

        Ok(response)
    }

    // =========================================================================
    // SHARD QUERIES
    // =========================================================================

    /// Get all shards created in a specific epoch
    pub async fn get_shards_by_epoch(&mut self, epoch: u64) -> Result<GetShardsByEpochResponse> {
        let request = GetShardsByEpochRequest {
            epoch: Some(epoch),
            cursor: None,
            limit: None,
        };

        self.0
            .ledger_client()
            .get_shards_by_epoch(request)
            .await
            .map(|r| r.into_inner())
    }

    /// Get all shards created in a specific epoch with pagination
    pub async fn get_shards_by_epoch_with_pagination(
        &mut self,
        epoch: u64,
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetShardsByEpochResponse> {
        let request = GetShardsByEpochRequest {
            epoch: Some(epoch),
            cursor,
            limit,
        };

        self.0
            .ledger_client()
            .get_shards_by_epoch(request)
            .await
            .map(|r| r.into_inner())
    }

    /// Get shards submitted by a specific address
    pub async fn get_shards_by_submitter(
        &mut self,
        submitter: &[u8],
        epoch: Option<u64>,
    ) -> Result<GetShardsBySubmitterResponse> {
        let request = GetShardsBySubmitterRequest {
            submitter: Some(submitter.to_vec().into()),
            epoch,
            cursor: None,
            limit: None,
        };

        self.0
            .ledger_client()
            .get_shards_by_submitter(request)
            .await
            .map(|r| r.into_inner())
    }

    /// Get shards submitted by a specific address with pagination
    pub async fn get_shards_by_submitter_with_pagination(
        &mut self,
        submitter: &[u8],
        epoch: Option<u64>,
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetShardsBySubmitterResponse> {
        let request = GetShardsBySubmitterRequest {
            submitter: Some(submitter.to_vec().into()),
            epoch,
            cursor,
            limit,
        };

        self.0
            .ledger_client()
            .get_shards_by_submitter(request)
            .await
            .map(|r| r.into_inner())
    }

    /// Get shards won by a specific encoder
    pub async fn get_shards_by_encoder(
        &mut self,
        encoder: &[u8],
    ) -> Result<GetShardsByEncoderResponse> {
        let request = GetShardsByEncoderRequest {
            encoder: Some(encoder.to_vec().into()),
            cursor: None,
            limit: None,
        };

        self.0
            .ledger_client()
            .get_shards_by_encoder(request)
            .await
            .map(|r| r.into_inner())
    }

    /// Get shards won by a specific encoder with pagination
    pub async fn get_shards_by_encoder_with_pagination(
        &mut self,
        encoder: &[u8],
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetShardsByEncoderResponse> {
        let request = GetShardsByEncoderRequest {
            encoder: Some(encoder.to_vec().into()),
            cursor,
            limit,
        };

        self.0
            .ledger_client()
            .get_shards_by_encoder(request)
            .await
            .map(|r| r.into_inner())
    }

    /// Get claimable escrows for the current epoch
    pub async fn get_claimable_escrows(
        &mut self,
        current_epoch: u64,
    ) -> Result<GetClaimableEscrowsResponse> {
        let request = GetClaimableEscrowsRequest {
            current_epoch: Some(current_epoch),
            cursor: None,
            limit: None,
        };

        self.0
            .ledger_client()
            .get_claimable_escrows(request)
            .await
            .map(|r| r.into_inner())
    }

    /// Get claimable escrows with pagination
    pub async fn get_claimable_escrows_with_pagination(
        &mut self,
        current_epoch: u64,
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetClaimableEscrowsResponse> {
        let request = GetClaimableEscrowsRequest {
            current_epoch: Some(current_epoch),
            cursor,
            limit,
        };

        self.0
            .ledger_client()
            .get_claimable_escrows(request)
            .await
            .map(|r| r.into_inner())
    }

    // =========================================================================
    // TARGET QUERIES
    // =========================================================================

    /// Get all targets valid for competition in the given epoch
    pub async fn get_valid_targets(&mut self, epoch: u64) -> Result<GetValidTargetsResponse> {
        let request = GetValidTargetsRequest {
            epoch: Some(epoch),
            cursor: None,
            limit: None,
        };

        self.0
            .ledger_client()
            .get_valid_targets(request)
            .await
            .map(|r| r.into_inner())
    }

    /// Get all targets valid for competition with pagination
    pub async fn get_valid_targets_with_pagination(
        &mut self,
        epoch: u64,
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetValidTargetsResponse> {
        let request = GetValidTargetsRequest {
            epoch: Some(epoch),
            cursor,
            limit,
        };

        self.0
            .ledger_client()
            .get_valid_targets(request)
            .await
            .map(|r| r.into_inner())
    }

    /// Get claimable rewards for the current epoch
    pub async fn get_claimable_rewards(
        &mut self,
        current_epoch: u64,
    ) -> Result<GetClaimableRewardsResponse> {
        let request = GetClaimableRewardsRequest {
            current_epoch: Some(current_epoch),
            cursor: None,
            limit: None,
        };

        self.0
            .ledger_client()
            .get_claimable_rewards(request)
            .await
            .map(|r| r.into_inner())
    }

    /// Get claimable rewards with pagination
    pub async fn get_claimable_rewards_with_pagination(
        &mut self,
        current_epoch: u64,
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetClaimableRewardsResponse> {
        let request = GetClaimableRewardsRequest {
            current_epoch: Some(current_epoch),
            cursor,
            limit,
        };

        self.0
            .ledger_client()
            .get_claimable_rewards(request)
            .await
            .map(|r| r.into_inner())
    }

    /// Extract the ShardInput object ID from EmbedData transaction effects.
    ///
    /// The EmbedData transaction creates a ShardInput object which tracks
    /// the encoding process and eventually becomes a Shard object.
    pub fn extract_shard_input_id(effects: &TransactionEffects) -> Result<ObjectID, ShardError> {
        // Look for newly created objects in the effects
        let created_objects: Vec<_> = effects
            .changed_objects
            .iter()
            .filter(|(_, change)| change.id_operation == IDOperation::Created)
            .collect();

        match created_objects.first() {
            Some((object_id, _)) => Ok(*object_id),
            None => Err(ShardError::NoShardInputFound),
        }
    }

    /// Subscribe to checkpoints and wait for a ReportWinner transaction
    /// for the specified shard input object.
    ///
    /// Returns information about the completed shard when the ReportWinner
    /// transaction is found.
    ///
    /// # Arguments
    /// * `shard_input_id` - The ObjectID of the ShardInput created by EmbedData
    /// * `timeout` - Maximum time to wait for completion
    ///
    /// # Example
    /// ```no_run
    /// let effects = client.execute_transaction(&embed_tx).await?;
    /// let shard_input_id = Client::extract_shard_input_id(&effects.effects)?;
    /// let completion = client.wait_for_shard_completion(&shard_input_id, Duration::from_secs(60)).await?;
    /// println!("Shard completed in checkpoint {}", completion.checkpoint_sequence);
    /// ```
    pub async fn wait_for_shard_completion(
        &mut self,
        shard_input_id: &ObjectID,
        timeout: Duration,
    ) -> Result<ShardCompletionInfo, ShardError> {
        // Create subscription request with fields needed to inspect transactions
        let mut request = SubscribeCheckpointsRequest::default();
        request.read_mask = Some(FieldMask::from_paths([
            "sequence_number",
            "transactions.digest",
            "transactions.transaction.kind",
        ]));

        let mut stream = self
            .0
            .subscription_client()
            .subscribe_checkpoints(request)
            .await
            .map_err(ShardError::Rpc)?
            .into_inner();

        let expected_object_id_hex = shard_input_id.to_hex();
        let shard_id = *shard_input_id;

        let wait_future = async {
            while let Some(response) = stream.try_next().await.map_err(ShardError::Rpc)? {
                let checkpoint = response
                    .checkpoint
                    .ok_or(ShardError::MissingField("checkpoint"))?;

                let seq_num = response.cursor.unwrap_or(0);

                for executed_tx in checkpoint.transactions.iter() {
                    let tx_kind = executed_tx
                        .transaction
                        .as_ref()
                        .and_then(|tx| tx.kind.as_ref())
                        .and_then(|k| k.kind.as_ref());

                    if let Some(Kind::ReportWinner(report)) = tx_kind {
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
                                .ok_or(ShardError::MissingField("digest"))?;

                            return Ok(ShardCompletionInfo {
                                winner_tx_digest: digest,
                                checkpoint_sequence: seq_num,
                                signers: report.signers.clone(),
                                shard_id,
                            });
                        }
                    }
                }
            }

            Err(ShardError::StreamEnded)
        };

        tokio::select! {
            result = wait_future => result,
            _ = tokio::time::sleep(timeout) => {
                Err(ShardError::Timeout(timeout))
            }
        }
    }

    /// Execute an EmbedData transaction and wait for the shard to complete.
    ///
    /// This is a high-level helper that:
    /// 1. Executes the transaction and waits for it to be checkpointed
    /// 2. Extracts the ShardInput object ID from the effects
    /// 3. Subscribes and waits for the ReportWinner transaction
    ///
    /// # Arguments
    /// * `transaction` - The EmbedData transaction to execute
    /// * `timeout` - Maximum time to wait for shard completion after execution
    ///
    /// # Returns
    /// A tuple of (TransactionExecutionResponse, ShardCompletionInfo)
    pub async fn execute_embed_data_and_wait_for_completion(
        &mut self,
        transaction: &Transaction,
        timeout: Duration,
    ) -> Result<
        (
            TransactionExecutionResponseWithCheckpoint,
            ShardCompletionInfo,
        ),
        ShardError,
    > {
        // Execute and wait for checkpointing
        let response = self
            .execute_transaction_and_wait_for_checkpoint(transaction, timeout)
            .await
            .map_err(ShardError::Rpc)?;

        // Extract shard input ID
        let shard_input_id = Self::extract_shard_input_id(&response.effects)?;

        // Initiate shard work - this is required before encoders will process the shard
        let tx_digest = response.effects.transaction_digest();
        let request = proto::InitiateShardWorkRequest::default()
            .with_checkpoint_seq(response.checkpoint_sequence_number)
            .with_tx_digest(tx_digest.to_string());
        self.initiate_shard_work(request)
            .await
            .map_err(ShardError::Rpc)?;

        // Wait for completion
        let completion = self
            .wait_for_shard_completion(&shard_input_id, timeout)
            .await?;

        Ok((response, completion))
    }
}

/// Information about a completed shard encoding round
#[derive(Debug, Clone)]
pub struct ShardCompletionInfo {
    /// The digest of the ReportWinner transaction
    pub winner_tx_digest: String,
    /// The checkpoint sequence number where the ReportWinner was included
    pub checkpoint_sequence: u64,
    /// The encoder public keys that signed the report (as hex strings)
    pub signers: Vec<String>,
    /// The shard object ID (same as shard_input_id for now)
    pub shard_id: ObjectID,
}

/// Error type for shard operations
#[derive(Debug, thiserror::Error)]
pub enum ShardError {
    #[error("No ShardInput object found in transaction effects")]
    NoShardInputFound,
    #[error("Timeout waiting for shard completion after {0:?}")]
    Timeout(Duration),
    #[error("Checkpoint stream ended unexpectedly")]
    StreamEnded,
    #[error("RPC error: {0}")]
    Rpc(#[from] tonic::Status),
    #[error("Missing field in response: {0}")]
    MissingField(&'static str),
}

#[derive(Debug)]
pub struct TransactionExecutionResponseWithCheckpoint {
    pub effects: TransactionEffects,
    pub balance_changes: Vec<types::balance_change::BalanceChange>,
    pub objects: ObjectSet,
    pub checkpoint_sequence_number: CheckpointSequenceNumber,
}

#[derive(Debug)]
pub struct TransactionExecutionResponse {
    pub effects: TransactionEffects,
    pub balance_changes: Vec<types::balance_change::BalanceChange>,
    pub objects: ObjectSet,
}

/// Field mask for checkpoint data requests.
pub fn checkpoint_data_field_mask() -> FieldMask {
    FieldMask::from_paths([
        "sequence_number",
        "summary",
        "signature",
        "contents",
        "transactions.transaction",
        "transactions.effects",
        "objects.objects",
    ])
}

/// Attempts to parse `CertifiedCheckpointSummary` from a proto::Checkpoint
#[allow(clippy::result_large_err)]
fn certified_checkpoint_summary_try_from_proto(
    checkpoint: &proto::Checkpoint,
) -> Result<CertifiedCheckpointSummary, TryFromProtoError> {
    // Convert proto CheckpointSummary -> SDK CheckpointSummary -> domain CheckpointSummary
    let proto_summary = checkpoint
        .summary
        .as_ref()
        .ok_or_else(|| TryFromProtoError::missing("summary"))?;

    let sdk_summary: crate::types::CheckpointSummary = proto_summary
        .try_into()
        .map_err(|e: TryFromProtoError| TryFromProtoError::invalid("summary", e))?;

    let summary: types::checkpoints::CheckpointSummary = sdk_summary
        .try_into()
        .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("summary", e))?;

    // Convert proto signature -> SDK signature -> domain signature
    let sdk_signature: crate::types::ValidatorAggregatedSignature = checkpoint
        .signature
        .as_ref()
        .ok_or_else(|| TryFromProtoError::missing("signature"))?
        .try_into()?;

    let signature = types::crypto::AuthorityStrongQuorumSignInfo::try_from(sdk_signature)
        .map_err(|e| TryFromProtoError::invalid("signature", e))?;

    Ok(CertifiedCheckpointSummary::new_from_data_and_sig(
        summary, signature,
    ))
}

/// Attempts to parse `Object` from the bcs fields in `GetObjectResponse`
#[allow(clippy::result_large_err)]
fn object_try_from_proto(object: &proto::Object) -> Result<Object, TryFromProtoError> {
    // Convert proto Object -> SDK Object -> domain Object
    let sdk_object: crate::types::Object = object
        .try_into()
        .map_err(|e: TryFromProtoError| TryFromProtoError::invalid("object", e))?;

    sdk_object
        .try_into()
        .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("object", e))
}

/// Attempts to parse `TransactionExecutionResponse` from the fields in `TransactionExecutionResponse`
#[allow(clippy::result_large_err)]
fn execute_transaction_response_try_from_proto(
    response: &proto::ExecuteTransactionResponse,
) -> Result<TransactionExecutionResponse, TryFromProtoError> {
    let executed_transaction = response
        .transaction
        .as_ref()
        .ok_or_else(|| TryFromProtoError::missing("transaction"))?;

    // Convert proto TransactionEffects -> SDK TransactionEffects -> domain TransactionEffects
    let proto_effects = executed_transaction
        .effects
        .as_ref()
        .ok_or_else(|| TryFromProtoError::missing("effects"))?;

    let sdk_effects: crate::types::TransactionEffects = proto_effects
        .try_into()
        .map_err(|e: TryFromProtoError| TryFromProtoError::invalid("effects", e))?;

    let effects: TransactionEffects = sdk_effects
        .try_into()
        .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("effects", e))?;

    // Convert balance changes
    let balance_changes = executed_transaction
        .balance_changes
        .iter()
        .map(|bc| bc.try_into())
        .collect::<Result<Vec<types::balance_change::BalanceChange>, _>>()?;

    // Convert objects
    let objects = executed_transaction
        .objects()
        .try_into()
        .map_err(|e: TryFromProtoError| TryFromProtoError::invalid("objects", e))?;

    TransactionExecutionResponse {
        effects,
        balance_changes,
        objects,
    }
    .pipe(Ok)
}

fn status_from_error_with_metadata<T: Into<BoxError>>(err: T, metadata: MetadataMap) -> Status {
    let mut status = Status::from_error(err.into());
    *status.metadata_mut() = metadata;
    status
}
