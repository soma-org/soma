use std::pin::Pin;
use std::time::Duration;

use futures::Stream;
use futures::StreamExt;
use tap::Pipe;
use tonic::metadata::MetadataMap;
use tracing::info;

use crate::api::rpc_client;
use crate::api::rpc_client::HeadersInterceptor;
use crate::proto::TryFromProtoError;
use crate::proto::soma as proto;
use crate::proto::soma::InitiateShardWorkRequest;
use crate::proto::soma::InitiateShardWorkResponse;
use crate::proto::soma::ListOwnedObjectsRequest;
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
