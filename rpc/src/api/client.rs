// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::time::Duration;

use futures::{Stream, StreamExt, TryStreamExt};
use prost_types::FieldMask;
use tap::Pipe;
use tonic::metadata::MetadataMap;
use tracing::info;
use types::checkpoints::{CertifiedCheckpointSummary, CheckpointSequenceNumber};
use types::digests::TransactionDigest;
use types::effects::object_change::IDOperation;
use types::effects::{TransactionEffects, TransactionEffectsAPI as _};
use types::full_checkpoint_content::{CheckpointData, ObjectSet};
use types::object::{Object, ObjectID, Version};
use types::transaction::Transaction;

use crate::api::rpc_client::HeadersInterceptor;
use crate::api::{ServerVersion, rpc_client};
use crate::proto::soma::transaction_kind::Kind;
use crate::proto::soma::{
    GetEpochResponse, ListOwnedObjectsRequest, SimulateTransactionRequest,
    SimulateTransactionResponse, SubscribeCheckpointsRequest,
};
use crate::proto::{TryFromProtoError, soma as proto};
use crate::utils::field::FieldMaskUtil;
use crate::utils::types_conversions::SdkTypeConversionError;
pub type Result<T, E = tonic::Status> = std::result::Result<T, E>;
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

use tonic::Status;

#[derive(Clone)]
pub struct Client(rpc_client::Client);

impl Client {
    #[allow(clippy::result_large_err)]
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

        let (metadata, checkpoint, _extentions) =
            self.0.ledger_client().get_checkpoint(request).await?.into_parts();

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
        self.get_object_internal(object_id, Some(version.value())).await
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

        let (metadata, object, _extentions) =
            self.0.ledger_client().get_object(request).await?.into_parts();

        let object = object.object.ok_or_else(|| tonic::Status::not_found("no object returned"))?;
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
            .with_read_mask(FieldMask::from_paths(["effects", "balance_changes", "objects"]));

        let (metadata, response, _extentions) =
            self.0.execution_client().execute_transaction(request).await?.into_parts();

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
            .with_read_mask(FieldMask::from_paths(["effects", "balance_changes", "objects"]));

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
        self.0.subscription_client().subscribe_checkpoints(request).await.map(|r| r.into_inner())
    }

    pub async fn get_latest_system_state(&mut self) -> Result<types::system_state::SystemState> {
        let request = proto::GetEpochRequest::latest()
            .with_read_mask(FieldMask::from_paths(["system_state", "epoch"]));
        let response = self.0.ledger_client().get_epoch(request).await?.into_inner();

        (*response
            .epoch
            .ok_or_else(|| {
                tonic::Status::not_found("epoch not found in get system state response")
            })?
            .system_state
            .ok_or_else(|| {
                tonic::Status::not_found("system_state not found in get system state response")
            })?)
        .try_into()
        .map_err(|e| {
            tonic::Status::internal(format!(
                "system_state could not be converted to domain type: {}",
                e
            ))
        })
    }

    pub async fn get_balance(&mut self, owner: &types::base::SomaAddress) -> Result<u64> {
        let request = proto::GetBalanceRequest { owner: Some(owner.to_string()) };
        let response = self.0.state_client().get_balance(request).await?.into_inner();
        response.balance.ok_or_else(|| tonic::Status::not_found("balance not found in response"))
    }

    pub async fn get_chain_identifier(&mut self) -> Result<String> {
        let request = crate::proto::soma::GetServiceInfoRequest::default();
        let response = self.0.ledger_client().get_service_info(request).await?.into_inner();

        response
            .chain_id
            .ok_or_else(|| tonic::Status::not_found("chain_id not found in service info response"))
    }

    pub async fn get_chain_name(&mut self) -> Result<String> {
        let request = crate::proto::soma::GetServiceInfoRequest::default();
        let response = self.0.ledger_client().get_service_info(request).await?.into_inner();

        response
            .chain
            .ok_or_else(|| tonic::Status::not_found("chain not found in service info response"))
    }

    pub async fn get_server_version(&mut self) -> Result<String> {
        let request = crate::proto::soma::GetServiceInfoRequest::default();
        let response = self.0.ledger_client().get_service_info(request).await?.into_inner();

        response.server.ok_or_else(|| {
            tonic::Status::not_found("server_version not found in service info response")
        })
    }

    pub async fn get_epoch(&mut self, epoch: Option<u64>) -> Result<proto::GetEpochResponse> {
        let request = match epoch {
            Some(e) => proto::GetEpochRequest::new(e),
            None => proto::GetEpochRequest::latest(),
        }
        .with_read_mask(FieldMask::from_paths(["epoch", "protocol_config.protocol_version"]));

        self.0.ledger_client().get_epoch(request).await.map(|r| r.into_inner())
    }

    pub async fn get_protocol_version(&mut self) -> Result<u64> {
        let response = self.get_epoch(None).await?;
        response
            .epoch
            .and_then(|e| e.protocol_config)
            .and_then(|c| c.protocol_version)
            .ok_or_else(|| tonic::Status::not_found("protocol version not found"))
    }

    /// Fetch the epoch with full protocol config (all attributes).
    pub async fn get_epoch_with_config(
        &mut self,
        epoch: Option<u64>,
    ) -> Result<proto::GetEpochResponse> {
        let request = match epoch {
            Some(e) => proto::GetEpochRequest::new(e),
            None => proto::GetEpochRequest::latest(),
        }
        .with_read_mask(FieldMask::from_paths(["epoch", "protocol_config"]));

        self.0.ledger_client().get_epoch(request).await.map(|r| r.into_inner())
    }

    pub async fn simulate_transaction(
        &mut self,
        tx_data: &types::transaction::TransactionData,
    ) -> Result<SimulationResult> {
        let proto_transaction: proto::Transaction = tx_data.clone().into();

        let request = proto::SimulateTransactionRequest::default()
            .with_transaction(proto_transaction)
            .with_read_mask(FieldMask::from_paths([
                "transaction.effects",
                "transaction.balance_changes",
                "transaction.objects",
            ]));

        let (metadata, response, _extensions) =
            self.0.execution_client().simulate_transaction(request).await?.into_parts();

        simulation_result_try_from_proto(&response)
            .map_err(|e| status_from_error_with_metadata(e, metadata))
    }

    pub async fn get_transaction(
        &mut self,
        digest: TransactionDigest,
    ) -> Result<TransactionQueryResult> {
        let request = proto::GetTransactionRequest::default()
            .with_digest(digest.to_string())
            .with_read_mask(FieldMask::from_paths([
                "digest",
                "effects",
                "checkpoint",
                "timestamp",
                "balance_changes",
            ]));

        let (metadata, response, _extensions) =
            self.0.ledger_client().get_transaction(request).await?.into_parts();

        let executed_tx = response
            .transaction
            .ok_or_else(|| tonic::Status::not_found("transaction not found"))?;

        transaction_query_result_try_from_proto(&executed_tx)
            .map_err(|e| status_from_error_with_metadata(e, metadata))
    }

    // --- Marketplace query methods ---

    pub async fn get_ask(&mut self, ask_id: ObjectID) -> Result<Object> {
        let request = proto::GetAskRequest {
            ask_id: Some(ask_id.to_string()),
        };
        let response = self.0.state_client().get_ask(request).await?.into_inner();
        let proto_object =
            response.ask.ok_or_else(|| tonic::Status::not_found("ask not found"))?;
        object_try_from_proto(&proto_object).map_err(|e| tonic::Status::internal(e.to_string()))
    }

    pub async fn get_bid_object(&mut self, bid_id: ObjectID) -> Result<Object> {
        let request = proto::GetBidRequest {
            bid_id: Some(bid_id.to_string()),
        };
        let response = self.0.state_client().get_bid(request).await?.into_inner();
        let proto_object =
            response.bid.ok_or_else(|| tonic::Status::not_found("bid not found"))?;
        object_try_from_proto(&proto_object).map_err(|e| tonic::Status::internal(e.to_string()))
    }

    pub async fn get_settlement(&mut self, settlement_id: ObjectID) -> Result<Object> {
        let request = proto::GetSettlementRequest {
            settlement_id: Some(settlement_id.to_string()),
        };
        let response = self.0.state_client().get_settlement(request).await?.into_inner();
        let proto_object = response
            .settlement
            .ok_or_else(|| tonic::Status::not_found("settlement not found"))?;
        object_try_from_proto(&proto_object).map_err(|e| tonic::Status::internal(e.to_string()))
    }

    /// Get all bids for an ask, returning raw Objects. Caller deserializes Bid from contents.
    pub async fn get_bids_for_ask(
        &mut self,
        ask_id: ObjectID,
        status_filter: Option<&str>,
    ) -> Result<Vec<Object>> {
        let request = proto::GetBidsForAskRequest {
            ask_id: Some(ask_id.to_string()),
            status_filter: status_filter.map(|s| s.to_string()),
        };
        let response = self
            .0
            .state_client()
            .get_bids_for_ask(request)
            .await?
            .into_inner();
        response
            .bids
            .iter()
            .map(|proto_obj| {
                object_try_from_proto(proto_obj)
                    .map_err(|e| tonic::Status::internal(e.to_string()))
            })
            .collect()
    }

    /// Get all open asks. Optionally filter by buyer.
    pub async fn get_open_asks(
        &mut self,
        buyer: Option<&types::base::SomaAddress>,
        page_size: Option<u32>,
    ) -> Result<Vec<Object>> {
        let request = proto::GetOpenAsksRequest {
            buyer: buyer.map(|b| b.to_string()),
            page_size,
        };
        let response = self
            .0
            .state_client()
            .get_open_asks(request)
            .await?
            .into_inner();
        response
            .asks
            .iter()
            .map(|proto_obj| {
                object_try_from_proto(proto_obj)
                    .map_err(|e| tonic::Status::internal(e.to_string()))
            })
            .collect()
    }

    /// Get settlements filtered by buyer and/or seller.
    pub async fn get_settlements(
        &mut self,
        buyer: Option<&types::base::SomaAddress>,
        seller: Option<&types::base::SomaAddress>,
        page_size: Option<u32>,
    ) -> Result<Vec<Object>> {
        let request = proto::GetSettlementsRequest {
            buyer: buyer.map(|b| b.to_string()),
            seller: seller.map(|s| s.to_string()),
            page_size,
        };
        let response = self
            .0
            .state_client()
            .get_settlements(request)
            .await?
            .into_inner();
        response
            .settlements
            .iter()
            .map(|proto_obj| {
                object_try_from_proto(proto_obj)
                    .map_err(|e| tonic::Status::internal(e.to_string()))
            })
            .collect()
    }

    /// Get the current Protocol Fund USDC balance.
    pub async fn get_protocol_fund(&mut self) -> Result<u64> {
        let request = proto::GetProtocolFundRequest {};
        let response = self.0.state_client().get_protocol_fund(request).await?.into_inner();
        response
            .balance
            .ok_or_else(|| tonic::Status::internal("missing balance in response"))
    }

    /// Get reputation summary for an address, computed server-side from settlement data.
    pub async fn get_reputation(
        &mut self,
        address: &types::base::SomaAddress,
    ) -> Result<proto::GetReputationResponse> {
        let request = proto::GetReputationRequest {
            address: Some(address.to_string()),
        };
        Ok(self.0.state_client().get_reputation(request).await?.into_inner())
    }

    /// Get all seller vaults for an owner.
    pub async fn get_vaults(
        &mut self,
        owner: &types::base::SomaAddress,
    ) -> Result<Vec<Object>> {
        let request = proto::GetVaultRequest {
            owner: Some(owner.to_string()),
        };
        let response = self.0.state_client().get_vault(request).await?.into_inner();
        response
            .vaults
            .iter()
            .map(|proto_obj| {
                object_try_from_proto(proto_obj)
                    .map_err(|e| tonic::Status::internal(e.to_string()))
            })
            .collect()
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
    let proto_summary =
        checkpoint.summary.as_ref().ok_or_else(|| TryFromProtoError::missing("summary"))?;

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

    let signature: types::crypto::AuthorityStrongQuorumSignInfo = sdk_signature.into();

    Ok(CertifiedCheckpointSummary::new_from_data_and_sig(summary, signature))
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
    let executed_transaction =
        response.transaction.as_ref().ok_or_else(|| TryFromProtoError::missing("transaction"))?;

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

    TransactionExecutionResponse { effects, balance_changes, objects }.pipe(Ok)
}

fn status_from_error_with_metadata<T: Into<BoxError>>(err: T, metadata: MetadataMap) -> Status {
    let mut status = Status::from_error(err.into());
    *status.metadata_mut() = metadata;
    status
}

#[derive(Debug)]
pub struct SimulationResult {
    pub effects: TransactionEffects,
    pub balance_changes: Vec<types::balance_change::BalanceChange>,
    pub objects: types::full_checkpoint_content::ObjectSet,
}

#[allow(clippy::result_large_err)]
fn simulation_result_try_from_proto(
    response: &proto::SimulateTransactionResponse,
) -> Result<SimulationResult, TryFromProtoError> {
    let executed_tx =
        response.transaction.as_ref().ok_or_else(|| TryFromProtoError::missing("transaction"))?;

    let proto_effects =
        executed_tx.effects.as_ref().ok_or_else(|| TryFromProtoError::missing("effects"))?;

    let sdk_effects: crate::types::TransactionEffects = proto_effects
        .try_into()
        .map_err(|e: TryFromProtoError| TryFromProtoError::invalid("effects", e))?;

    let effects: TransactionEffects = sdk_effects
        .try_into()
        .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("effects", e))?;

    let balance_changes = executed_tx
        .balance_changes
        .iter()
        .map(|bc| bc.try_into())
        .collect::<Result<Vec<types::balance_change::BalanceChange>, _>>()?;

    let objects = executed_tx
        .objects()
        .try_into()
        .map_err(|e: TryFromProtoError| TryFromProtoError::invalid("objects", e))?;

    Ok(SimulationResult { effects, balance_changes, objects })
}

#[derive(Debug)]
pub struct TransactionQueryResult {
    pub digest: TransactionDigest,
    pub effects: TransactionEffects,
    pub checkpoint: Option<u64>,
    pub timestamp_ms: Option<u64>,
    pub balance_changes: Vec<types::balance_change::BalanceChange>,
}

#[allow(clippy::result_large_err)]
fn transaction_query_result_try_from_proto(
    executed_tx: &proto::ExecutedTransaction,
) -> Result<TransactionQueryResult, TryFromProtoError> {
    let digest_str =
        executed_tx.digest.as_ref().ok_or_else(|| TryFromProtoError::missing("digest"))?;

    let digest =
        digest_str.parse().map_err(|e| TryFromProtoError::invalid("digest", format!("{}", e)))?;

    let proto_effects =
        executed_tx.effects.as_ref().ok_or_else(|| TryFromProtoError::missing("effects"))?;

    let sdk_effects: crate::types::TransactionEffects = proto_effects
        .try_into()
        .map_err(|e: TryFromProtoError| TryFromProtoError::invalid("effects", e))?;

    let effects: TransactionEffects = sdk_effects
        .try_into()
        .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("effects", e))?;

    let balance_changes = executed_tx
        .balance_changes
        .iter()
        .map(|bc| bc.try_into())
        .collect::<Result<Vec<_>, _>>()?;

    // Extract timestamp from proto Timestamp
    let timestamp_ms = executed_tx
        .timestamp
        .as_ref()
        .map(|ts| (ts.seconds as u64 * 1000) + (ts.nanos as u64 / 1_000_000));

    Ok(TransactionQueryResult {
        digest,
        effects,
        checkpoint: executed_tx.checkpoint,
        timestamp_ms,
        balance_changes,
    })
}
