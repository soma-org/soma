use bytes::Bytes;
use serde::{Deserialize, Serialize};
use strum::EnumIter;

use crate::{
    crypto::{AuthoritySignInfo, AuthorityStrongQuorumSignInfo},
    digests::TransactionDigest,
    effects::{SignedTransactionEffects, VerifiedSignedTransactionEffects},
    error::SomaError,
    object::{Object, ObjectID},
    transaction::{CertifiedTransaction, SenderSignedData, Transaction},
};

/// A request for information about an object and optionally its
/// parent certificate at a specific version.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct ObjectInfoRequest {
    /// The id of the object to retrieve, at the latest version.
    pub object_id: ObjectID,
}

/// This message provides information about the latest object and its lock.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectInfoResponse {
    /// Value of the requested object in this authority
    pub object: Object,
}

/// Verified version of `ObjectInfoResponse`.
#[derive(Debug, Clone)]
pub struct VerifiedObjectInfoResponse {
    /// Value of the requested object in this authority
    pub object: Object,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransactionInfoRequest {
    pub transaction_digest: TransactionDigest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum TransactionStatus {
    /// Signature over the transaction.
    Signed(AuthoritySignInfo),
    /// For executed transaction, we could return an optional certificate signature on the transaction
    /// (i.e. the signature part of the CertifiedTransaction), as well as the signed effects.
    /// The certificate signature is optional because for transactions executed in previous
    /// epochs, we won't keep around the certificate signatures.
    Executed(Option<AuthorityStrongQuorumSignInfo>, SignedTransactionEffects),
}

impl TransactionStatus {
    pub fn into_signed_for_testing(self) -> AuthoritySignInfo {
        match self {
            Self::Signed(s) => s,
            _ => unreachable!("Incorrect response type"),
        }
    }

    pub fn into_effects_for_testing(self) -> SignedTransactionEffects {
        match self {
            Self::Executed(_, e) => e,
            _ => unreachable!("Incorrect response type"),
        }
    }
}

impl PartialEq for TransactionStatus {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::Signed(s1) => match other {
                Self::Signed(s2) => s1.epoch == s2.epoch,
                _ => false,
            },
            Self::Executed(c1, e1) => match other {
                Self::Executed(c2, e2) => {
                    c1.as_ref().map(|a| a.epoch) == c2.as_ref().map(|a| a.epoch)
                        && e1.epoch() == e2.epoch()
                        && e1.digest() == e2.digest()
                }
                _ => false,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HandleTransactionResponse {
    pub status: TransactionStatus,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransactionInfoResponse {
    pub transaction: SenderSignedData,
    pub status: TransactionStatus,
}

#[derive(Clone, Debug)]
pub struct VerifiedHandleCertificateResponse {
    pub signed_effects: VerifiedSignedTransactionEffects,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SystemStateRequest {
    // This is needed to make gRPC happy.
    pub _unused: bool,
}

/// Given Validators operate with very
/// aggressive object pruning, the return of input/output objects is only done immediately after
/// the transaction has been executed locally on the validator and will not be returned for
/// requests to previously executed transactions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandleCertificateResponse {
    pub effects: SignedTransactionEffects,

    /// If requested, will included all initial versions of objects modified in this transaction.
    /// This includes owned objects included as input into the transaction as well as the assigned
    /// versions of shared objects.
    //
    // TODO: In the future we may want to include shared objects or child objects which were read
    // but not modified during execution.
    pub input_objects: Option<Vec<Object>>,

    /// If requested, will included all changed objects, including mutated, created and unwrapped
    /// objects. In other words, all objects that still exist in the object state after this
    /// transaction.
    pub output_objects: Option<Vec<Object>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandleCertificateRequest {
    pub certificate: CertifiedTransaction,

    pub include_input_objects: bool,
    pub include_output_objects: bool,
}

// =========== ExecutedData ===========

#[derive(Default, Clone)]
pub struct ExecutedData {
    pub effects: crate::effects::TransactionEffects,
    pub input_objects: Vec<crate::object::Object>,
    pub output_objects: Vec<crate::object::Object>,
}

#[derive(Clone, prost::Message)]
pub struct RawExecutedData {
    #[prost(bytes = "bytes", tag = "1")]
    pub effects: Bytes,
    #[prost(bytes = "bytes", repeated, tag = "2")]
    pub input_objects: Vec<Bytes>,
    #[prost(bytes = "bytes", repeated, tag = "3")]
    pub output_objects: Vec<Bytes>,
}

// =========== SubmitTx types ===========

#[derive(Clone, Debug)]
pub struct SubmitTxRequest {
    pub transaction: Option<Transaction>,
    pub ping_type: Option<PingType>,
}

impl SubmitTxRequest {
    pub fn new_transaction(transaction: Transaction) -> Self {
        Self { transaction: Some(transaction), ping_type: None }
    }

    pub fn new_ping(ping_type: PingType) -> Self {
        Self { transaction: None, ping_type: Some(ping_type) }
    }

    pub fn tx_type(&self) -> TxType {
        if let Some(ping_type) = self.ping_type {
            return if ping_type == PingType::FastPath {
                TxType::SingleWriter
            } else {
                TxType::SharedObject
            };
        }
        let transaction = self.transaction.as_ref().unwrap();
        if transaction.is_consensus_tx() { TxType::SharedObject } else { TxType::SingleWriter }
    }

    /// Returns the digest of the transaction if it is a transaction request.
    /// Returns None if it is a ping request.
    pub fn tx_digest(&self) -> Option<TransactionDigest> {
        self.transaction.as_ref().map(|t| *t.digest())
    }
}

pub const TX_TYPE_SINGLE_WRITER_TX: &str = "single_writer";
pub const TX_TYPE_SHARED_OBJ_TX: &str = "shared_object";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumIter)]
pub enum TxType {
    SingleWriter,
    SharedObject,
}

impl TxType {
    pub fn as_str(&self) -> &str {
        match self {
            TxType::SingleWriter => TX_TYPE_SINGLE_WRITER_TX,
            TxType::SharedObject => TX_TYPE_SHARED_OBJ_TX,
        }
    }
}

impl SubmitTxRequest {
    pub fn into_raw(&self) -> Result<RawSubmitTxRequest, SomaError> {
        let transactions = if let Some(transaction) = &self.transaction {
            vec![
                bcs::to_bytes(&transaction)
                    .map_err(|e| SomaError::TransactionSerializationError { error: e.to_string() })?
                    .into(),
            ]
        } else {
            vec![]
        };

        let submit_type =
            if self.ping_type.is_some() { SubmitTxType::Ping } else { SubmitTxType::Default };

        Ok(RawSubmitTxRequest { transactions, submit_type: submit_type.into() })
    }
}

#[derive(Clone)]
pub enum SubmitTxResult {
    Submitted {
        consensus_position: crate::consensus::ConsensusPosition,
    },
    Executed {
        effects_digest: crate::digests::TransactionEffectsDigest,
        // Response should always include details for executed transactions.
        // TODO(fastpath): validate this field is always present and return an error during deserialization.
        details: Option<Box<ExecutedData>>,
        // Whether the transaction was executed using fast path.
        fast_path: bool,
    },
    Rejected {
        error: crate::error::SomaError,
    },
}

impl std::fmt::Debug for SubmitTxResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Submitted { consensus_position } => {
                f.debug_struct("Submitted").field("consensus_position", consensus_position).finish()
            }
            Self::Executed { effects_digest, fast_path, .. } => f
                .debug_struct("Executed")
                .field("effects_digest", &format_args!("{}", effects_digest))
                .field("fast_path", fast_path)
                .finish(),
            Self::Rejected { error } => f.debug_struct("Rejected").field("error", &error).finish(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SubmitTxResponse {
    pub results: Vec<SubmitTxResult>,
}

#[derive(Clone, prost::Message)]
pub struct RawSubmitTxRequest {
    /// The transactions to be submitted. When the vector is empty, then this is treated as a ping request.
    #[prost(bytes = "bytes", repeated, tag = "1")]
    pub transactions: Vec<Bytes>,

    /// The type of submission.
    #[prost(enumeration = "SubmitTxType", tag = "2")]
    pub submit_type: i32,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, prost::Enumeration)]
#[repr(i32)]
pub enum SubmitTxType {
    /// Default submission, submitting one or more transactions.
    /// When there are multiple transactions, allow the transactions to be included separately
    /// and out of order in blocks (batch).
    Default = 0,
    /// Ping request to measure latency, no transactions.
    Ping = 1,
}

#[derive(Clone, prost::Message)]
pub struct RawSubmitTxResponse {
    // Results corresponding to each transaction in the request.
    #[prost(message, repeated, tag = "1")]
    pub results: Vec<RawSubmitTxResult>,
}

#[derive(Clone, prost::Message)]
pub struct RawSubmitTxResult {
    #[prost(oneof = "RawValidatorSubmitStatus", tags = "1, 2, 3")]
    pub inner: Option<RawValidatorSubmitStatus>,
}

#[derive(Clone, prost::Oneof)]
pub enum RawValidatorSubmitStatus {
    // Serialized Consensus Position.
    #[prost(bytes = "bytes", tag = "1")]
    Submitted(Bytes),

    // Transaction has already been executed (finalized).
    #[prost(message, tag = "2")]
    Executed(RawExecutedStatus),

    // Transaction is rejected from consensus submission.
    #[prost(message, tag = "3")]
    Rejected(RawRejectedStatus),
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, prost::Enumeration)]
#[repr(i32)]
pub enum PingType {
    /// Measures the end to end latency from when a transaction is included by a proposed block,
    /// to when the block is committed by consensus.
    Consensus = 0,
    /// Measures the end to end latency from when a transaction is included by a proposed block,
    /// to when the block is certified.
    FastPath = 1,
}

impl PingType {
    pub fn as_str(&self) -> &str {
        match self {
            PingType::FastPath => "fastpath",
            PingType::Consensus => "consensus",
        }
    }
}

// =========== WaitForEffects types ===========

pub struct WaitForEffectsRequest {
    pub transaction_digest: Option<crate::digests::TransactionDigest>,
    /// If consensus position is provided, waits in the server handler for the transaction in it to execute,
    /// either in fastpath outputs or finalized.
    /// If it is not provided, only waits for finalized effects of the transaction in the server handler,
    /// but not for fastpath outputs.
    pub consensus_position: Option<crate::consensus::ConsensusPosition>,
    /// Whether to include details of the effects,
    /// including the effects content, input objects, and output objects.
    pub include_details: bool,
    /// Type of ping request, or None if this is not a ping request.
    pub ping_type: Option<PingType>,
}

#[derive(Clone)]
pub enum WaitForEffectsResponse {
    Executed {
        effects_digest: crate::digests::TransactionEffectsDigest,
        details: Option<Box<ExecutedData>>,
        fast_path: bool,
    },
    // The transaction was rejected by consensus.
    Rejected {
        // The reason of the reject vote casted by the validator.
        // If None, the validator did not cast a reject vote.
        error: Option<crate::error::SomaError>,
    },
    // The transaction position is expired, with the local epoch and committed round.
    // When round is None, the expiration is due to lagging epoch in the request.
    Expired {
        epoch: u64,
        round: Option<u32>,
    },
}

impl std::fmt::Debug for WaitForEffectsResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Executed { effects_digest, fast_path, .. } => f
                .debug_struct("Executed")
                .field("effects_digest", effects_digest)
                .field("fast_path", fast_path)
                .finish(),
            Self::Rejected { error } => f.debug_struct("Rejected").field("error", error).finish(),
            Self::Expired { epoch, round } => {
                f.debug_struct("Expired").field("epoch", epoch).field("round", round).finish()
            }
        }
    }
}

#[derive(Clone, prost::Message)]
pub struct RawWaitForEffectsRequest {
    /// The transaction's digest. If it's a ping request, then this will practically be ignored.
    #[prost(bytes = "bytes", optional, tag = "1")]
    pub transaction_digest: Option<Bytes>,

    /// If provided, wait for the consensus position to execute and wait for fastpath outputs of the transaction,
    /// in addition to waiting for finalized effects.
    /// If not provided, only wait for finalized effects.
    #[prost(bytes = "bytes", optional, tag = "2")]
    pub consensus_position: Option<Bytes>,

    /// Whether to include details of the effects,
    /// including the effects content, input objects, and output objects.
    #[prost(bool, tag = "3")]
    pub include_details: bool,

    /// Set when this is a ping request, to differentiate between fastpath and consensus pings.
    #[prost(enumeration = "PingType", optional, tag = "4")]
    pub ping_type: Option<i32>,
}

impl RawWaitForEffectsRequest {
    pub fn get_ping_type(&self) -> Option<PingType> {
        self.ping_type.map(|p| PingType::try_from(p).expect("Invalid ping type"))
    }
}

#[derive(Clone, prost::Message)]
pub struct RawWaitForEffectsResponse {
    // In order to represent an enum in protobuf, we need to use oneof.
    // However, oneof also allows the value to be unset, which corresponds to None value.
    // Hence, we need to use Option type for `inner`.
    // We expect the value to be set in a valid response.
    #[prost(oneof = "RawValidatorTransactionStatus", tags = "1, 2, 3")]
    pub inner: Option<RawValidatorTransactionStatus>,
}

#[derive(Clone, prost::Oneof)]
pub enum RawValidatorTransactionStatus {
    #[prost(message, tag = "1")]
    Executed(RawExecutedStatus),
    #[prost(message, tag = "2")]
    Rejected(RawRejectedStatus),
    #[prost(message, tag = "3")]
    Expired(RawExpiredStatus),
}

#[derive(Clone, prost::Message)]
pub struct RawExecutedStatus {
    #[prost(bytes = "bytes", tag = "1")]
    pub effects_digest: Bytes,
    #[prost(message, optional, tag = "2")]
    pub details: Option<RawExecutedData>,
    #[prost(bool, tag = "3")]
    pub fast_path: bool,
}

#[derive(Clone, prost::Message)]
pub struct RawRejectedStatus {
    #[prost(bytes = "bytes", optional, tag = "1")]
    pub error: Option<Bytes>,
}

#[derive(Clone, prost::Message)]
pub struct RawExpiredStatus {
    // Validator's current epoch.
    #[prost(uint64, tag = "1")]
    pub epoch: u64,
    // Validator's current round. 0 if it is not yet checked.
    #[prost(uint32, optional, tag = "2")]
    pub round: Option<u32>,
}

// =========== ValidatorHealth types ===========

/// Request for validator health information (used for latency measurement)
#[derive(Clone, Debug, Default)]
pub struct ValidatorHealthRequest {}

/// Response with validator health metrics (data collected but not used for scoring yet)
#[derive(Clone, Debug, Default)]
pub struct ValidatorHealthResponse {
    // Number of in-flight execution transactions from execution scheduler
    // pub num_inflight_execution_transactions: u64,
    /// Number of in-flight consensus transactions
    pub num_inflight_consensus_transactions: u64,
    /// Last committed leader round from Mysticeti consensus
    pub last_committed_leader_round: u32,
    /// Last locally built checkpoint sequence number
    pub last_locally_built_checkpoint: u64,
}

/// Raw protobuf request for validator health information (evolvable)
#[derive(Clone, prost::Message)]
pub struct RawValidatorHealthRequest {}

/// Raw protobuf response with validator health metrics (evolvable)
#[derive(Clone, prost::Message)]
pub struct RawValidatorHealthResponse {
    // Number of pending certificates
    // #[prost(uint64, optional, tag = "1")]
    // pub pending_certificates: Option<u64>,
    /// Number of in-flight consensus messages
    #[prost(uint64, optional, tag = "1")]
    pub inflight_consensus_messages: Option<u64>,
    /// Current consensus round
    #[prost(uint64, optional, tag = "2")]
    pub consensus_round: Option<u64>,
    /// Current checkpoint sequence number
    #[prost(uint64, optional, tag = "3")]
    pub checkpoint_sequence: Option<u64>,
}
// =========== Parse helpers ===========

impl TryFrom<ExecutedData> for RawExecutedData {
    type Error = crate::error::SomaError;

    fn try_from(value: ExecutedData) -> Result<Self, Self::Error> {
        let effects = bcs::to_bytes(&value.effects)
            .map_err(|err| crate::error::SomaError::GrpcMessageSerializeError {
                type_info: "ExecutedData.effects".to_string(),
                error: err.to_string(),
            })?
            .into();

        let mut input_objects = Vec::with_capacity(value.input_objects.len());
        for object in value.input_objects {
            input_objects.push(
                bcs::to_bytes(&object)
                    .map_err(|err| crate::error::SomaError::GrpcMessageSerializeError {
                        type_info: "ExecutedData.input_objects".to_string(),
                        error: err.to_string(),
                    })?
                    .into(),
            );
        }
        let mut output_objects = Vec::with_capacity(value.output_objects.len());
        for object in value.output_objects {
            output_objects.push(
                bcs::to_bytes(&object)
                    .map_err(|err| crate::error::SomaError::GrpcMessageSerializeError {
                        type_info: "ExecutedData.output_objects".to_string(),
                        error: err.to_string(),
                    })?
                    .into(),
            );
        }
        Ok(RawExecutedData { effects, input_objects, output_objects })
    }
}

impl TryFrom<RawExecutedData> for ExecutedData {
    type Error = crate::error::SomaError;

    fn try_from(value: RawExecutedData) -> Result<Self, Self::Error> {
        let effects = bcs::from_bytes(&value.effects).map_err(|err| {
            crate::error::SomaError::GrpcMessageDeserializeError {
                type_info: "RawExecutedData.effects".to_string(),
                error: err.to_string(),
            }
        })?;

        let mut input_objects = Vec::with_capacity(value.input_objects.len());
        for object in value.input_objects {
            input_objects.push(bcs::from_bytes(&object).map_err(|err| {
                crate::error::SomaError::GrpcMessageDeserializeError {
                    type_info: "RawExecutedData.input_objects".to_string(),
                    error: err.to_string(),
                }
            })?);
        }
        let mut output_objects = Vec::with_capacity(value.output_objects.len());
        for object in value.output_objects {
            output_objects.push(bcs::from_bytes(&object).map_err(|err| {
                crate::error::SomaError::GrpcMessageDeserializeError {
                    type_info: "RawExecutedData.output_objects".to_string(),
                    error: err.to_string(),
                }
            })?);
        }
        Ok(ExecutedData { effects, input_objects, output_objects })
    }
}

impl TryFrom<SubmitTxResult> for RawSubmitTxResult {
    type Error = crate::error::SomaError;

    fn try_from(value: SubmitTxResult) -> Result<Self, Self::Error> {
        let inner = match value {
            SubmitTxResult::Submitted { consensus_position } => {
                let consensus_position = consensus_position.into_raw()?;
                RawValidatorSubmitStatus::Submitted(consensus_position)
            }
            SubmitTxResult::Executed { effects_digest, details, fast_path } => {
                let raw_executed = try_from_response_executed(effects_digest, details, fast_path)?;
                RawValidatorSubmitStatus::Executed(raw_executed)
            }
            SubmitTxResult::Rejected { error } => {
                RawValidatorSubmitStatus::Rejected(try_from_response_rejected(Some(error))?)
            }
        };
        Ok(RawSubmitTxResult { inner: Some(inner) })
    }
}

impl TryFrom<RawSubmitTxResult> for SubmitTxResult {
    type Error = crate::error::SomaError;

    fn try_from(value: RawSubmitTxResult) -> Result<Self, Self::Error> {
        match value.inner {
            Some(RawValidatorSubmitStatus::Submitted(consensus_position)) => {
                Ok(SubmitTxResult::Submitted {
                    consensus_position: consensus_position.as_ref().try_into()?,
                })
            }
            Some(RawValidatorSubmitStatus::Executed(executed)) => {
                let (effects_digest, details, fast_path) = try_from_raw_executed_status(executed)?;
                Ok(SubmitTxResult::Executed { effects_digest, details, fast_path })
            }
            Some(RawValidatorSubmitStatus::Rejected(error)) => {
                let error = try_from_raw_rejected_status(error)?.unwrap_or(
                    crate::error::SomaError::GrpcMessageDeserializeError {
                        type_info: "RawSubmitTxResult.inner.Error".to_string(),
                        error: "RawSubmitTxResult.inner.Error is None".to_string(),
                    },
                );
                Ok(SubmitTxResult::Rejected { error })
            }
            None => Err(crate::error::SomaError::GrpcMessageDeserializeError {
                type_info: "RawSubmitTxResult.inner".to_string(),
                error: "RawSubmitTxResult.inner is None".to_string(),
            }),
        }
    }
}

impl TryFrom<RawSubmitTxResponse> for SubmitTxResponse {
    type Error = crate::error::SomaError;

    fn try_from(value: RawSubmitTxResponse) -> Result<Self, Self::Error> {
        // TODO(fastpath): handle multiple transactions.
        if value.results.len() != 1 {
            return Err(crate::error::SomaError::GrpcMessageDeserializeError {
                type_info: "RawSubmitTxResponse.results".to_string(),
                error: format!("Expected exactly 1 result, got {}", value.results.len()),
            });
        }

        let results = value
            .results
            .into_iter()
            .map(|result| result.try_into())
            .collect::<Result<Vec<SubmitTxResult>, crate::error::SomaError>>()?;

        Ok(Self { results })
    }
}

fn try_from_raw_executed_status(
    executed: RawExecutedStatus,
) -> Result<
    (crate::digests::TransactionEffectsDigest, Option<Box<ExecutedData>>, bool),
    crate::error::SomaError,
> {
    let effects_digest = bcs::from_bytes(&executed.effects_digest).map_err(|err| {
        crate::error::SomaError::GrpcMessageDeserializeError {
            type_info: "RawWaitForEffectsResponse.effects_digest".to_string(),
            error: err.to_string(),
        }
    })?;
    let executed_data = if let Some(details) = executed.details {
        Some(Box::new(details.try_into()?))
    } else {
        None
    };
    Ok((effects_digest, executed_data, executed.fast_path))
}

fn try_from_raw_rejected_status(
    rejected: RawRejectedStatus,
) -> Result<Option<crate::error::SomaError>, crate::error::SomaError> {
    match rejected.error {
        Some(error_bytes) => {
            let error = bcs::from_bytes(&error_bytes).map_err(|err| {
                crate::error::SomaError::GrpcMessageDeserializeError {
                    type_info: "RawWaitForEffectsResponse.rejected.reason".to_string(),
                    error: err.to_string(),
                }
            })?;
            Ok(Some(error))
        }
        None => Ok(None),
    }
}

fn try_from_response_rejected(
    error: Option<crate::error::SomaError>,
) -> Result<RawRejectedStatus, crate::error::SomaError> {
    let error = match error {
        Some(e) => Some(
            bcs::to_bytes(&e)
                .map_err(|err| crate::error::SomaError::GrpcMessageSerializeError {
                    type_info: "RawRejectedStatus.error".to_string(),
                    error: err.to_string(),
                })?
                .into(),
        ),
        None => None,
    };
    Ok(RawRejectedStatus { error })
}

fn try_from_response_executed(
    effects_digest: crate::digests::TransactionEffectsDigest,
    details: Option<Box<ExecutedData>>,
    fast_path: bool,
) -> Result<RawExecutedStatus, crate::error::SomaError> {
    let effects_digest = bcs::to_bytes(&effects_digest)
        .map_err(|err| crate::error::SomaError::GrpcMessageSerializeError {
            type_info: "RawWaitForEffectsResponse.effects_digest".to_string(),
            error: err.to_string(),
        })?
        .into();
    let details = if let Some(details) = details { Some((*details).try_into()?) } else { None };
    Ok(RawExecutedStatus { effects_digest, details, fast_path })
}

impl TryFrom<RawWaitForEffectsRequest> for WaitForEffectsRequest {
    type Error = crate::error::SomaError;

    fn try_from(value: RawWaitForEffectsRequest) -> Result<Self, Self::Error> {
        let transaction_digest = match value.transaction_digest {
            Some(digest) => Some(bcs::from_bytes(&digest).map_err(|err| {
                crate::error::SomaError::GrpcMessageDeserializeError {
                    type_info: "RawWaitForEffectsRequest.transaction_digest".to_string(),
                    error: err.to_string(),
                }
            })?),
            None => None,
        };
        let consensus_position = match value.consensus_position {
            Some(cp) => Some(cp.as_ref().try_into()?),
            None => None,
        };
        let ping_type = value
            .ping_type
            .map(|p| {
                PingType::try_from(p).map_err(|e| SomaError::GrpcMessageDeserializeError {
                    type_info: "RawWaitForEffectsRequest.ping_type".to_string(),
                    error: e.to_string(),
                })
            })
            .transpose()?;
        Ok(Self {
            consensus_position,
            transaction_digest,
            include_details: value.include_details,
            ping_type,
        })
    }
}

impl TryFrom<WaitForEffectsRequest> for RawWaitForEffectsRequest {
    type Error = crate::error::SomaError;

    fn try_from(value: WaitForEffectsRequest) -> Result<Self, Self::Error> {
        let transaction_digest = match value.transaction_digest {
            Some(digest) => Some(
                bcs::to_bytes(&digest)
                    .map_err(|err| crate::error::SomaError::GrpcMessageSerializeError {
                        type_info: "RawWaitForEffectsRequest.transaction_digest".to_string(),
                        error: err.to_string(),
                    })?
                    .into(),
            ),
            None => None,
        };
        let consensus_position = match value.consensus_position {
            Some(cp) => Some(cp.into_raw()?),
            None => None,
        };
        let ping_type = value.ping_type.map(|p| p.into());
        Ok(Self {
            consensus_position,
            transaction_digest,
            include_details: value.include_details,
            ping_type,
        })
    }
}

impl TryFrom<RawWaitForEffectsResponse> for WaitForEffectsResponse {
    type Error = crate::error::SomaError;

    fn try_from(value: RawWaitForEffectsResponse) -> Result<Self, Self::Error> {
        match value.inner {
            Some(RawValidatorTransactionStatus::Executed(executed)) => {
                let (effects_digest, details, fast_path) = try_from_raw_executed_status(executed)?;
                Ok(Self::Executed { effects_digest, details, fast_path })
            }
            Some(RawValidatorTransactionStatus::Rejected(rejected)) => {
                let error = try_from_raw_rejected_status(rejected)?;
                Ok(Self::Rejected { error })
            }
            Some(RawValidatorTransactionStatus::Expired(expired)) => {
                Ok(Self::Expired { epoch: expired.epoch, round: expired.round })
            }
            None => Err(crate::error::SomaError::GrpcMessageDeserializeError {
                type_info: "RawWaitForEffectsResponse.inner".to_string(),
                error: "RawWaitForEffectsResponse.inner is None".to_string(),
            }),
        }
    }
}

impl TryFrom<WaitForEffectsResponse> for RawWaitForEffectsResponse {
    type Error = crate::error::SomaError;

    fn try_from(value: WaitForEffectsResponse) -> Result<Self, Self::Error> {
        let inner = match value {
            WaitForEffectsResponse::Executed { effects_digest, details, fast_path } => {
                let raw_executed = try_from_response_executed(effects_digest, details, fast_path)?;
                RawValidatorTransactionStatus::Executed(raw_executed)
            }
            WaitForEffectsResponse::Rejected { error } => {
                let raw_rejected = try_from_response_rejected(error)?;
                RawValidatorTransactionStatus::Rejected(raw_rejected)
            }
            WaitForEffectsResponse::Expired { epoch, round } => {
                RawValidatorTransactionStatus::Expired(RawExpiredStatus { epoch, round })
            }
        };
        Ok(RawWaitForEffectsResponse { inner: Some(inner) })
    }
}

impl TryFrom<ValidatorHealthRequest> for RawValidatorHealthRequest {
    type Error = crate::error::SomaError;

    fn try_from(_value: ValidatorHealthRequest) -> Result<Self, Self::Error> {
        Ok(Self {})
    }
}

impl TryFrom<RawValidatorHealthRequest> for ValidatorHealthRequest {
    type Error = crate::error::SomaError;

    fn try_from(_value: RawValidatorHealthRequest) -> Result<Self, Self::Error> {
        // Empty request - ignore reserved field for now
        Ok(Self {})
    }
}

impl TryFrom<ValidatorHealthResponse> for RawValidatorHealthResponse {
    type Error = crate::error::SomaError;

    fn try_from(value: ValidatorHealthResponse) -> Result<Self, Self::Error> {
        Ok(Self {
            // pending_certificates: Some(value.num_inflight_execution_transactions),
            inflight_consensus_messages: Some(value.num_inflight_consensus_transactions),
            consensus_round: Some(value.last_committed_leader_round as u64),
            checkpoint_sequence: Some(value.last_locally_built_checkpoint),
        })
    }
}

impl TryFrom<RawValidatorHealthResponse> for ValidatorHealthResponse {
    type Error = crate::error::SomaError;

    fn try_from(value: RawValidatorHealthResponse) -> Result<Self, Self::Error> {
        Ok(Self {
            num_inflight_consensus_transactions: value.inflight_consensus_messages.unwrap_or(0),
            // num_inflight_execution_transactions: value.pending_certificates.unwrap_or(0),
            last_locally_built_checkpoint: value.checkpoint_sequence.unwrap_or(0),
            last_committed_leader_round: value.consensus_round.unwrap_or(0) as u32,
        })
    }
}
