// crates/soma-rpc/src/api/ledger/get_shard.rs

use crate::api::RpcService;
use crate::api::error::RpcError;
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::{
    ErrorReason, GetClaimableEscrowsRequest, GetClaimableEscrowsResponse,
    GetShardsByEncoderRequest, GetShardsByEncoderResponse, GetShardsByEpochRequest,
    GetShardsByEpochResponse, GetShardsBySubmitterRequest, GetShardsBySubmitterResponse,
    ObjectReference, ShardInfo,
};

use types::base::SomaAddress;
use types::shard_crypto::keys::EncoderPublicKey;
use types::storage::read_store::ShardIndexInfo;

// =============================================================================
// CONVERSIONS
// =============================================================================

impl From<ShardIndexInfo> for ShardInfo {
    fn from(info: ShardIndexInfo) -> Self {
        ShardInfo {
            shard_id: Some(info.shard_id.as_ref().to_vec().into()),
            created_epoch: Some(info.created_epoch),
            amount: Some(info.amount),
            data_submitter: Some(info.data_submitter.as_ref().to_vec().into()),
            target: info.target.map(|t| ObjectReference {
                object_id: Some(t.0.to_string()),
                version: Some(t.1.value()),
                digest: Some(t.2.to_string()),
            }),
            has_winner: Some(info.has_winner),
            winning_encoder: info.winning_encoder.map(|e| e.to_bytes().to_vec().into()),
        }
    }
}

// =============================================================================
// PARSING HELPERS
// =============================================================================

fn parse_address(bytes: &[u8], field_name: &str) -> Result<SomaAddress, RpcError> {
    SomaAddress::try_from(bytes).map_err(|_| {
        FieldViolation::new(field_name)
            .with_description("invalid address")
            .with_reason(ErrorReason::FieldInvalid)
            .into()
    })
}

fn parse_encoder_pubkey(bytes: &[u8]) -> Result<EncoderPublicKey, RpcError> {
    EncoderPublicKey::from_bytes(bytes).map_err(|_| {
        FieldViolation::new("encoder")
            .with_description("invalid encoder public key")
            .with_reason(ErrorReason::FieldInvalid)
            .into()
    })
}

fn parse_cursor(cursor: &Option<Vec<u8>>) -> Option<usize> {
    cursor.as_ref().and_then(|c| {
        if c.len() >= 8 {
            let bytes: [u8; 8] = c[..8].try_into().ok()?;
            Some(u64::from_le_bytes(bytes) as usize)
        } else {
            None
        }
    })
}

fn encode_cursor(offset: usize) -> Vec<u8> {
    (offset as u64).to_le_bytes().to_vec()
}

/// Paginate a vector of items
fn paginate<T>(
    items: Vec<T>,
    cursor: Option<usize>,
    limit: Option<u32>,
) -> (Vec<T>, Option<Vec<u8>>) {
    let start = cursor.unwrap_or(0);
    let limit = limit.map(|l| l as usize).unwrap_or(100).min(1000);

    if start >= items.len() {
        return (Vec::new(), None);
    }

    let end = (start + limit).min(items.len());
    let paginated: Vec<T> = items.into_iter().skip(start).take(limit).collect();

    let next_cursor = if end < start + limit {
        None // No more items
    } else {
        Some(encode_cursor(end))
    };

    (paginated, next_cursor)
}

// =============================================================================
// HANDLERS
// =============================================================================

#[tracing::instrument(skip(service))]
pub fn get_shards_by_epoch(
    service: &RpcService,
    request: GetShardsByEpochRequest,
) -> Result<GetShardsByEpochResponse, RpcError> {
    let epoch = request
        .epoch
        .ok_or_else(|| FieldViolation::new("epoch").with_reason(ErrorReason::FieldMissing))?;

    let indexes = service
        .reader
        .inner()
        .indexes()
        .ok_or_else(|| RpcError::new(tonic::Code::Unavailable, "Indexes not available"))?;

    let shards = indexes
        .get_shards_by_epoch(epoch)
        .map_err(|e| RpcError::new(tonic::Code::Internal, format!("Query failed: {}", e)))?;

    let (shards, next_cursor) = paginate(
        shards,
        parse_cursor(&request.cursor.map(Into::into)),
        request.limit,
    );

    Ok(GetShardsByEpochResponse {
        shards: shards.into_iter().map(Into::into).collect(),
        next_cursor: next_cursor.map(Into::into),
    })
}

#[tracing::instrument(skip(service))]
pub fn get_shards_by_submitter(
    service: &RpcService,
    request: GetShardsBySubmitterRequest,
) -> Result<GetShardsBySubmitterResponse, RpcError> {
    let submitter_bytes = request
        .submitter
        .as_ref()
        .ok_or_else(|| FieldViolation::new("submitter").with_reason(ErrorReason::FieldMissing))?;
    let submitter = parse_address(submitter_bytes, "submitter")?;

    let indexes = service
        .reader
        .inner()
        .indexes()
        .ok_or_else(|| RpcError::new(tonic::Code::Unavailable, "Indexes not available"))?;

    let shards = indexes
        .get_shards_by_submitter(submitter, request.epoch)
        .map_err(|e| RpcError::new(tonic::Code::Internal, format!("Query failed: {}", e)))?;

    let (shards, next_cursor) = paginate(
        shards,
        parse_cursor(&request.cursor.map(Into::into)),
        request.limit,
    );

    Ok(GetShardsBySubmitterResponse {
        shards: shards.into_iter().map(Into::into).collect(),
        next_cursor: next_cursor.map(Into::into),
    })
}

#[tracing::instrument(skip(service))]
pub fn get_shards_by_encoder(
    service: &RpcService,
    request: GetShardsByEncoderRequest,
) -> Result<GetShardsByEncoderResponse, RpcError> {
    let encoder_bytes = request
        .encoder
        .as_ref()
        .ok_or_else(|| FieldViolation::new("encoder").with_reason(ErrorReason::FieldMissing))?;
    let encoder = parse_encoder_pubkey(encoder_bytes)?;

    let indexes = service
        .reader
        .inner()
        .indexes()
        .ok_or_else(|| RpcError::new(tonic::Code::Unavailable, "Indexes not available"))?;

    let shards = indexes
        .get_shards_by_encoder(&encoder)
        .map_err(|e| RpcError::new(tonic::Code::Internal, format!("Query failed: {}", e)))?;

    let (shards, next_cursor) = paginate(
        shards,
        parse_cursor(&request.cursor.map(Into::into)),
        request.limit,
    );

    Ok(GetShardsByEncoderResponse {
        shards: shards.into_iter().map(Into::into).collect(),
        next_cursor: next_cursor.map(Into::into),
    })
}

#[tracing::instrument(skip(service))]
pub fn get_claimable_escrows(
    service: &RpcService,
    request: GetClaimableEscrowsRequest,
) -> Result<GetClaimableEscrowsResponse, RpcError> {
    let current_epoch = request.current_epoch.ok_or_else(|| {
        FieldViolation::new("current_epoch").with_reason(ErrorReason::FieldMissing)
    })?;

    let indexes = service
        .reader
        .inner()
        .indexes()
        .ok_or_else(|| RpcError::new(tonic::Code::Unavailable, "Indexes not available"))?;

    let shards = indexes
        .get_claimable_escrows(current_epoch)
        .map_err(|e| RpcError::new(tonic::Code::Internal, format!("Query failed: {}", e)))?;

    let (shards, next_cursor) = paginate(
        shards,
        parse_cursor(&request.cursor.map(Into::into)),
        request.limit,
    );

    Ok(GetClaimableEscrowsResponse {
        shards: shards.into_iter().map(Into::into).collect(),
        next_cursor: next_cursor.map(Into::into),
    })
}
