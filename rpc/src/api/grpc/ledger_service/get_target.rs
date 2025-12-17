// crates/soma-rpc/src/api/ledger/get_target.rs

use crate::api::RpcService;
use crate::api::error::RpcError;
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::{
    ErrorReason, GetClaimableRewardsRequest, GetClaimableRewardsResponse, GetValidTargetsRequest,
    GetValidTargetsResponse, TargetInfo, TargetOriginType as ProtoOriginType,
};

use types::storage::read_store::{TargetIndexInfo, TargetOriginType};

// =============================================================================
// CONVERSIONS
// =============================================================================

impl From<TargetIndexInfo> for TargetInfo {
    fn from(info: TargetIndexInfo) -> Self {
        TargetInfo {
            target_id: Some(info.target_id.as_ref().to_vec().into()),
            created_epoch: Some(info.created_epoch),
            valid_epoch: Some(info.valid_epoch),
            origin: Some(match info.origin {
                TargetOriginType::System => ProtoOriginType::TargetOriginSystem as i32,
                TargetOriginType::User => ProtoOriginType::TargetOriginUser as i32,
                TargetOriginType::Genesis => ProtoOriginType::TargetOriginGenesis as i32,
            }),
            creator: info.creator.map(|c| c.as_ref().to_vec().into()),
            reward_amount: info.reward_amount,
            has_winner: Some(info.has_winner),
        }
    }
}

// =============================================================================
// PAGINATION HELPERS (shared with get_shard.rs, could be in a common module)
// =============================================================================

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
        None
    } else {
        Some(encode_cursor(end))
    };

    (paginated, next_cursor)
}

// =============================================================================
// HANDLERS
// =============================================================================

#[tracing::instrument(skip(service))]
pub fn get_valid_targets(
    service: &RpcService,
    request: GetValidTargetsRequest,
) -> Result<GetValidTargetsResponse, RpcError> {
    let epoch = request
        .epoch
        .ok_or_else(|| FieldViolation::new("epoch").with_reason(ErrorReason::FieldMissing))?;

    let indexes = service
        .reader
        .inner()
        .indexes()
        .ok_or_else(|| RpcError::new(tonic::Code::Unavailable, "Indexes not available"))?;

    let targets = indexes
        .get_valid_targets(epoch)
        .map_err(|e| RpcError::new(tonic::Code::Internal, format!("Query failed: {}", e)))?;

    let (targets, next_cursor) = paginate(
        targets,
        parse_cursor(&request.cursor.map(Into::into)),
        request.limit,
    );

    Ok(GetValidTargetsResponse {
        targets: targets.into_iter().map(Into::into).collect(),
        next_cursor: next_cursor.map(Into::into),
    })
}

#[tracing::instrument(skip(service))]
pub fn get_claimable_rewards(
    service: &RpcService,
    request: GetClaimableRewardsRequest,
) -> Result<GetClaimableRewardsResponse, RpcError> {
    let current_epoch = request.current_epoch.ok_or_else(|| {
        FieldViolation::new("current_epoch").with_reason(ErrorReason::FieldMissing)
    })?;

    let indexes = service
        .reader
        .inner()
        .indexes()
        .ok_or_else(|| RpcError::new(tonic::Code::Unavailable, "Indexes not available"))?;

    let targets = indexes
        .get_claimable_rewards(current_epoch)
        .map_err(|e| RpcError::new(tonic::Code::Internal, format!("Query failed: {}", e)))?;

    let (targets, next_cursor) = paginate(
        targets,
        parse_cursor(&request.cursor.map(Into::into)),
        request.limit,
    );

    Ok(GetClaimableRewardsResponse {
        targets: targets.into_iter().map(Into::into).collect(),
        next_cursor: next_cursor.map(Into::into),
    })
}
