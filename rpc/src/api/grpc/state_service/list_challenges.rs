// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! ListChallenges RPC handler - lists challenges with optional filtering.
//!
//! This implementation uses an index-based approach for efficient challenge listing.
//! The RpcIndexes trait provides a challenges_iter method that iterates through Challenge objects.

use bytes::Bytes;
use prost::Message;
use prost_types::FieldMask;
use types::object::ObjectID;
use types::storage::read_store::ChallengeInfo;

use crate::api::RpcService;
use crate::api::error::{Result, RpcError};
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::{Challenge, ErrorReason, ListChallengesRequest, ListChallengesResponse};
use crate::utils::field::{FieldMaskTree, FieldMaskUtil};

const MAX_PAGE_SIZE: usize = 1000;
const DEFAULT_PAGE_SIZE: usize = 50;
const MAX_PAGE_SIZE_BYTES: usize = 512 * 1024; // 512KiB

/// Default fields to return if no read_mask is specified.
pub const READ_MASK_DEFAULT: &str = "id,target_id,challenger,status,challenge_epoch";

#[tracing::instrument(skip(service))]
pub fn list_challenges(
    service: &RpcService,
    request: ListChallengesRequest,
) -> Result<ListChallengesResponse> {
    // Validate status filter if provided
    let status_filter = request
        .status_filter
        .as_ref()
        .map(|s| match s.to_lowercase().as_str() {
            "pending" | "resolved" => Ok(s.to_lowercase()),
            _ => Err(FieldViolation::new("status_filter")
                .with_description(format!(
                    "invalid status_filter: '{}'. Must be 'pending' or 'resolved'",
                    s
                ))
                .with_reason(ErrorReason::FieldInvalid)),
        })
        .transpose()?;

    let epoch_filter = request.epoch_filter;

    // Parse target_id filter if provided
    let target_filter = request
        .target_id
        .as_ref()
        .map(|id| {
            id.parse::<ObjectID>().map_err(|e| {
                FieldViolation::new("target_id")
                    .with_description(format!("invalid target_id: {e}"))
                    .with_reason(ErrorReason::FieldInvalid)
            })
        })
        .transpose()?;

    let page_size = request
        .page_size
        .map(|s| (s as usize).clamp(1, MAX_PAGE_SIZE))
        .unwrap_or(DEFAULT_PAGE_SIZE);

    let page_token =
        request.page_token.as_ref().map(|token| decode_page_token(token)).transpose()?;

    // Validate page token parameters match request
    if let Some(ref token) = page_token {
        if token.status_filter != status_filter
            || token.epoch_filter != epoch_filter
            || token.target_filter != target_filter
        {
            return Err(FieldViolation::new("page_token")
                .with_description("page_token parameters do not match request filters")
                .with_reason(ErrorReason::FieldInvalid)
                .into());
        }
    }

    // Validate and build field mask
    let read_mask = {
        let read_mask = request.read_mask.unwrap_or_else(|| FieldMask::from_str(READ_MASK_DEFAULT));
        read_mask.validate::<Challenge>().map_err(|path| {
            FieldViolation::new("read_mask")
                .with_description(format!("invalid read_mask path: {path}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;
        FieldMaskTree::from(read_mask)
    };

    // Get indexes for challenge iteration
    let indexes = service.reader.inner().indexes().ok_or_else(RpcError::not_found)?;

    // Get the cursor from page token if present
    let cursor = page_token.as_ref().map(|t| t.cursor.clone());

    // Iterate through challenges using the index
    let mut iter = indexes
        .challenges_iter(status_filter.clone(), epoch_filter, target_filter, cursor)
        .map_err(|e| RpcError::new(tonic::Code::Internal, e.to_string()))?;

    let mut challenges = Vec::with_capacity(page_size);
    let mut size_bytes = 0;

    while let Some(challenge_info) =
        iter.next().transpose().map_err(|e| RpcError::new(tonic::Code::Internal, e.to_string()))?
    {
        // Load the full challenge object
        let Some(object) = service
            .reader
            .inner()
            .get_object_by_key(&challenge_info.challenge_id, challenge_info.version)
        else {
            tracing::debug!(
                "unable to find challenge {}:{} while iterating",
                challenge_info.challenge_id,
                challenge_info.version.value()
            );
            continue;
        };

        // Deserialize the challenge
        let challenge: types::challenge::ChallengeV1 = match bcs::from_bytes(object.data.contents())
        {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    "failed to deserialize challenge {}: {}",
                    challenge_info.challenge_id,
                    e
                );
                continue;
            }
        };

        // Convert to proto
        let challenge_proto =
            challenge_to_proto_with_id(&challenge_info.challenge_id, &challenge, &read_mask);

        size_bytes += challenge_proto.encoded_len();
        challenges.push(challenge_proto);

        if challenges.len() >= page_size || size_bytes >= MAX_PAGE_SIZE_BYTES {
            break;
        }
    }

    // Build next page token if there are more results
    let next_page_token = iter
        .next()
        .transpose()
        .map_err(|e| RpcError::new(tonic::Code::Internal, e.to_string()))?
        .map(|cursor_info| {
            encode_page_token(PageToken {
                status_filter: status_filter.clone(),
                epoch_filter,
                target_filter,
                cursor: cursor_info,
            })
        });

    let response = ListChallengesResponse { challenges, next_page_token, ..Default::default() };
    Ok(response)
}

/// Page token for ListChallenges pagination.
#[derive(serde::Serialize, serde::Deserialize)]
struct PageToken {
    status_filter: Option<String>,
    epoch_filter: Option<u64>,
    target_filter: Option<ObjectID>,
    cursor: ChallengeInfo,
}

fn decode_page_token(page_token: &[u8]) -> Result<PageToken> {
    bcs::from_bytes(page_token).map_err(|_| {
        FieldViolation::new("page_token")
            .with_description("invalid page_token")
            .with_reason(ErrorReason::FieldInvalid)
            .into()
    })
}

fn encode_page_token(page_token: PageToken) -> Bytes {
    bcs::to_bytes(&page_token).unwrap().into()
}

/// Convert a types::challenge::ChallengeV1 to proto Challenge with field mask.
fn challenge_to_proto_with_id(
    id: &ObjectID,
    challenge: &types::challenge::ChallengeV1,
    mask: &FieldMaskTree,
) -> Challenge {
    let mut proto = Challenge::default();

    if mask.contains("id") {
        proto.id = Some(id.to_string());
    }
    if mask.contains("target_id") {
        proto.target_id = Some(challenge.target_id.to_string());
    }
    if mask.contains("challenger") {
        proto.challenger = Some(challenge.challenger.to_string());
    }
    if mask.contains("challenger_bond") {
        proto.challenger_bond = Some(challenge.challenger_bond);
    }
    if mask.contains("challenge_epoch") {
        proto.challenge_epoch = Some(challenge.challenge_epoch);
    }
    if mask.contains("status") {
        proto.status = Some(format_status(&challenge.status));
    }
    // Simplified design: verdict is now part of status (challenger_lost: bool)
    if mask.contains("verdict") {
        if let types::challenge::ChallengeStatus::Resolved { challenger_lost } = &challenge.status {
            proto.verdict = Some(if *challenger_lost {
                "challenger_lost".to_string()
            } else {
                "challenger_won".to_string()
            });
        }
    }
    // win_reason is no longer applicable in simplified design
    if mask.contains("distance_threshold") {
        proto.distance_threshold = Some(challenge.distance_threshold.as_scalar());
    }
    if mask.contains("winning_distance_score") {
        proto.winning_distance_score = Some(challenge.winning_distance_score.as_scalar());
    }
    if mask.contains("winning_model_id") {
        proto.winning_model_id = Some(challenge.winning_model_id.to_string());
    }

    proto
}

fn format_status(status: &types::challenge::ChallengeStatus) -> String {
    match status {
        types::challenge::ChallengeStatus::Pending => "pending".to_string(),
        types::challenge::ChallengeStatus::Resolved { .. } => "resolved".to_string(),
    }
}
