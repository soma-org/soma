// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! GetChallenge RPC handler - loads a Challenge shared object by ID.

use crate::api::RpcService;
use crate::api::error::{ObjectNotFoundError, Result, RpcError};
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::{Challenge, ErrorReason, GetChallengeRequest, GetChallengeResponse};
use crate::types::Address;
use crate::utils::field::{FieldMaskTree, FieldMaskUtil};
use prost_types::FieldMask;
use types::object::ObjectID;

/// Default fields to return if no read_mask is specified.
pub const READ_MASK_DEFAULT: &str =
    "id,target_id,challenger,status,challenge_epoch,challenger_bond";

#[tracing::instrument(skip(service))]
pub fn get_challenge(
    service: &RpcService,
    request: GetChallengeRequest,
) -> Result<GetChallengeResponse> {
    // Parse and validate challenge_id
    let challenge_id: Address = request
        .challenge_id
        .as_ref()
        .ok_or_else(|| {
            FieldViolation::new("challenge_id")
                .with_description("missing challenge_id")
                .with_reason(ErrorReason::FieldMissing)
        })?
        .parse()
        .map_err(|e| {
            FieldViolation::new("challenge_id")
                .with_description(format!("invalid challenge_id: {e}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;

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

    // Load the object from storage
    let object = service
        .reader
        .inner()
        .get_object(&challenge_id.into())
        .ok_or_else(|| ObjectNotFoundError::new(challenge_id))?;

    // Verify it's actually a Challenge object
    let object_type = object.data.object_type();
    if *object_type != types::object::ObjectType::Challenge {
        return Err(RpcError::new(
            tonic::Code::InvalidArgument,
            format!("Object {} is not a Challenge (type: {:?})", challenge_id, object_type),
        ));
    }

    // Deserialize the Challenge from the object contents
    let challenge: types::challenge::ChallengeV1 = bcs::from_bytes(object.data.contents())
        .map_err(|e| {
            RpcError::new(tonic::Code::Internal, format!("Failed to deserialize Challenge: {e}"))
        })?;

    // Convert to proto with field mask
    let object_id: ObjectID = challenge_id.into();
    let challenge_proto = challenge_to_proto_with_id(&object_id, &challenge, &read_mask);

    let response = GetChallengeResponse { challenge: Some(challenge_proto), ..Default::default() };
    Ok(response)
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
