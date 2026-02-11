// Copyright (c) Soma Foundation
// SPDX-License-Identifier: Apache-2.0

//! GetTarget RPC handler - loads a Target shared object by ID.

use crate::api::RpcService;
use crate::api::error::{ObjectNotFoundError, Result, RpcError};
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::target::target_to_proto_with_id;
use crate::proto::soma::{ErrorReason, GetTargetRequest, GetTargetResponse, Target};
use crate::types::Address;
use crate::utils::field::{FieldMaskTree, FieldMaskUtil};
use prost_types::FieldMask;
use types::object::ObjectID;

/// Default fields to return if no read_mask is specified.
pub const READ_MASK_DEFAULT: &str =
    "id,status,generation_epoch,reward_pool,distance_threshold";

#[tracing::instrument(skip(service))]
pub fn get_target(service: &RpcService, request: GetTargetRequest) -> Result<GetTargetResponse> {
    // Parse and validate target_id
    let target_id: Address = request
        .target_id
        .as_ref()
        .ok_or_else(|| {
            FieldViolation::new("target_id")
                .with_description("missing target_id")
                .with_reason(ErrorReason::FieldMissing)
        })?
        .parse()
        .map_err(|e| {
            FieldViolation::new("target_id")
                .with_description(format!("invalid target_id: {e}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;

    // Validate and build field mask
    let read_mask = {
        let read_mask = request.read_mask.unwrap_or_else(|| FieldMask::from_str(READ_MASK_DEFAULT));
        read_mask.validate::<Target>().map_err(|path| {
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
        .get_object(&target_id.into())
        .ok_or_else(|| ObjectNotFoundError::new(target_id))?;

    // Verify it's actually a Target object
    let object_type = object.data.object_type();
    if *object_type != types::object::ObjectType::Target {
        return Err(RpcError::new(
            tonic::Code::InvalidArgument,
            format!("Object {} is not a Target (type: {:?})", target_id, object_type),
        ));
    }

    // Deserialize the Target from the object contents
    let target: types::target::Target = bcs::from_bytes(object.data.contents()).map_err(|e| {
        RpcError::new(tonic::Code::Internal, format!("Failed to deserialize Target: {e}"))
    })?;

    // Convert to proto with field mask
    let object_id: ObjectID = target_id.into();
    let target_proto = target_to_proto_with_id(&object_id, &target, &read_mask);

    let response = GetTargetResponse {
        target: Some(target_proto),
        ..Default::default()
    };
    Ok(response)
}
