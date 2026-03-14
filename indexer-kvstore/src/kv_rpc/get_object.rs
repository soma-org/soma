// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use rpc::api::error::{ObjectNotFoundError, RpcError};
use rpc::proto::google::rpc::bad_request::FieldViolation;
use rpc::proto::soma::{
    BatchGetObjectsRequest, BatchGetObjectsResponse, ErrorReason, GetObjectRequest,
    GetObjectResponse, GetObjectResult, Object,
};
use rpc::types::Address;
use rpc::utils::field::{FieldMask, FieldMaskTree, FieldMaskUtil};
use rpc::utils::merge::Merge;

use crate::KeyValueStoreReader;

use super::KvRpcServer;

pub const READ_MASK_DEFAULT: &str = "object_id,version,digest";

pub async fn get_object(
    server: &KvRpcServer,
    GetObjectRequest { object_id, version, read_mask, .. }: GetObjectRequest,
) -> Result<GetObjectResponse, RpcError> {
    let read_mask = validate_read_mask(read_mask)?;

    let object_id_str = object_id.ok_or_else(|| {
        FieldViolation::new("object_id")
            .with_description("missing object_id")
            .with_reason(ErrorReason::FieldMissing)
    })?;
    let address: Address = object_id_str.parse().map_err(|e| {
        FieldViolation::new("object_id")
            .with_description(format!("invalid object_id: {e}"))
            .with_reason(ErrorReason::FieldInvalid)
    })?;

    let object = get_object_impl(server, address, version, &read_mask).await?;
    Ok(GetObjectResponse::new(object))
}

pub async fn batch_get_objects(
    server: &KvRpcServer,
    BatchGetObjectsRequest { requests, read_mask, .. }: BatchGetObjectsRequest,
) -> Result<BatchGetObjectsResponse, RpcError> {
    let read_mask = validate_read_mask(read_mask)?;

    let mut results = Vec::with_capacity(requests.len());
    for (idx, req) in requests.into_iter().enumerate() {
        let result = async {
            let address: Address = req
                .object_id
                .as_ref()
                .ok_or_else(|| {
                    FieldViolation::new("object_id")
                        .with_reason(ErrorReason::FieldMissing)
                        .nested_at("requests", idx)
                })?
                .parse()
                .map_err(|e| {
                    RpcError::from(
                        FieldViolation::new("object_id")
                            .with_description(format!("invalid object_id: {e}"))
                            .with_reason(ErrorReason::FieldInvalid)
                            .nested_at("requests", idx),
                    )
                })?;

            get_object_impl(server, address, req.version, &read_mask).await
        }
        .await;

        results.push(match result {
            Ok(object) => GetObjectResult::new_object(object),
            Err(error) => GetObjectResult::new_error(error.into_status_proto()),
        });
    }

    Ok(BatchGetObjectsResponse::new(results))
}

fn validate_read_mask(read_mask: Option<FieldMask>) -> Result<FieldMaskTree, RpcError> {
    let read_mask = read_mask.unwrap_or_else(|| FieldMask::from_str(READ_MASK_DEFAULT));
    read_mask.validate::<Object>().map_err(|path| {
        FieldViolation::new("read_mask")
            .with_description(format!("invalid read_mask path: {path}"))
            .with_reason(ErrorReason::FieldInvalid)
    })?;
    Ok(FieldMaskTree::from(read_mask))
}

async fn get_object_impl(
    server: &KvRpcServer,
    object_id: Address,
    version: Option<u64>,
    read_mask: &FieldMaskTree,
) -> Result<Object, RpcError> {
    let mut client = server.client.clone();
    let core_object_id: types::object::ObjectID = object_id.into();

    let core_object = if let Some(version) = version {
        let object_key =
            types::storage::ObjectKey(core_object_id, types::object::Version::from_u64(version));
        let mut results = client.get_objects(&[object_key]).await?;
        results.pop().ok_or_else(|| ObjectNotFoundError::new_with_version(object_id, version))?
    } else {
        client
            .get_latest_object(&core_object_id)
            .await?
            .ok_or_else(|| ObjectNotFoundError::new(object_id))?
    };

    // Convert core Object to SDK Object
    let sdk_object: rpc::types::Object = core_object.try_into().map_err(|e| {
        RpcError::new(rpc_tonic::Code::Internal, format!("failed to convert object: {e}"))
    })?;

    let mut message = Object::default();
    message.merge(sdk_object, read_mask);
    Ok(message)
}
