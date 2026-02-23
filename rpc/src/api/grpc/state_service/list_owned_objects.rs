use crate::api::RpcService;
use crate::api::error::Result;
use crate::api::error::RpcError;
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::ErrorReason;
use crate::proto::soma::ListOwnedObjectsRequest;
use crate::proto::soma::ListOwnedObjectsResponse;
use crate::proto::soma::Object;
use crate::types::Address;
use crate::utils::field::FieldMaskTree;
use crate::utils::field::FieldMaskUtil;
use bytes::Bytes;
use prost::Message;
use prost_types::FieldMask;
use types::object::ObjectType;
use types::storage::read_store::OwnedObjectInfo;

const MAX_PAGE_SIZE: usize = 1000;
const DEFAULT_PAGE_SIZE: usize = 50;
const MAX_PAGE_SIZE_BYTES: usize = 512 * 1024; // 512KiB
const READ_MASK_DEFAULT: &str = "object_id,version,object_type";

#[tracing::instrument(skip(service))]
pub fn list_owned_objects(
    service: &RpcService,
    request: ListOwnedObjectsRequest,
) -> Result<ListOwnedObjectsResponse> {
    let indexes = service.reader.inner().indexes().ok_or_else(RpcError::not_found)?;

    let owner: Address = request
        .owner
        .as_ref()
        .ok_or_else(|| FieldViolation::new("owner").with_reason(ErrorReason::FieldMissing))?
        .parse()
        .map_err(|e| {
            FieldViolation::new("owner")
                .with_description(format!("invalid owner: {e}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;
    let object_type = request.object_type.map(|s| s.parse()).transpose().map_err(|e| {
        FieldViolation::new("object_type")
            .with_description(format!("invalid object_type: {e}"))
            .with_reason(ErrorReason::FieldInvalid)
    })?;

    let page_size = request
        .page_size
        .map(|s| (s as usize).clamp(1, MAX_PAGE_SIZE))
        .unwrap_or(DEFAULT_PAGE_SIZE);
    let page_token = request.page_token.map(|token| decode_page_token(&token)).transpose()?;
    if let Some(token) = &page_token {
        if token.owner != owner || token.object_type != object_type {
            return Err(FieldViolation::new("page_token")
                .with_description("invalid page_token")
                .with_reason(ErrorReason::FieldInvalid)
                .into());
        }
    }
    let read_mask = {
        let read_mask = request.read_mask.unwrap_or_else(|| FieldMask::from_str(READ_MASK_DEFAULT));
        read_mask.validate::<Object>().map_err(|path| {
            FieldViolation::new("read_mask")
                .with_description(format!("invalid read_mask path: {path}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;
        FieldMaskTree::from(read_mask)
    };

    let should_load_object = should_load_object(&read_mask);
    let mut iter = indexes.owned_objects_iter(
        owner.into(),
        object_type.clone(),
        page_token.map(|t| t.inner),
    )?;
    let mut objects = Vec::with_capacity(page_size);
    let mut size_bytes = 0;
    while let Some(object_info) =
        iter.next().transpose().map_err(|e| RpcError::new(tonic::Code::Internal, e.to_string()))?
    {
        let object = if should_load_object {
            let Some(object) = service
                .reader
                .inner()
                .get_object_by_key(&object_info.object_id, object_info.version)
            else {
                tracing::debug!(
                    "unable to find object {}:{} while iterating through owned objects",
                    object_info.object_id,
                    object_info.version.value()
                );
                continue;
            };

            let mut message = Object::default();

            // if read_mask.contains(Object::JSON_FIELD) {
            //     message.json =
            //         crate::api::grpc::render_object_to_json(service, &object).map(Box::new);
            // }
            crate::utils::merge::Merge::merge(&mut message, &object, &read_mask);
            message
        } else {
            owned_object_to_proto(object_info, &read_mask)
        };

        size_bytes += object.encoded_len();
        objects.push(object);

        if objects.len() >= page_size || size_bytes >= MAX_PAGE_SIZE_BYTES {
            break;
        }
    }

    let next_page_token = iter
        .next()
        .transpose()
        .map_err(|e| RpcError::new(tonic::Code::Internal, e.to_string()))?
        .map(|cursor| encode_page_token(PageToken { owner, object_type, inner: cursor }));

    let message = ListOwnedObjectsResponse { objects, next_page_token, ..Default::default() };
    Ok(message)
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

fn owned_object_to_proto(info: OwnedObjectInfo, mask: &FieldMaskTree) -> Object {
    let mut message = Object::default();

    if mask.contains(Object::OBJECT_ID_FIELD) {
        message.object_id = Some(info.object_id.to_hex());
    }
    if mask.contains(Object::VERSION_FIELD) {
        message.version = Some(info.version.value());
    }
    if mask.contains(Object::OBJECT_TYPE_FIELD) {
        message.object_type = Some(info.object_type.to_string());
    }

    message
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PageToken {
    owner: Address,
    object_type: Option<ObjectType>,
    inner: OwnedObjectInfo,
}

fn should_load_object(mask: &FieldMaskTree) -> bool {
    [
        Object::DIGEST_FIELD,
        Object::OWNER_FIELD,
        Object::CONTENTS_FIELD,
        Object::PREVIOUS_TRANSACTION_FIELD,
    ]
    .into_iter()
    .any(|field| mask.contains(field))
}
