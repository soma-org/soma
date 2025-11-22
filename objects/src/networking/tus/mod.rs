use crate::networking::{
    tus::headers::{
        LOCATION, TUS_EXTENSION, TUS_MAX_SIZE, TUS_RESUMABLE, TUS_VERSION, UPLOAD_LENGTH,
        UPLOAD_OFFSET,
    },
    MAX_PART_SIZE, MIN_PART_SIZE,
};
use axum::{
    body::{to_bytes, Body},
    extract::{Path, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{head, options},
    Router,
};
use bytes::Bytes;
use fastcrypto::hash::HashFunction;
use object_store::{MultipartUpload, ObjectStore, PutPayload};
use soma_http::ServerHandle;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};
use types::{
    checksum::Checksum,
    crypto::DefaultHash,
    error::{ObjectError, ObjectResult},
    multiaddr::Multiaddr,
    sync::to_socket_addr,
};
use uuid::Uuid;

pub mod client;
mod headers;

const SOMA_SUPPORTED_VERSION: &str = "1.0.0";

pub struct TusManager {
    server: Option<ServerHandle>,
}

impl TusManager {
    pub fn new() -> Self {
        Self { server: None }
    }

    pub async fn start<S: ObjectStore>(
        &mut self,
        state: Arc<S>,
        address: &Multiaddr,
        max_size: Option<u64>,
    ) -> ObjectResult<()> {
        let own_address = if address.is_localhost_ip() {
            address.clone()
        } else {
            address.with_zero_ip()
        };
        let own_address = to_socket_addr(&own_address).unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        let server = loop {
            match soma_http::Builder::new()
                .serve(own_address, Tus::new(state.clone(), max_size)?.router())
            {
                Ok(server) => break server,
                Err(err) => {
                    warn!("Error starting server: {err:?}");
                    if Instant::now() > deadline {
                        panic!("Failed to start server within required deadline");
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        };
        info!("Tus server started at: {own_address}");
        self.server = Some(server);
        Ok(())
    }

    async fn stop(&mut self) {
        if let Some(server) = self.server.take() {
            server.shutdown().await;
        }
    }
}

impl Drop for TusManager {
    fn drop(&mut self) {
        if let Some(server) = self.server.as_ref() {
            server.trigger_shutdown();
        }
    }
}

#[derive(Clone)]
pub struct UploadState {
    offset: u64,
    length: u64,
    multipart: Arc<Mutex<Box<dyn MultipartUpload>>>,
    hasher: Option<Arc<Mutex<DefaultHash>>>,
    checksum: Option<Checksum>,
}

struct Tus<S: ObjectStore> {
    supported_versions: HeaderValue,
    server_version: HeaderValue,
    max_size: Option<HeaderValue>,
    extensions: HeaderValue,
    state: Arc<RwLock<HashMap<Uuid, UploadState>>>,
    storage: Arc<S>,
}

impl<S: ObjectStore> Clone for Tus<S> {
    fn clone(&self) -> Self {
        Self {
            supported_versions: self.supported_versions.clone(),
            server_version: self.server_version.clone(),
            max_size: self.max_size.clone(),
            extensions: self.extensions.clone(),
            state: self.state.clone(),
            storage: self.storage.clone(),
        }
    }
}

impl<S: ObjectStore> Tus<S> {
    fn new(storage: Arc<S>, max_size: Option<u64>) -> ObjectResult<Self> {
        let version = HeaderValue::from_static(SOMA_SUPPORTED_VERSION);
        Ok(Self {
            supported_versions: version.clone(),
            server_version: version,
            max_size: max_size
                .map(|size| HeaderValue::from_str(&size.to_string()))
                .transpose()
                .map_err(|e| ObjectError::TusError(e.to_string()))?,
            extensions: HeaderValue::from_static("creation,expiration"),
            state: Arc::new(RwLock::new(HashMap::new())),
            storage,
        })
    }

    fn router(self) -> Router {
        Router::new()
            .route(
                "/tus/{uuid}",
                head(Self::head_handler).patch(Self::patch_handler),
            )
            .route("/tus", options(Self::options_handler))
            .with_state(self)
    }

    pub async fn head_handler(
        Path(uuid): Path<Uuid>,
        req_headers: HeaderMap,
        State(Self {
            supported_versions,
            server_version,
            extensions,
            max_size,
            state,
            storage,
        }): State<Self>,
    ) -> impl IntoResponse {
        let mut headers = HeaderMap::new();
        headers.insert(TUS_RESUMABLE, server_version);
        headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        if Self::version_validation(&req_headers).is_err() {
            return (StatusCode::PRECONDITION_FAILED, headers);
        }
        match state.read().await.get(&uuid) {
            Some(state) => {
                headers.insert(UPLOAD_OFFSET, state.offset.to_string().parse().unwrap());
                headers.insert(UPLOAD_LENGTH, state.length.to_string().parse().unwrap());
                (StatusCode::NO_CONTENT, headers)
            }
            None => (StatusCode::NOT_FOUND, headers),
        }
    }

    pub async fn post_handler(
        req_headers: HeaderMap,
        State(Self {
            supported_versions,
            server_version,
            extensions,
            max_size,
            state,
            storage,
        }): State<Self>,
    ) -> impl IntoResponse {
        let mut headers = HeaderMap::new();
        headers.insert(TUS_RESUMABLE, server_version);
        if Self::version_validation(&req_headers).is_err() {
            return (StatusCode::PRECONDITION_FAILED, headers);
        }

        if let Some(upload_length) = headers
            .get(UPLOAD_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
        {
            if let Some(max_size) = max_size
                .as_ref()
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok())
            {
                if upload_length > max_size {
                    return (StatusCode::PAYLOAD_TOO_LARGE, headers);
                }
            }
            let uuid = Uuid::new_v4();
            let multipart = match storage
                .put_multipart(&object_store::path::Path::from(format!("/tus/{}", uuid)))
                .await
            {
                Ok(m) => m,
                Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, headers),
            };
            state.write().await.insert(
                uuid,
                UploadState {
                    offset: 0,
                    length: upload_length,
                    multipart: Arc::new(Mutex::new(multipart)),
                    hasher: Some(Arc::new(Mutex::new(DefaultHash::new()))),
                    checksum: None,
                },
            );
            let location_header = match HeaderValue::from_str(&format!("/{}", uuid)) {
                Ok(h) => h,
                Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, headers),
            };
            headers.insert(LOCATION, location_header);
            (StatusCode::CREATED, headers)
        } else {
            return (StatusCode::PRECONDITION_FAILED, headers);
        }
    }

    pub async fn patch_handler(
        Path(uuid): Path<Uuid>,
        req_headers: HeaderMap,
        State(Self {
            supported_versions,
            server_version,
            extensions,
            max_size,
            state,
            storage,
        }): State<Self>,
        body: Body,
    ) -> impl IntoResponse {
        let mut headers = HeaderMap::new();
        headers.insert(TUS_RESUMABLE, server_version);
        if Self::version_validation(&req_headers).is_err() {
            return (StatusCode::PRECONDITION_FAILED, headers);
        }
        if Self::patch_content_type_validation(&req_headers).is_err() {
            return (StatusCode::UNSUPPORTED_MEDIA_TYPE, headers);
        }
        let mut state_opt = match state.read().await.get(&uuid) {
            Some(s) => s.clone(),
            None => return (StatusCode::NOT_FOUND, headers),
        };
        if Self::upload_state_validation(&state_opt, &req_headers).is_err() {
            return (StatusCode::CONFLICT, headers);
        }
        // Read the request body
        let bytes: Bytes = match to_bytes(body, usize::MAX).await {
            Ok(b) => b,
            Err(_) => {
                warn!("Failed to read request body");
                return (StatusCode::INTERNAL_SERVER_ERROR, headers);
            }
        };
        let chunk_size: u64 = match bytes.len().try_into() {
            Ok(size) => size,
            Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, headers),
        };
        let new_offset = match state_opt.offset.checked_add(chunk_size) {
            Some(offset) => offset,
            None => return (StatusCode::PAYLOAD_TOO_LARGE, headers),
        };
        if let Some(hasher_arc) = &state_opt.hasher {
            let mut h = hasher_arc.lock().await;
            h.update(&bytes);
        } else {
            // Defensive: Should not happen for non-final chunks
            warn!("Hasher missing for in-progress upload {}", uuid);
            return (StatusCode::INTERNAL_SERVER_ERROR, headers);
        }
        if new_offset > state_opt.length {
            return (StatusCode::PAYLOAD_TOO_LARGE, headers);
        }
        let is_final = new_offset == state_opt.length;
        if chunk_size > MAX_PART_SIZE {
            return (StatusCode::PAYLOAD_TOO_LARGE, headers);
        }
        if chunk_size < MIN_PART_SIZE && !is_final {
            return (StatusCode::PRECONDITION_FAILED, headers);
        }
        let mut checksum: Option<Checksum> = None;
        {
            let mut m = state_opt.multipart.lock().await;
            match m.put_part(PutPayload::from(bytes)).await {
                Ok(_) => (),
                Err(e) => {
                    warn!("Failed to upload part for upload {}: {:?}", uuid, e);
                    return (StatusCode::INTERNAL_SERVER_ERROR, headers);
                }
            }
            if is_final {
                let digest_output = if let Some(hasher_arc) = state_opt.hasher.take() {
                    let mut guard = hasher_arc.lock().await;
                    std::mem::take(&mut *guard).finalize()
                } else {
                    warn!("Hasher missing on final chunk for upload {}", uuid);
                    return (StatusCode::INTERNAL_SERVER_ERROR, headers);
                };
                checksum = Some(Checksum::new_from_hash(digest_output.digest));
                match m.complete().await {
                    Ok(_) => {}
                    Err(e) => {
                        warn!("Failed to complete multipart upload for {}: {:?}", uuid, e);
                        return (StatusCode::INTERNAL_SERVER_ERROR, headers);
                    }
                }
            }
        }
        // Update the state
        let new_state = UploadState {
            offset: new_offset,
            length: state_opt.length,
            multipart: state_opt.multipart,
            hasher: if is_final { None } else { state_opt.hasher },
            checksum,
        };
        state.write().await.insert(uuid, new_state);
        // Respond with new offset
        let offset_header = match HeaderValue::from_str(&new_offset.to_string()) {
            Ok(h) => h,
            Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, headers),
        };
        headers.insert(UPLOAD_OFFSET, offset_header);
        (StatusCode::NO_CONTENT, headers)
    }

    pub async fn options_handler(
        State(Self {
            supported_versions,
            server_version,
            extensions,
            max_size,
            state,
            storage,
        }): State<Self>,
    ) -> impl IntoResponse {
        let mut headers = HeaderMap::new();
        headers.insert(TUS_RESUMABLE, server_version);
        headers.insert(TUS_VERSION, supported_versions);
        if let Some(size) = max_size {
            headers.insert(TUS_MAX_SIZE, size);
        }
        headers.insert(TUS_EXTENSION, extensions);
        (StatusCode::NO_CONTENT, headers)
    }

    fn version_validation(headers: &HeaderMap) -> ObjectResult<()> {
        if let Some(client_version) = headers.get(TUS_RESUMABLE).and_then(|v| v.to_str().ok()) {
            if client_version == SOMA_SUPPORTED_VERSION {
                return Ok(());
            }
        }
        Err(ObjectError::TusError(
            "TUS client is unsupported".to_string(),
        ))
    }

    fn upload_state_validation(state: &UploadState, headers: &HeaderMap) -> ObjectResult<()> {
        if let Some(upload_offset) = headers
            .get(UPLOAD_OFFSET)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
        {
            if upload_offset == state.offset {
                return Ok(());
            }
        }
        Err(ObjectError::TusError(
            "patch does not match state".to_string(),
        ))
    }

    fn patch_content_type_validation(headers: &HeaderMap) -> ObjectResult<()> {
        if let Some(content_type) = headers
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
        {
            if content_type == "application/offset+octet-stream" {
                return Ok(());
            }
        }
        Err(ObjectError::TusError(
            "Patch content type is unsupported".to_string(),
        ))
    }
}
