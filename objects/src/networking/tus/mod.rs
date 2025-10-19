use std::time::{Duration, Instant};

use async_trait::async_trait;
use axum::{
    extract::{Path, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{head, options},
    Router,
};
use soma_http::ServerHandle;
use tracing::{info, warn};
use types::{
    error::{ObjectError, ObjectResult},
    multiaddr::Multiaddr,
    p2p::to_socket_addr,
};
use uuid::Uuid;

const UPLOAD_OFFSET: &str = "upload-offset";
const UPLOAD_LENGTH: &str = "upload-length";
const TUS_VERSION: &str = "tus-version";
const TUS_RESUMABLE: &str = "tus-resumable";
const TUS_EXTENSION: &str = "tus-extension";
const TUS_MAX_SIZE: &str = "tus-max-size";

const SOMA_SUPPORTED_VERSION: &str = "1.0.0";
// X-HTTP-Method-Override is unsupported

pub struct TusManager {
    server: Option<ServerHandle>,
}

impl TusManager {
    pub fn new() -> Self {
        Self { server: None }
    }

    pub async fn start<S: TusState + Clone>(
        &mut self,
        state: S,
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
}

#[async_trait]
pub trait TusState: Send + Sync + Sized + 'static {
    async fn update(&self, uuid: Uuid, state: UploadState);
    async fn get(&self, uuid: Uuid) -> Option<UploadState>;
}

#[derive(Clone)]
struct Tus<S: TusState> {
    supported_versions: HeaderValue,
    server_version: HeaderValue,
    max_size: Option<HeaderValue>,
    extensions: HeaderValue,
    state: S,
}

impl<S: TusState + Clone> Tus<S> {
    fn new(state: S, max_size: Option<u64>) -> ObjectResult<Self> {
        let version = HeaderValue::from_static(SOMA_SUPPORTED_VERSION);
        Ok(Self {
            supported_versions: version.clone(),
            server_version: version,
            max_size: max_size
                .map(|size| HeaderValue::from_str(&size.to_string()))
                .transpose()
                .map_err(|e| ObjectError::TusError(e.to_string()))?,
            extensions: HeaderValue::from_static("creation,expiration"),
            state,
        })
    }
    fn router(self) -> Router {
        Router::new()
            .route(
                "/files/{uuid}",
                head(Self::head_handler).patch(Self::patch_handler),
            )
            .route("/files", options(Self::options_handler))
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
        }): State<Self>,
    ) -> impl IntoResponse {
        let mut headers = HeaderMap::new();
        headers.insert(TUS_RESUMABLE, server_version);
        headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        if Self::version_validation(&req_headers).is_err() {
            return (StatusCode::PRECONDITION_FAILED, headers);
        }
        match state.get(uuid).await {
            Some(state) => {
                headers.insert(UPLOAD_OFFSET, state.offset.to_string().parse().unwrap());
                headers.insert(UPLOAD_LENGTH, state.length.to_string().parse().unwrap());
                (StatusCode::NO_CONTENT, headers)
            }
            None => (StatusCode::NOT_FOUND, headers),
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
        }): State<Self>,
    ) -> impl IntoResponse {
        let mut headers = HeaderMap::new();
        headers.insert(TUS_RESUMABLE, server_version);
        if Self::version_validation(&req_headers).is_err() {
            return (StatusCode::PRECONDITION_FAILED, headers);
        }
        if Self::patch_content_type_validation(&req_headers).is_err() {
            return (StatusCode::UNSUPPORTED_MEDIA_TYPE, headers);
        }
        match state.get(uuid).await {
            Some(state) => {
                if Self::upload_state_validation(&state, &req_headers).is_err() {
                    return (StatusCode::CONFLICT, headers);
                }
                //
                (StatusCode::NO_CONTENT, headers)
            }

            None => (StatusCode::NOT_FOUND, headers),
        }
    }
    pub async fn options_handler(
        State(Self {
            supported_versions,
            server_version,
            extensions,
            max_size,
            state,
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
        if let Some(upload_length) = headers
            .get(UPLOAD_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
        {
            if upload_length == state.length {
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
