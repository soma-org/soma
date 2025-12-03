use axum::{
    body::Body,
    extract::{Path, State},
    http::StatusCode,
    routing::get,
    Router,
};
use bytes::Bytes;
use fastcrypto::hash::HashFunction as _;
use object_store::{memory::InMemory, path::Path as ObjectPath, ObjectStore};
use std::{net::SocketAddr, sync::Arc};
use tokio::net::TcpListener;
use tracing::info;
use types::metadata::{
    DefaultDownloadMetadata, DefaultDownloadMetadataV1, DownloadMetadata, Metadata, MetadataV1,
    ObjectPath as SomaObjectPath,
};
use types::{checksum::Checksum, crypto::DefaultHash};
use url::Url;

/// Handle for the test object server
pub struct TestObjectServer {
    pub store: Arc<InMemory>,
    pub address: SocketAddr,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl TestObjectServer {
    pub async fn new() -> Self {
        let store = Arc::new(InMemory::new());
        let address = Self::start_server(store.clone()).await;

        Self {
            store,
            address,
            shutdown_tx: None,
        }
    }

    pub async fn new_with_shutdown() -> (Self, tokio::sync::oneshot::Receiver<()>) {
        let store = Arc::new(InMemory::new());
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let address = Self::start_server(store.clone()).await;

        let server = Self {
            store,
            address,
            shutdown_tx: Some(shutdown_tx),
        };

        (server, shutdown_rx)
    }

    async fn start_server(store: Arc<InMemory>) -> SocketAddr {
        let app = Router::new()
            .route("/uploads/{checksum}", get(serve_upload))
            .with_state(store);

        // Bind to a random available port
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        info!("Test object server started at {}", address);
        address
    }

    /// Upload data and return the download metadata
    pub async fn upload_data(&self, data: &[u8]) -> (Metadata, DownloadMetadata) {
        let mut h = DefaultHash::new();
        h.update(data);
        let checksum = Checksum::new_from_hash(h.finalize().into());
        let metadata = Metadata::V1(MetadataV1::new(checksum.clone(), data.len()));

        // Store the data
        let path = SomaObjectPath::Uploads(checksum.clone()).path();
        self.store
            .put(&path, Bytes::copy_from_slice(data).into())
            .await
            .expect("Failed to store data");

        // Create download URL
        let url = Url::parse(&format!("http://{}/uploads/{}", self.address, checksum))
            .expect("Failed to create URL");

        let download_metadata = DownloadMetadata::Default(DefaultDownloadMetadata::V1(
            DefaultDownloadMetadataV1::new(url, metadata.clone()),
        ));

        (metadata, download_metadata)
    }

    /// Get the base URL for this server
    pub fn base_url(&self) -> String {
        format!("http://{}", self.address)
    }
}

/// Axum handler for serving uploads
async fn serve_upload(
    State(store): State<Arc<InMemory>>,
    Path(checksum): Path<String>,
) -> Result<Body, StatusCode> {
    let path = ObjectPath::from(format!("uploads/{}", checksum));

    match store.get(&path).await {
        Ok(result) => {
            let bytes = result
                .bytes()
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Ok(Body::from(bytes))
        }
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}
