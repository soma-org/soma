use std::{str::FromStr, sync::Arc, time::Duration};

use crate::{
    error::{ShardError, ShardResult},
    storage::blob::BlobPath,
    types::network_committee::NetworkingIndex,
};
use async_trait::async_trait;
use axum::{http::Response, routing::get, Router};
use bytes::Bytes;
use reqwest::{Client, StatusCode};
use tokio::sync::oneshot;
use url::Url;

use super::{BlobNetworkClient, BlobNetworkManager, BlobNetworkService};

pub(crate) struct BlobHttpClient {
    client: Client,
}

impl BlobHttpClient {
    pub fn new() -> ShardResult<Self> {
        Ok(Self {
            client: Client::builder()
                .pool_idle_timeout(Duration::from_secs(90))
                .build()
                .map_err(|_| ShardError::FailedBuildingHttpClient)?,
        })
    }
}

#[async_trait]
impl BlobNetworkClient for BlobHttpClient {
    async fn get_object(
        &self,
        peer: NetworkingIndex,
        path: &BlobPath,
        timeout: Duration,
    ) -> ShardResult<Bytes> {
        // TODO: construct URL from peer and path
        let url = Url::from_str("https://github.com")
            .map_err(|e| ShardError::UrlParseError(e.to_string()))?;
        let response = self
            .client
            .get(url.clone())
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| ShardError::NetworkRequest(e.to_string()))?;
        Ok(response
            .bytes()
            .await
            .map_err(|e| ShardError::NetworkRequest(e.to_string()))?)
    }
}

pub struct BlobHttpManager {
    client: Arc<BlobHttpClient>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl BlobHttpManager {
    pub fn new() -> ShardResult<Self> {
        Ok(Self {
            client: Arc::new(BlobHttpClient::new()?),
            shutdown_tx: None,
        })
    }
}



struct BlobHttpServiceProxy<S: BlobNetworkService> {
    service: Arc<S>,
}

impl<S: BlobNetworkService> BlobHttpServiceProxy<S> {
    /// Creates the tonic service proxy using pre-established context and service
    const fn new(service: Arc<S>) -> Self {
        Self { service }
    }

    fn router(&self) -> Router {
        Router::new()
        // .route("/", get(Self::get_object()))
    }
}

impl<S: BlobNetworkService> BlobHttpServiceProxy<S> {
    fn get_object(&self) -> Result<Response, StatusCode>  {

        self.service.handle_get_object(peer, path)

        let response = Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/octet-stream")
        .body(axum::body::Full::from(bytes))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;


    }
}

impl BlobNetworkManager for BlobHttpManager {
    type Client = BlobHttpClient;

    fn new() -> Self {
        Self::new().unwrap()
    }

    fn client(&self) -> Arc<Self::Client> {
        self.client.clone()
    }

    async fn start(&mut self) {
        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);
        // TODO: fix to include context and look this up via that rather than hard coding
        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        axum::serve(listener, app)
        .with_graceful_shutdown(async {
            rx.await.ok();
        })
        .await
        .unwrap();

    }

    async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }

}
