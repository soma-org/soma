use std::{str::FromStr, sync::Arc, time::Duration};

use crate::{
    error::{ShardError, ShardResult},
    networking::messaging::{to_host_port_str, to_socket_addr},
    storage::blob::BlobPath,
    types::{
        context::{EncoderContext, NetworkingContext},
        network_committee::NetworkingIndex,
    },
};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::{Path, State},
    http::Response,
    response::IntoResponse,
    routing::get,
    Router,
};
use bytes::Bytes;
use reqwest::{Client, StatusCode};
use tokio::sync::oneshot;
use url::Url;

use super::{BlobNetworkClient, BlobNetworkManager, BlobNetworkService, GetObjectResponse};

pub(crate) struct BlobHttpClient {
    client: Client,
    context: Arc<EncoderContext>,
}

impl BlobHttpClient {
    pub fn new(context: Arc<EncoderContext>) -> ShardResult<Self> {
        Ok(Self {
            client: Client::builder()
                .pool_idle_timeout(Duration::from_secs(60 * 5))
                .build()
                .map_err(|_| ShardError::FailedBuildingHttpClient)?,

            context,
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
        let network_identity = self.context.network_committee().identity(peer);

        let address = to_host_port_str(&network_identity.blob_address).map_err(|e| {
            ShardError::NetworkConfig(format!("Cannot convert address to host:port: {e:?}"))
        })?;
        let address = format!("http://{address}/{}", path.path());

        let url = Url::from_str(&address).map_err(|e| ShardError::UrlParseError(e.to_string()))?;
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

#[derive(Clone)]
struct BlobHttpServiceProxy<S: BlobNetworkService + Clone> {
    service: Arc<S>,
}

impl IntoResponse for GetObjectResponse {
    fn into_response(self) -> Response<Body> {
        match self {
            GetObjectResponse::Direct(bytes) => Response::builder()
                .status(StatusCode::OK)
                .body(Body::from(bytes))
                .expect("Failed to build direct response"),
            GetObjectResponse::Redirect(url) => Response::builder()
                .status(StatusCode::PERMANENT_REDIRECT)
                .header("Location", url)
                .body(Body::empty())
                .expect("Failed to build redirect response"),
        }
    }
}

impl<S: BlobNetworkService + Clone> BlobHttpServiceProxy<S> {
    const fn new(service: Arc<S>) -> Self {
        Self { service }
    }

    fn router(self) -> Router {
        Router::new()
            .route("/:path", get(Self::get_object))
            .with_state(self)
    }

    pub async fn get_object(
        Path(path): Path<String>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let peer = NetworkingIndex::default();
        let path = BlobPath::new(path).map_err(|_| StatusCode::BAD_REQUEST)?;
        service
            .handle_get_object(peer, &path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)
    }
}

pub struct BlobHttpManager {
    context: Arc<EncoderContext>,
    client: Arc<BlobHttpClient>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl BlobHttpManager {
    pub fn new(context: Arc<EncoderContext>) -> ShardResult<Self> {
        Ok(Self {
            context: context.clone(),
            client: Arc::new(BlobHttpClient::new(context)?),
            shutdown_tx: None,
        })
    }
}

impl<S: BlobNetworkService + Clone> BlobNetworkManager<S> for BlobHttpManager {
    type Client = BlobHttpClient;

    fn new(context: Arc<EncoderContext>) -> Self {
        Self::new(context).unwrap()
    }

    fn client(&self) -> Arc<Self::Client> {
        self.client.clone()
    }

    async fn start(&mut self, service: Arc<S>) {
        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);

        let network_identity = self
            .context
            .network_committee
            .identity(self.context.own_network_index);
        let own_address = if network_identity.blob_address.is_localhost_ip() {
            network_identity.blob_address.clone()
        } else {
            network_identity.blob_address.with_zero_ip()
        };

        // TODO: fix the identity address to support different ports for different services?
        let own_address = to_socket_addr(&own_address).unwrap();
        let listener = tokio::net::TcpListener::bind(own_address).await.unwrap();

        axum::serve(listener, BlobHttpServiceProxy::new(service).router())
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
