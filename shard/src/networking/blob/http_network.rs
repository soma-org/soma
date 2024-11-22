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
    extract::{Path, State},
    http::Response,
    routing::get,
    Router,
};
use bytes::Bytes;
use reqwest::{Client, StatusCode};
use tokio::sync::oneshot;
use url::Url;

use super::{BlobNetworkClient, BlobNetworkManager, BlobNetworkService, BlobStorageNetworkService};

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

        let address = to_host_port_str(&network_identity.address).map_err(|e| {
            ShardError::NetworkConfig(format!("Cannot convert address to host:port: {e:?}"))
        })?;
        let address = format!("http://{address}/{}", path.path());
        // TODO: configure the port correctly to match with the changed identity

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

#[derive(Clone)]
struct BlobHttpServiceProxy<S: BlobNetworkService + Clone> {
    service: Arc<S>,
}

impl<S: BlobNetworkService + Clone> BlobHttpServiceProxy<S> {
    /// Creates the tonic service proxy using pre-established context and service
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
        State(Self { service, .. }): State<Self>,
    ) -> Result<Bytes, StatusCode> {
        let peer = NetworkingIndex::default();
        // TODO: get peer correctly
        // TODO: change this to handle redirection?
        let path = BlobPath::new(path).map_err(|_| StatusCode::BAD_REQUEST)?;
        Ok(service
            .handle_get_object(peer, &path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)?)
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
        let own_address = if network_identity.address.is_localhost_ip() {
            network_identity.address.clone()
        } else {
            network_identity.address.with_zero_ip()
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
