use std::{marker::PhantomData, str::FromStr, sync::Arc, time::Duration};

use super::{
    EncoderIndex, Epoch, ObjectNetworkClient, ObjectNetworkManager, ObjectNetworkService,
    ObjectStorage,
};
use crate::{
    error::{ShardError, ShardResult},
    messaging::{to_host_port_str, to_socket_addr},
    storage::object::{ObjectPath, ServedObjectResponse},
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
use shared::{
    crypto::keys::{PeerKeyPair, PeerPublicKey},
    metadata::{Metadata, MetadataAPI},
    multiaddr::Multiaddr,
};
use tokio::sync::oneshot;
use url::Url;

pub(crate) struct ObjectHttpClient {
    client: Client,
    peer_keypair: Arc<PeerKeyPair>,
}

impl ObjectHttpClient {
    pub fn new(peer_keypair: Arc<PeerKeyPair>) -> ShardResult<Self> {
        Ok(Self {
            client: Client::builder()
                .pool_idle_timeout(Duration::from_secs(60 * 5))
                .build()
                .map_err(|_| ShardError::FailedBuildingHttpClient)?,
            peer_keypair,
        })
    }
}

#[async_trait]
impl ObjectNetworkClient for ObjectHttpClient {
    async fn get_object(
        &self,
        peer: &PeerPublicKey,
        address: &Multiaddr,
        metadata: &Metadata,
        timeout: Duration,
    ) -> ShardResult<Bytes> {
        let address = to_host_port_str(&address).map_err(|e| {
            ShardError::NetworkConfig(format!("Cannot convert address to host:port: {e:?}"))
        })?;
        let address = format!("http://{address}/{}", metadata.checksum());

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
struct ObjectHttpServiceProxy<S: ObjectStorage> {
    service: ObjectNetworkService<S>,
}

impl IntoResponse for ServedObjectResponse {
    fn into_response(self) -> Response<Body> {
        match self {
            ServedObjectResponse::Direct(bytes) => Response::builder()
                .status(StatusCode::OK)
                .body(Body::from(bytes))
                .expect("Failed to build direct response"),
            ServedObjectResponse::Redirect(url) => Response::builder()
                .status(StatusCode::PERMANENT_REDIRECT)
                .header("Location", url)
                .body(Body::empty())
                .expect("Failed to build redirect response"),
        }
    }
}

impl<S: ObjectStorage + Clone> ObjectHttpServiceProxy<S> {
    const fn new(service: ObjectNetworkService<S>) -> Self {
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
        let peer: PeerPublicKey = todo!();
        let path = ObjectPath::new(path).map_err(|_| StatusCode::BAD_REQUEST)?;
        service
            .handle_get_object(&peer, &path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)
    }
}

pub struct ObjectHttpManager<S: ObjectStorage> {
    client: Arc<ObjectHttpClient>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    marker: PhantomData<S>,
}

impl<S: ObjectStorage + Clone> ObjectNetworkManager<S> for ObjectHttpManager<S> {
    type Client = ObjectHttpClient;

    fn new(peer_keypair: Arc<PeerKeyPair>) -> ShardResult<Self> {
        Ok(Self {
            client: Arc::new(ObjectHttpClient::new(peer_keypair)?),
            shutdown_tx: None,
            marker: PhantomData,
        })
    }

    fn client(&self) -> Arc<Self::Client> {
        self.client.clone()
    }

    async fn start(&mut self, address: &Multiaddr, service: ObjectNetworkService<S>) {
        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);

        let own_address = if address.is_localhost_ip() {
            address.clone()
        } else {
            address.with_zero_ip()
        };

        let own_address = to_socket_addr(&own_address).unwrap();
        let listener = tokio::net::TcpListener::bind(own_address).await.unwrap();

        axum::serve(listener, ObjectHttpServiceProxy::new(service).router())
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
