use std::{marker::PhantomData, str::FromStr, sync::Arc, time::Duration};

use super::{
    EncoderIndex, GetObjectResponse, ObjectNetworkClient, ObjectNetworkManager,
    ObjectNetworkService,
};
use crate::{
    error::{ShardError, ShardResult},
    networking::messaging::{to_host_port_str, to_socket_addr},
    storage::object::ObjectPath,
    types::encoder_context::EncoderContext,
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

pub(crate) struct ObjectHttpClient {
    client: Client,
    context: Arc<EncoderContext>,
}

impl ObjectHttpClient {
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
impl ObjectNetworkClient for ObjectHttpClient {
    async fn get_object(
        &self,
        peer: EncoderIndex,
        path: &ObjectPath,
        timeout: Duration,
    ) -> ShardResult<Bytes> {
        let encoder = self.context.encoder_committee.encoder(peer);

        let address = to_host_port_str(&encoder.address).map_err(|e| {
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
struct ObjectHttpServiceProxy<S: ObjectNetworkService + Clone> {
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

impl<S: ObjectNetworkService + Clone> ObjectHttpServiceProxy<S> {
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
        let peer = EncoderIndex::default();
        let path = ObjectPath::new(path).map_err(|_| StatusCode::BAD_REQUEST)?;
        service
            .handle_get_object(peer, &path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)
    }
}

pub struct ObjectHttpManager<S: ObjectNetworkService + Clone> {
    context: Arc<EncoderContext>,
    client: Arc<ObjectHttpClient>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    marker: PhantomData<S>,
}

impl<S: ObjectNetworkService + Clone> ObjectNetworkManager<S> for ObjectHttpManager<S> {
    type Client = ObjectHttpClient;

    fn new(context: Arc<EncoderContext>) -> ShardResult<Self> {
        Ok(Self {
            context: context.clone(),
            client: Arc::new(ObjectHttpClient::new(context)?),
            shutdown_tx: None,
            marker: PhantomData,
        })
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
