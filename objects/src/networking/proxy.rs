use crate::{
    networking::ObjectNetworkService,
    storage::{ObjectPath, ObjectStorage},
};
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, Response, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};
use soma_http::ServerHandle;
use std::time::{Duration, Instant};
use tokio_util::io::ReaderStream;
use tracing::{info, warn};
use types::{multiaddr::Multiaddr, p2p::to_socket_addr};

pub struct LocalObjectServerManager {
    server: Option<ServerHandle>,
}

impl LocalObjectServerManager {
    pub fn new() -> Self {
        Self { server: None }
    }

    pub async fn start<S: ObjectStorage + Clone>(
        &mut self,
        address: &Multiaddr,
        service: ObjectNetworkService<S>,
    ) {
        let own_address = if address.is_localhost_ip() {
            address.clone()
        } else {
            address.with_zero_ip()
        };

        let own_address = to_socket_addr(&own_address).unwrap();

        let deadline = Instant::now() + Duration::from_secs(20);
        let server = loop {
            match soma_http::Builder::new().serve(
                own_address,
                LocalObjectServer::new(service.clone()).router(),
            ) {
                Ok(server) => break server,
                Err(err) => {
                    warn!("Error starting internal encoder server: {err:?}");
                    if Instant::now() > deadline {
                        panic!("Failed to start internal encoder server within required deadline");
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        };

        info!("Object server started at: {own_address}");
        self.server = Some(server);
    }

    async fn stop(&mut self) {
        if let Some(server) = self.server.take() {
            server.shutdown().await;
        }
    }
}

impl Drop for LocalObjectServerManager {
    fn drop(&mut self) {
        if let Some(server) = self.server.as_ref() {
            server.trigger_shutdown();
        }
    }
}

#[derive(Clone)]
struct LocalObjectServer<S: ObjectStorage> {
    service: ObjectNetworkService<S>,
}

impl<S: ObjectStorage + Clone> LocalObjectServer<S> {
    const fn new(service: ObjectNetworkService<S>) -> Self {
        Self { service }
    }
    fn router(self) -> Router {
        Router::new()
            .route("/{path}", get(Self::download_object))
            .with_state(self)
    }

    pub async fn download_object(
        Path(path): Path<String>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::new(path).map_err(|_| StatusCode::BAD_REQUEST)?;
        let reader = service
            .handle_download_object(&path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)?;
        // Convert BufReader<fs::File> to a Stream of Bytes
        let stream = ReaderStream::new(reader);

        // Wrap the stream in StreamBody
        let body = Body::from_stream(stream);

        // Build the response
        let response = Response::builder()
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(body)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        Ok(response)
    }
}
