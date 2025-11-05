use std::convert::Infallible;
use tonic::server::NamedService;
use tower::Service;

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

mod ledger_service;
mod live_data_service;
mod subscription_service;
mod transaction_execution_service;

#[derive(Default)]
pub struct Services {
    router: axum::Router,
}

impl Services {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new service.
    pub fn add_service<S>(mut self, svc: S) -> Self
    where
        S: Service<
                axum::extract::Request,
                Response: axum::response::IntoResponse,
                Error = Infallible,
            > + NamedService
            + Clone
            + Send
            + Sync
            + 'static,
        S::Future: Send + 'static,
        S::Error: Into<BoxError> + Send,
    {
        self.router = self
            .router
            .route_service(&format!("/{}/{{*rest}}", S::NAME), svc);
        self
    }

    pub fn into_router(self) -> axum::Router {
        self.router.layer(tonic_web::GrpcWebLayer::new())
    }
}
