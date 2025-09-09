use std::sync::Arc;

use types::transaction_executor::TransactionExecutor;
pub mod client;
mod error;
mod grpc;
mod response;

#[derive(Clone)]
pub struct RpcService {
    // reader: StateReader,
    // chain_id: types::digests::ChainIdentifier,
    // config: Config,
    executor: Option<Arc<dyn TransactionExecutor>>,
}

impl RpcService {
    pub fn new() -> Self {
        // let chain_id = reader.get_chain_identifier().unwrap();
        Self {
            // reader: StateReader::new(reader),
            executor: None,
            // chain_id,
            // config: Config::default(),
        }
    }

    pub async fn into_router(self) -> axum::Router {
        let router = {
            // let ledger_service2 =
            //     sui_rpc::proto::sui::rpc::v2beta2::ledger_service_server::LedgerServiceServer::new(
            //         self.clone(),
            //     );
            let transaction_execution_service = crate::proto::soma::transaction_execution_service_server::TransactionExecutionServiceServer::new(self.clone());
            // let live_data_service2 =
            //     sui_rpc::proto::sui::rpc::v2beta2::live_data_service_server::LiveDataServiceServer::new(
            //         self.clone(),
            //     ).send_compressed(tonic::codec::CompressionEncoding::Zstd);

            // let reflection_v1alpha = tonic_reflection::server::Builder::configure()
            //     .register_encoded_file_descriptor_set(
            //         crate::proto::google::protobuf::FILE_DESCRIPTOR_SET,
            //     )
            //     .register_encoded_file_descriptor_set(
            //         crate::proto::google::rpc::FILE_DESCRIPTOR_SET,
            //     )
            //     .register_encoded_file_descriptor_set(
            //         sui_rpc::proto::sui::rpc::v2beta2::FILE_DESCRIPTOR_SET,
            //     )
            //     .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
            //     .build_v1alpha()
            //     .unwrap();

            fn service_name<S: tonic::server::NamedService>(_service: &S) -> &'static str {
                S::NAME
            }

            grpc::Services::new()
                // .add_service(ledger_service2)
                .add_service(transaction_execution_service)
                // .add_service(live_data_service2)
                // .add_service(reflection_v1alpha)
                .into_router()
        };

        router.layer(axum::middleware::map_response_with_state(
            self,
            response::append_info_headers,
        ))
    }

    pub async fn start_service(self, socket_address: std::net::SocketAddr) {
        let listener = tokio::net::TcpListener::bind(socket_address).await.unwrap();
        axum::serve(listener, self.into_router().await)
            .await
            .unwrap();
    }
}
