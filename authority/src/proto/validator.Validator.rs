/// Generated client implementations.
pub mod validator_client {
    #![allow(
        unused_variables,
        dead_code,
        missing_docs,
        clippy::wildcard_imports,
        clippy::let_unit_value,
    )]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    /// The Validator interface
    #[derive(Debug, Clone)]
    pub struct ValidatorClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl ValidatorClient<tonic::transport::Channel> {
        /// Attempt to create a new client by connecting to a given endpoint.
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> ValidatorClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::Body>,
        T::Error: Into<StdError>,
        T::ResponseBody: Body<Data = Bytes> + std::marker::Send + 'static,
        <T::ResponseBody as Body>::Error: Into<StdError> + std::marker::Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_origin(inner: T, origin: Uri) -> Self {
            let inner = tonic::client::Grpc::with_origin(inner, origin);
            Self { inner }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> ValidatorClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                http::Request<tonic::body::Body>,
                Response = http::Response<
                    <T as tonic::client::GrpcService<tonic::body::Body>>::ResponseBody,
                >,
            >,
            <T as tonic::codegen::Service<
                http::Request<tonic::body::Body>,
            >>::Error: Into<StdError> + std::marker::Send + std::marker::Sync,
        {
            ValidatorClient::new(InterceptedService::new(inner, interceptor))
        }
        /// Compress requests with the given encoding.
        ///
        /// This requires the server to support it otherwise it might respond with an
        /// error.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.send_compressed(encoding);
            self
        }
        /// Enable decompressing responses.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.accept_compressed(encoding);
            self
        }
        /// Limits the maximum size of a decoded message.
        ///
        /// Default: `4MB`
        #[must_use]
        pub fn max_decoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_decoding_message_size(limit);
            self
        }
        /// Limits the maximum size of an encoded message.
        ///
        /// Default: `usize::MAX`
        #[must_use]
        pub fn max_encoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_encoding_message_size(limit);
            self
        }
        pub async fn submit_transaction(
            &mut self,
            request: impl tonic::IntoRequest<types::messages_grpc::RawSubmitTxRequest>,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::RawSubmitTxResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::unknown(
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/validator.Validator/SubmitTransaction",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("validator.Validator", "SubmitTransaction"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn wait_for_effects(
            &mut self,
            request: impl tonic::IntoRequest<
                types::messages_grpc::RawWaitForEffectsRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::RawWaitForEffectsResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::unknown(
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/validator.Validator/WaitForEffects",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("validator.Validator", "WaitForEffects"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn transaction(
            &mut self,
            request: impl tonic::IntoRequest<types::transaction::Transaction>,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::HandleTransactionResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::unknown(
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/validator.Validator/Transaction",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("validator.Validator", "Transaction"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn handle_certificate(
            &mut self,
            request: impl tonic::IntoRequest<
                types::messages_grpc::HandleCertificateRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::HandleCertificateResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::unknown(
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/validator.Validator/CertifiedTransaction",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("validator.Validator", "CertifiedTransaction"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn object_info(
            &mut self,
            request: impl tonic::IntoRequest<types::messages_grpc::ObjectInfoRequest>,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::ObjectInfoResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::unknown(
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/validator.Validator/ObjectInfo",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("validator.Validator", "ObjectInfo"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn transaction_info(
            &mut self,
            request: impl tonic::IntoRequest<
                types::messages_grpc::TransactionInfoRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::TransactionInfoResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::unknown(
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/validator.Validator/TransactionInfo",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("validator.Validator", "TransactionInfo"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn checkpoint(
            &mut self,
            request: impl tonic::IntoRequest<types::checkpoints::CheckpointRequest>,
        ) -> std::result::Result<
            tonic::Response<types::checkpoints::CheckpointResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::unknown(
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/validator.Validator/Checkpoint",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("validator.Validator", "Checkpoint"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn get_system_state_object(
            &mut self,
            request: impl tonic::IntoRequest<types::messages_grpc::SystemStateRequest>,
        ) -> std::result::Result<
            tonic::Response<types::system_state::SystemState>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::unknown(
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/validator.Validator/GetSystemStateObject",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("validator.Validator", "GetSystemStateObject"));
            self.inner.unary(req, path, codec).await
        }
    }
}
/// Generated server implementations.
pub mod validator_server {
    #![allow(
        unused_variables,
        dead_code,
        missing_docs,
        clippy::wildcard_imports,
        clippy::let_unit_value,
    )]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with ValidatorServer.
    #[async_trait]
    pub trait Validator: std::marker::Send + std::marker::Sync + 'static {
        async fn submit_transaction(
            &self,
            request: tonic::Request<types::messages_grpc::RawSubmitTxRequest>,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::RawSubmitTxResponse>,
            tonic::Status,
        >;
        async fn wait_for_effects(
            &self,
            request: tonic::Request<types::messages_grpc::RawWaitForEffectsRequest>,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::RawWaitForEffectsResponse>,
            tonic::Status,
        >;
        async fn transaction(
            &self,
            request: tonic::Request<types::transaction::Transaction>,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::HandleTransactionResponse>,
            tonic::Status,
        >;
        async fn handle_certificate(
            &self,
            request: tonic::Request<types::messages_grpc::HandleCertificateRequest>,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::HandleCertificateResponse>,
            tonic::Status,
        >;
        async fn object_info(
            &self,
            request: tonic::Request<types::messages_grpc::ObjectInfoRequest>,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::ObjectInfoResponse>,
            tonic::Status,
        >;
        async fn transaction_info(
            &self,
            request: tonic::Request<types::messages_grpc::TransactionInfoRequest>,
        ) -> std::result::Result<
            tonic::Response<types::messages_grpc::TransactionInfoResponse>,
            tonic::Status,
        >;
        async fn checkpoint(
            &self,
            request: tonic::Request<types::checkpoints::CheckpointRequest>,
        ) -> std::result::Result<
            tonic::Response<types::checkpoints::CheckpointResponse>,
            tonic::Status,
        >;
        async fn get_system_state_object(
            &self,
            request: tonic::Request<types::messages_grpc::SystemStateRequest>,
        ) -> std::result::Result<
            tonic::Response<types::system_state::SystemState>,
            tonic::Status,
        >;
    }
    /// The Validator interface
    #[derive(Debug)]
    pub struct ValidatorServer<T> {
        inner: Arc<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
        max_decoding_message_size: Option<usize>,
        max_encoding_message_size: Option<usize>,
    }
    impl<T> ValidatorServer<T> {
        pub fn new(inner: T) -> Self {
            Self::from_arc(Arc::new(inner))
        }
        pub fn from_arc(inner: Arc<T>) -> Self {
            Self {
                inner,
                accept_compression_encodings: Default::default(),
                send_compression_encodings: Default::default(),
                max_decoding_message_size: None,
                max_encoding_message_size: None,
            }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> InterceptedService<Self, F>
        where
            F: tonic::service::Interceptor,
        {
            InterceptedService::new(Self::new(inner), interceptor)
        }
        /// Enable decompressing requests with the given encoding.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.accept_compression_encodings.enable(encoding);
            self
        }
        /// Compress responses with the given encoding, if the client supports it.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.send_compression_encodings.enable(encoding);
            self
        }
        /// Limits the maximum size of a decoded message.
        ///
        /// Default: `4MB`
        #[must_use]
        pub fn max_decoding_message_size(mut self, limit: usize) -> Self {
            self.max_decoding_message_size = Some(limit);
            self
        }
        /// Limits the maximum size of an encoded message.
        ///
        /// Default: `usize::MAX`
        #[must_use]
        pub fn max_encoding_message_size(mut self, limit: usize) -> Self {
            self.max_encoding_message_size = Some(limit);
            self
        }
    }
    impl<T, B> tonic::codegen::Service<http::Request<B>> for ValidatorServer<T>
    where
        T: Validator,
        B: Body + std::marker::Send + 'static,
        B::Error: Into<StdError> + std::marker::Send + 'static,
    {
        type Response = http::Response<tonic::body::Body>;
        type Error = std::convert::Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(
            &mut self,
            _cx: &mut Context<'_>,
        ) -> Poll<std::result::Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, req: http::Request<B>) -> Self::Future {
            match req.uri().path() {
                "/validator.Validator/SubmitTransaction" => {
                    #[allow(non_camel_case_types)]
                    struct SubmitTransactionSvc<T: Validator>(pub Arc<T>);
                    impl<
                        T: Validator,
                    > tonic::server::UnaryService<
                        types::messages_grpc::RawSubmitTxRequest,
                    > for SubmitTransactionSvc<T> {
                        type Response = types::messages_grpc::RawSubmitTxResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::messages_grpc::RawSubmitTxRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as Validator>::submit_transaction(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = SubmitTransactionSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/validator.Validator/WaitForEffects" => {
                    #[allow(non_camel_case_types)]
                    struct WaitForEffectsSvc<T: Validator>(pub Arc<T>);
                    impl<
                        T: Validator,
                    > tonic::server::UnaryService<
                        types::messages_grpc::RawWaitForEffectsRequest,
                    > for WaitForEffectsSvc<T> {
                        type Response = types::messages_grpc::RawWaitForEffectsResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::messages_grpc::RawWaitForEffectsRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as Validator>::wait_for_effects(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = WaitForEffectsSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/validator.Validator/Transaction" => {
                    #[allow(non_camel_case_types)]
                    struct TransactionSvc<T: Validator>(pub Arc<T>);
                    impl<
                        T: Validator,
                    > tonic::server::UnaryService<types::transaction::Transaction>
                    for TransactionSvc<T> {
                        type Response = types::messages_grpc::HandleTransactionResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<types::transaction::Transaction>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as Validator>::transaction(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = TransactionSvc(inner);
                        let codec = utils::codec::BcsCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/validator.Validator/CertifiedTransaction" => {
                    #[allow(non_camel_case_types)]
                    struct CertifiedTransactionSvc<T: Validator>(pub Arc<T>);
                    impl<
                        T: Validator,
                    > tonic::server::UnaryService<
                        types::messages_grpc::HandleCertificateRequest,
                    > for CertifiedTransactionSvc<T> {
                        type Response = types::messages_grpc::HandleCertificateResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::messages_grpc::HandleCertificateRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as Validator>::handle_certificate(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = CertifiedTransactionSvc(inner);
                        let codec = utils::codec::BcsCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/validator.Validator/ObjectInfo" => {
                    #[allow(non_camel_case_types)]
                    struct ObjectInfoSvc<T: Validator>(pub Arc<T>);
                    impl<
                        T: Validator,
                    > tonic::server::UnaryService<
                        types::messages_grpc::ObjectInfoRequest,
                    > for ObjectInfoSvc<T> {
                        type Response = types::messages_grpc::ObjectInfoResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::messages_grpc::ObjectInfoRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as Validator>::object_info(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = ObjectInfoSvc(inner);
                        let codec = utils::codec::BcsCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/validator.Validator/TransactionInfo" => {
                    #[allow(non_camel_case_types)]
                    struct TransactionInfoSvc<T: Validator>(pub Arc<T>);
                    impl<
                        T: Validator,
                    > tonic::server::UnaryService<
                        types::messages_grpc::TransactionInfoRequest,
                    > for TransactionInfoSvc<T> {
                        type Response = types::messages_grpc::TransactionInfoResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::messages_grpc::TransactionInfoRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as Validator>::transaction_info(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = TransactionInfoSvc(inner);
                        let codec = utils::codec::BcsCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/validator.Validator/Checkpoint" => {
                    #[allow(non_camel_case_types)]
                    struct CheckpointSvc<T: Validator>(pub Arc<T>);
                    impl<
                        T: Validator,
                    > tonic::server::UnaryService<types::checkpoints::CheckpointRequest>
                    for CheckpointSvc<T> {
                        type Response = types::checkpoints::CheckpointResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::checkpoints::CheckpointRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as Validator>::checkpoint(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = CheckpointSvc(inner);
                        let codec = utils::codec::BcsCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/validator.Validator/GetSystemStateObject" => {
                    #[allow(non_camel_case_types)]
                    struct GetSystemStateObjectSvc<T: Validator>(pub Arc<T>);
                    impl<
                        T: Validator,
                    > tonic::server::UnaryService<
                        types::messages_grpc::SystemStateRequest,
                    > for GetSystemStateObjectSvc<T> {
                        type Response = types::system_state::SystemState;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::messages_grpc::SystemStateRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as Validator>::get_system_state_object(&inner, request)
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = GetSystemStateObjectSvc(inner);
                        let codec = utils::codec::BcsCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                _ => {
                    Box::pin(async move {
                        let mut response = http::Response::new(
                            tonic::body::Body::default(),
                        );
                        let headers = response.headers_mut();
                        headers
                            .insert(
                                tonic::Status::GRPC_STATUS,
                                (tonic::Code::Unimplemented as i32).into(),
                            );
                        headers
                            .insert(
                                http::header::CONTENT_TYPE,
                                tonic::metadata::GRPC_CONTENT_TYPE,
                            );
                        Ok(response)
                    })
                }
            }
        }
    }
    impl<T> Clone for ValidatorServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
                max_decoding_message_size: self.max_decoding_message_size,
                max_encoding_message_size: self.max_encoding_message_size,
            }
        }
    }
    /// Generated gRPC service name
    pub const SERVICE_NAME: &str = "validator.Validator";
    impl<T> tonic::server::NamedService for ValidatorServer<T> {
        const NAME: &'static str = SERVICE_NAME;
    }
}
