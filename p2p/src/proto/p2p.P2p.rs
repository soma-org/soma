/// Generated client implementations.
pub mod p2p_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    #[derive(Debug, Clone)]
    pub struct P2pClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl P2pClient<tonic::transport::Channel> {
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
    impl<T> P2pClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
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
        ) -> P2pClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
                Response = http::Response<
                    <T as tonic::client::GrpcService<tonic::body::BoxBody>>::ResponseBody,
                >,
            >,
            <T as tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
            >>::Error: Into<StdError> + std::marker::Send + std::marker::Sync,
        {
            P2pClient::new(InterceptedService::new(inner, interceptor))
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
        pub async fn get_known_peers(
            &mut self,
            request: impl tonic::IntoRequest<types::discovery::GetKnownPeersRequest>,
        ) -> std::result::Result<
            tonic::Response<types::discovery::GetKnownPeersResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static("/p2p.P2p/GetKnownPeers");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new("p2p.P2p", "GetKnownPeers"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn get_commit_availability(
            &mut self,
            request: impl tonic::IntoRequest<
                types::state_sync::GetCommitAvailabilityRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<types::state_sync::GetCommitAvailabilityResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/p2p.P2p/GetCommitAvailability",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("p2p.P2p", "GetCommitAvailability"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn push_commit(
            &mut self,
            request: impl tonic::IntoRequest<types::state_sync::PushCommitRequest>,
        ) -> std::result::Result<
            tonic::Response<types::state_sync::PushCommitResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static("/p2p.P2p/PushCommit");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new("p2p.P2p", "PushCommit"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn get_commit_info(
            &mut self,
            request: impl tonic::IntoRequest<types::state_sync::GetCommitInfoRequest>,
        ) -> std::result::Result<
            tonic::Response<types::state_sync::GetCommitInfoResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static("/p2p.P2p/GetCommitInfo");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new("p2p.P2p", "GetCommitInfo"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn fetch_blocks(
            &mut self,
            request: impl tonic::IntoRequest<types::state_sync::FetchBlocksRequest>,
        ) -> std::result::Result<
            tonic::Response<
                tonic::codec::Streaming<types::state_sync::FetchBlocksResponse>,
            >,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static("/p2p.P2p/FetchBlocks");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new("p2p.P2p", "FetchBlocks"));
            self.inner.server_streaming(req, path, codec).await
        }
        pub async fn fetch_commits(
            &mut self,
            request: impl tonic::IntoRequest<types::state_sync::FetchCommitsRequest>,
        ) -> std::result::Result<
            tonic::Response<types::state_sync::FetchCommitsResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = utils::codec::BcsCodec::default();
            let path = http::uri::PathAndQuery::from_static("/p2p.P2p/FetchCommits");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new("p2p.P2p", "FetchCommits"));
            self.inner.unary(req, path, codec).await
        }
    }
}
/// Generated server implementations.
pub mod p2p_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with P2pServer.
    #[async_trait]
    pub trait P2p: std::marker::Send + std::marker::Sync + 'static {
        async fn get_known_peers(
            &self,
            request: tonic::Request<types::discovery::GetKnownPeersRequest>,
        ) -> std::result::Result<
            tonic::Response<types::discovery::GetKnownPeersResponse>,
            tonic::Status,
        >;
        async fn get_commit_availability(
            &self,
            request: tonic::Request<types::state_sync::GetCommitAvailabilityRequest>,
        ) -> std::result::Result<
            tonic::Response<types::state_sync::GetCommitAvailabilityResponse>,
            tonic::Status,
        >;
        async fn push_commit(
            &self,
            request: tonic::Request<types::state_sync::PushCommitRequest>,
        ) -> std::result::Result<
            tonic::Response<types::state_sync::PushCommitResponse>,
            tonic::Status,
        >;
        async fn get_commit_info(
            &self,
            request: tonic::Request<types::state_sync::GetCommitInfoRequest>,
        ) -> std::result::Result<
            tonic::Response<types::state_sync::GetCommitInfoResponse>,
            tonic::Status,
        >;
        /// Server streaming response type for the FetchBlocks method.
        type FetchBlocksStream: tonic::codegen::tokio_stream::Stream<
                Item = std::result::Result<
                    types::state_sync::FetchBlocksResponse,
                    tonic::Status,
                >,
            >
            + std::marker::Send
            + 'static;
        async fn fetch_blocks(
            &self,
            request: tonic::Request<types::state_sync::FetchBlocksRequest>,
        ) -> std::result::Result<
            tonic::Response<Self::FetchBlocksStream>,
            tonic::Status,
        >;
        async fn fetch_commits(
            &self,
            request: tonic::Request<types::state_sync::FetchCommitsRequest>,
        ) -> std::result::Result<
            tonic::Response<types::state_sync::FetchCommitsResponse>,
            tonic::Status,
        >;
    }
    #[derive(Debug)]
    pub struct P2pServer<T> {
        inner: Arc<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
        max_decoding_message_size: Option<usize>,
        max_encoding_message_size: Option<usize>,
    }
    impl<T> P2pServer<T> {
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
    impl<T, B> tonic::codegen::Service<http::Request<B>> for P2pServer<T>
    where
        T: P2p,
        B: Body + std::marker::Send + 'static,
        B::Error: Into<StdError> + std::marker::Send + 'static,
    {
        type Response = http::Response<tonic::body::BoxBody>;
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
                "/p2p.P2p/GetKnownPeers" => {
                    #[allow(non_camel_case_types)]
                    struct GetKnownPeersSvc<T: P2p>(pub Arc<T>);
                    impl<
                        T: P2p,
                    > tonic::server::UnaryService<types::discovery::GetKnownPeersRequest>
                    for GetKnownPeersSvc<T> {
                        type Response = types::discovery::GetKnownPeersResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::discovery::GetKnownPeersRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as P2p>::get_known_peers(&inner, request).await
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
                        let method = GetKnownPeersSvc(inner);
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
                "/p2p.P2p/GetCommitAvailability" => {
                    #[allow(non_camel_case_types)]
                    struct GetCommitAvailabilitySvc<T: P2p>(pub Arc<T>);
                    impl<
                        T: P2p,
                    > tonic::server::UnaryService<
                        types::state_sync::GetCommitAvailabilityRequest,
                    > for GetCommitAvailabilitySvc<T> {
                        type Response = types::state_sync::GetCommitAvailabilityResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::state_sync::GetCommitAvailabilityRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as P2p>::get_commit_availability(&inner, request).await
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
                        let method = GetCommitAvailabilitySvc(inner);
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
                "/p2p.P2p/PushCommit" => {
                    #[allow(non_camel_case_types)]
                    struct PushCommitSvc<T: P2p>(pub Arc<T>);
                    impl<
                        T: P2p,
                    > tonic::server::UnaryService<types::state_sync::PushCommitRequest>
                    for PushCommitSvc<T> {
                        type Response = types::state_sync::PushCommitResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<types::state_sync::PushCommitRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as P2p>::push_commit(&inner, request).await
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
                        let method = PushCommitSvc(inner);
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
                "/p2p.P2p/GetCommitInfo" => {
                    #[allow(non_camel_case_types)]
                    struct GetCommitInfoSvc<T: P2p>(pub Arc<T>);
                    impl<
                        T: P2p,
                    > tonic::server::UnaryService<
                        types::state_sync::GetCommitInfoRequest,
                    > for GetCommitInfoSvc<T> {
                        type Response = types::state_sync::GetCommitInfoResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::state_sync::GetCommitInfoRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as P2p>::get_commit_info(&inner, request).await
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
                        let method = GetCommitInfoSvc(inner);
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
                "/p2p.P2p/FetchBlocks" => {
                    #[allow(non_camel_case_types)]
                    struct FetchBlocksSvc<T: P2p>(pub Arc<T>);
                    impl<
                        T: P2p,
                    > tonic::server::ServerStreamingService<
                        types::state_sync::FetchBlocksRequest,
                    > for FetchBlocksSvc<T> {
                        type Response = types::state_sync::FetchBlocksResponse;
                        type ResponseStream = T::FetchBlocksStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::state_sync::FetchBlocksRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as P2p>::fetch_blocks(&inner, request).await
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
                        let method = FetchBlocksSvc(inner);
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
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/p2p.P2p/FetchCommits" => {
                    #[allow(non_camel_case_types)]
                    struct FetchCommitsSvc<T: P2p>(pub Arc<T>);
                    impl<
                        T: P2p,
                    > tonic::server::UnaryService<types::state_sync::FetchCommitsRequest>
                    for FetchCommitsSvc<T> {
                        type Response = types::state_sync::FetchCommitsResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                types::state_sync::FetchCommitsRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as P2p>::fetch_commits(&inner, request).await
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
                        let method = FetchCommitsSvc(inner);
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
                        Ok(
                            http::Response::builder()
                                .status(200)
                                .header("grpc-status", tonic::Code::Unimplemented as i32)
                                .header(
                                    http::header::CONTENT_TYPE,
                                    tonic::metadata::GRPC_CONTENT_TYPE,
                                )
                                .body(empty_body())
                                .unwrap(),
                        )
                    })
                }
            }
        }
    }
    impl<T> Clone for P2pServer<T> {
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
    pub const SERVICE_NAME: &str = "p2p.P2p";
    impl<T> tonic::server::NamedService for P2pServer<T> {
        const NAME: &'static str = SERVICE_NAME;
    }
}
