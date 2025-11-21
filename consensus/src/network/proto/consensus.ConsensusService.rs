/// Generated client implementations.
pub mod consensus_service_client {
    #![allow(
        unused_variables,
        dead_code,
        missing_docs,
        clippy::wildcard_imports,
        clippy::let_unit_value,
    )]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    /// Consensus authority interface
    #[derive(Debug, Clone)]
    pub struct ConsensusServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl ConsensusServiceClient<tonic::transport::Channel> {
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
    impl<T> ConsensusServiceClient<T>
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
        ) -> ConsensusServiceClient<InterceptedService<T, F>>
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
            ConsensusServiceClient::new(InterceptedService::new(inner, interceptor))
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
        pub async fn send_block(
            &mut self,
            request: impl tonic::IntoRequest<
                crate::network::tonic_network::SendBlockRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<crate::network::tonic_network::SendBlockResponse>,
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
                "/consensus.ConsensusService/SendBlock",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("consensus.ConsensusService", "SendBlock"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn subscribe_blocks(
            &mut self,
            request: impl tonic::IntoStreamingRequest<
                Message = crate::network::tonic_network::SubscribeBlocksRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<
                tonic::codec::Streaming<
                    crate::network::tonic_network::SubscribeBlocksResponse,
                >,
            >,
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
                "/consensus.ConsensusService/SubscribeBlocks",
            );
            let mut req = request.into_streaming_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("consensus.ConsensusService", "SubscribeBlocks"),
                );
            self.inner.streaming(req, path, codec).await
        }
        pub async fn fetch_blocks(
            &mut self,
            request: impl tonic::IntoRequest<
                crate::network::tonic_network::FetchBlocksRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<
                tonic::codec::Streaming<
                    crate::network::tonic_network::FetchBlocksResponse,
                >,
            >,
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
                "/consensus.ConsensusService/FetchBlocks",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("consensus.ConsensusService", "FetchBlocks"));
            self.inner.server_streaming(req, path, codec).await
        }
        pub async fn fetch_commits(
            &mut self,
            request: impl tonic::IntoRequest<
                crate::network::tonic_network::FetchCommitsRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<crate::network::tonic_network::FetchCommitsResponse>,
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
                "/consensus.ConsensusService/FetchCommits",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("consensus.ConsensusService", "FetchCommits"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn fetch_latest_blocks(
            &mut self,
            request: impl tonic::IntoRequest<
                crate::network::tonic_network::FetchLatestBlocksRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<
                tonic::codec::Streaming<
                    crate::network::tonic_network::FetchLatestBlocksResponse,
                >,
            >,
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
                "/consensus.ConsensusService/FetchLatestBlocks",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("consensus.ConsensusService", "FetchLatestBlocks"),
                );
            self.inner.server_streaming(req, path, codec).await
        }
        pub async fn get_latest_rounds(
            &mut self,
            request: impl tonic::IntoRequest<
                crate::network::tonic_network::GetLatestRoundsRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<crate::network::tonic_network::GetLatestRoundsResponse>,
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
                "/consensus.ConsensusService/GetLatestRounds",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("consensus.ConsensusService", "GetLatestRounds"),
                );
            self.inner.unary(req, path, codec).await
        }
    }
}
/// Generated server implementations.
pub mod consensus_service_server {
    #![allow(
        unused_variables,
        dead_code,
        missing_docs,
        clippy::wildcard_imports,
        clippy::let_unit_value,
    )]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with ConsensusServiceServer.
    #[async_trait]
    pub trait ConsensusService: std::marker::Send + std::marker::Sync + 'static {
        async fn send_block(
            &self,
            request: tonic::Request<crate::network::tonic_network::SendBlockRequest>,
        ) -> std::result::Result<
            tonic::Response<crate::network::tonic_network::SendBlockResponse>,
            tonic::Status,
        >;
        /// Server streaming response type for the SubscribeBlocks method.
        type SubscribeBlocksStream: tonic::codegen::tokio_stream::Stream<
                Item = std::result::Result<
                    crate::network::tonic_network::SubscribeBlocksResponse,
                    tonic::Status,
                >,
            >
            + std::marker::Send
            + 'static;
        async fn subscribe_blocks(
            &self,
            request: tonic::Request<
                tonic::Streaming<crate::network::tonic_network::SubscribeBlocksRequest>,
            >,
        ) -> std::result::Result<
            tonic::Response<Self::SubscribeBlocksStream>,
            tonic::Status,
        >;
        /// Server streaming response type for the FetchBlocks method.
        type FetchBlocksStream: tonic::codegen::tokio_stream::Stream<
                Item = std::result::Result<
                    crate::network::tonic_network::FetchBlocksResponse,
                    tonic::Status,
                >,
            >
            + std::marker::Send
            + 'static;
        async fn fetch_blocks(
            &self,
            request: tonic::Request<crate::network::tonic_network::FetchBlocksRequest>,
        ) -> std::result::Result<
            tonic::Response<Self::FetchBlocksStream>,
            tonic::Status,
        >;
        async fn fetch_commits(
            &self,
            request: tonic::Request<crate::network::tonic_network::FetchCommitsRequest>,
        ) -> std::result::Result<
            tonic::Response<crate::network::tonic_network::FetchCommitsResponse>,
            tonic::Status,
        >;
        /// Server streaming response type for the FetchLatestBlocks method.
        type FetchLatestBlocksStream: tonic::codegen::tokio_stream::Stream<
                Item = std::result::Result<
                    crate::network::tonic_network::FetchLatestBlocksResponse,
                    tonic::Status,
                >,
            >
            + std::marker::Send
            + 'static;
        async fn fetch_latest_blocks(
            &self,
            request: tonic::Request<
                crate::network::tonic_network::FetchLatestBlocksRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<Self::FetchLatestBlocksStream>,
            tonic::Status,
        >;
        async fn get_latest_rounds(
            &self,
            request: tonic::Request<
                crate::network::tonic_network::GetLatestRoundsRequest,
            >,
        ) -> std::result::Result<
            tonic::Response<crate::network::tonic_network::GetLatestRoundsResponse>,
            tonic::Status,
        >;
    }
    /// Consensus authority interface
    #[derive(Debug)]
    pub struct ConsensusServiceServer<T> {
        inner: Arc<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
        max_decoding_message_size: Option<usize>,
        max_encoding_message_size: Option<usize>,
    }
    impl<T> ConsensusServiceServer<T> {
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
    impl<T, B> tonic::codegen::Service<http::Request<B>> for ConsensusServiceServer<T>
    where
        T: ConsensusService,
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
                "/consensus.ConsensusService/SendBlock" => {
                    #[allow(non_camel_case_types)]
                    struct SendBlockSvc<T: ConsensusService>(pub Arc<T>);
                    impl<
                        T: ConsensusService,
                    > tonic::server::UnaryService<
                        crate::network::tonic_network::SendBlockRequest,
                    > for SendBlockSvc<T> {
                        type Response = crate::network::tonic_network::SendBlockResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                crate::network::tonic_network::SendBlockRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as ConsensusService>::send_block(&inner, request).await
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
                        let method = SendBlockSvc(inner);
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
                "/consensus.ConsensusService/SubscribeBlocks" => {
                    #[allow(non_camel_case_types)]
                    struct SubscribeBlocksSvc<T: ConsensusService>(pub Arc<T>);
                    impl<
                        T: ConsensusService,
                    > tonic::server::StreamingService<
                        crate::network::tonic_network::SubscribeBlocksRequest,
                    > for SubscribeBlocksSvc<T> {
                        type Response = crate::network::tonic_network::SubscribeBlocksResponse;
                        type ResponseStream = T::SubscribeBlocksStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                tonic::Streaming<
                                    crate::network::tonic_network::SubscribeBlocksRequest,
                                >,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as ConsensusService>::subscribe_blocks(&inner, request)
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
                        let method = SubscribeBlocksSvc(inner);
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
                        let res = grpc.streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/consensus.ConsensusService/FetchBlocks" => {
                    #[allow(non_camel_case_types)]
                    struct FetchBlocksSvc<T: ConsensusService>(pub Arc<T>);
                    impl<
                        T: ConsensusService,
                    > tonic::server::ServerStreamingService<
                        crate::network::tonic_network::FetchBlocksRequest,
                    > for FetchBlocksSvc<T> {
                        type Response = crate::network::tonic_network::FetchBlocksResponse;
                        type ResponseStream = T::FetchBlocksStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                crate::network::tonic_network::FetchBlocksRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as ConsensusService>::fetch_blocks(&inner, request).await
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
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/consensus.ConsensusService/FetchCommits" => {
                    #[allow(non_camel_case_types)]
                    struct FetchCommitsSvc<T: ConsensusService>(pub Arc<T>);
                    impl<
                        T: ConsensusService,
                    > tonic::server::UnaryService<
                        crate::network::tonic_network::FetchCommitsRequest,
                    > for FetchCommitsSvc<T> {
                        type Response = crate::network::tonic_network::FetchCommitsResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                crate::network::tonic_network::FetchCommitsRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as ConsensusService>::fetch_commits(&inner, request)
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
                        let method = FetchCommitsSvc(inner);
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
                "/consensus.ConsensusService/FetchLatestBlocks" => {
                    #[allow(non_camel_case_types)]
                    struct FetchLatestBlocksSvc<T: ConsensusService>(pub Arc<T>);
                    impl<
                        T: ConsensusService,
                    > tonic::server::ServerStreamingService<
                        crate::network::tonic_network::FetchLatestBlocksRequest,
                    > for FetchLatestBlocksSvc<T> {
                        type Response = crate::network::tonic_network::FetchLatestBlocksResponse;
                        type ResponseStream = T::FetchLatestBlocksStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                crate::network::tonic_network::FetchLatestBlocksRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as ConsensusService>::fetch_latest_blocks(
                                        &inner,
                                        request,
                                    )
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
                        let method = FetchLatestBlocksSvc(inner);
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
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/consensus.ConsensusService/GetLatestRounds" => {
                    #[allow(non_camel_case_types)]
                    struct GetLatestRoundsSvc<T: ConsensusService>(pub Arc<T>);
                    impl<
                        T: ConsensusService,
                    > tonic::server::UnaryService<
                        crate::network::tonic_network::GetLatestRoundsRequest,
                    > for GetLatestRoundsSvc<T> {
                        type Response = crate::network::tonic_network::GetLatestRoundsResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                crate::network::tonic_network::GetLatestRoundsRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as ConsensusService>::get_latest_rounds(&inner, request)
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
                        let method = GetLatestRoundsSvc(inner);
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
    impl<T> Clone for ConsensusServiceServer<T> {
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
    pub const SERVICE_NAME: &str = "consensus.ConsensusService";
    impl<T> tonic::server::NamedService for ConsensusServiceServer<T> {
        const NAME: &'static str = SERVICE_NAME;
    }
}
