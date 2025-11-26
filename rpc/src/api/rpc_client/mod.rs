use std::time::Duration;
use tap::Pipe;
use tonic::codec::CompressionEncoding;
use tonic::transport::channel::ClientTlsConfig;

mod response_ext;
pub use response_ext::ResponseExt;

pub mod headers;
mod interceptors;
pub use interceptors::HeadersInterceptor;

// TODO: mod staking_rewards;
// TODO: mod coin_selection;
mod lists;
// pub use staking_rewards::DelegatedStake;

mod transaction_execution;
pub use transaction_execution::ExecuteAndWaitError;

use crate::proto::soma::ledger_service_client::LedgerServiceClient;
use crate::proto::soma::state_service_client::StateServiceClient;
use crate::proto::soma::subscription_service_client::SubscriptionServiceClient;
use crate::proto::soma::transaction_execution_service_client::TransactionExecutionServiceClient;

type Result<T, E = tonic::Status> = std::result::Result<T, E>;
type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
type Channel<'a> = tonic::service::interceptor::InterceptedService<
    &'a mut tonic::transport::Channel,
    &'a HeadersInterceptor,
>;

#[derive(Clone)]
pub struct Client {
    uri: http::Uri,
    channel: tonic::transport::Channel,
    headers: HeadersInterceptor,
    max_decoding_message_size: Option<usize>,
}

impl Client {
    /// URL for the public-good, Soma Foundation provided fullnodes for mainnet.
    pub const MAINNET_FULLNODE: &str = "https://fullnode.mainnet.soma.io";

    /// URL for the public-good, Soma Foundation provided fullnodes for testnet.
    pub const TESTNET_FULLNODE: &str = "https://fullnode.testnet.soma.io";

    /// URL for the public-good, Soma Foundation provided fullnodes for devnet.
    pub const DEVNET_FULLNODE: &str = "https://fullnode.devnet.soma.io";

    /// URL for the public-good, Soma Foundation provided archive for mainnet.
    pub const MAINNET_ARCHIVE: &str = "https://archive.mainnet.soma.io";

    /// URL for the public-good, Soma Foundation provided archive for testnet.
    pub const TESTNET_ARCHIVE: &str = "https://archive.testnet.soma.io";

    #[allow(clippy::result_large_err)]
    pub fn new<T>(uri: T) -> Result<Self>
    where
        T: TryInto<http::Uri>,
        T::Error: Into<BoxError>,
    {
        let uri = uri
            .try_into()
            .map_err(Into::into)
            .map_err(tonic::Status::from_error)?;
        let mut endpoint = tonic::transport::Endpoint::from(uri.clone());
        if uri.scheme() == Some(&http::uri::Scheme::HTTPS) {
            endpoint = endpoint
                .tls_config(ClientTlsConfig::new().with_enabled_roots())
                .map_err(Into::into)
                .map_err(tonic::Status::from_error)?;
        }
        let channel = endpoint
            .connect_timeout(Duration::from_secs(5))
            .http2_keep_alive_interval(Duration::from_secs(5))
            .connect_lazy();

        Ok(Self {
            uri,
            channel,
            headers: Default::default(),
            max_decoding_message_size: None,
        })
    }

    pub fn with_headers(mut self, headers: HeadersInterceptor) -> Self {
        self.headers = headers;
        self
    }

    pub fn with_max_decoding_message_size(mut self, limit: usize) -> Self {
        self.max_decoding_message_size = Some(limit);
        self
    }

    pub fn uri(&self) -> &http::Uri {
        &self.uri
    }

    pub fn ledger_client(&mut self) -> LedgerServiceClient<Channel<'_>> {
        LedgerServiceClient::with_interceptor(&mut self.channel, &self.headers)
            .accept_compressed(CompressionEncoding::Zstd)
            .pipe(|client| {
                if let Some(limit) = self.max_decoding_message_size {
                    client.max_decoding_message_size(limit)
                } else {
                    client
                }
            })
    }

    pub fn state_client(&mut self) -> StateServiceClient<Channel<'_>> {
        StateServiceClient::with_interceptor(&mut self.channel, &self.headers)
            .accept_compressed(CompressionEncoding::Zstd)
            .pipe(|client| {
                if let Some(limit) = self.max_decoding_message_size {
                    client.max_decoding_message_size(limit)
                } else {
                    client
                }
            })
    }

    pub fn execution_client(&mut self) -> TransactionExecutionServiceClient<Channel<'_>> {
        TransactionExecutionServiceClient::with_interceptor(&mut self.channel, &self.headers)
            .accept_compressed(CompressionEncoding::Zstd)
            .pipe(|client| {
                if let Some(limit) = self.max_decoding_message_size {
                    client.max_decoding_message_size(limit)
                } else {
                    client
                }
            })
    }

    pub fn subscription_client(&mut self) -> SubscriptionServiceClient<Channel<'_>> {
        SubscriptionServiceClient::with_interceptor(&mut self.channel, &self.headers)
            .accept_compressed(CompressionEncoding::Zstd)
            .pipe(|client| {
                if let Some(limit) = self.max_decoding_message_size {
                    client.max_decoding_message_size(limit)
                } else {
                    client
                }
            })
    }
}
