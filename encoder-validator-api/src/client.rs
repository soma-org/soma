use std::net::SocketAddr;

use crate::tonic_gen::encoder_validator_api_client::EncoderValidatorApiClient;
use anyhow::anyhow;
use async_trait::async_trait;
use tonic::{IntoRequest, metadata::KeyAndValueRef, transport::Channel};
use types::{
    client::{connect, connect_lazy},
    encoder_validator::{FetchCommitteesRequest, FetchCommitteesResponse},
    error::{SomaError, SomaResult},
    multiaddr::Multiaddr,
};

#[async_trait]
pub trait EncoderValidatorAPI {
    /// Initiate a new transaction to a Sui or Primary account.
    async fn fetch_committees(
        &self,
        request: FetchCommitteesRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<FetchCommitteesResponse, SomaError>;
}

#[derive(Clone)]
pub struct EncoderValidatorClient {
    client: SomaResult<EncoderValidatorApiClient<Channel>>,
}

impl EncoderValidatorClient {
    pub async fn connect(address: &Multiaddr) -> anyhow::Result<Self> {
        let channel = connect(address)
            .await
            .map_err(|err| anyhow!(err.to_string()))?;
        Ok(Self::new(channel))
    }

    pub fn connect_lazy(address: &Multiaddr) -> Self {
        let client: SomaResult<_> = connect_lazy(address)
            .map(EncoderValidatorApiClient::new)
            .map_err(|err| err.to_string().into());
        Self { client }
    }

    pub fn new(channel: Channel) -> Self {
        Self {
            client: Ok(EncoderValidatorApiClient::new(channel)),
        }
    }

    fn new_lazy(client: SomaResult<Channel>) -> Self {
        Self {
            client: client.map(EncoderValidatorApiClient::new),
        }
    }

    fn client(&self) -> SomaResult<EncoderValidatorApiClient<Channel>> {
        self.client.clone()
    }
}

#[async_trait]
impl EncoderValidatorAPI for EncoderValidatorClient {
    async fn fetch_committees(
        &self,
        request: FetchCommitteesRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<FetchCommitteesResponse, SomaError> {
        let mut request = request.into_request();
        insert_metadata(&mut request, client_addr);

        let response = self
            .client()?
            .fetch_committees(request)
            .await
            .map(tonic::Response::into_inner);

        response.map_err(Into::into)
    }
}

fn insert_metadata<T>(request: &mut tonic::Request<T>, client_addr: Option<SocketAddr>) {
    if let Some(client_addr) = client_addr {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert("x-forwarded-for", client_addr.to_string().parse().unwrap());
        metadata
            .iter()
            .for_each(|key_and_value| match key_and_value {
                KeyAndValueRef::Ascii(key, value) => {
                    request.metadata_mut().insert(key, value.clone());
                }
                KeyAndValueRef::Binary(key, value) => {
                    request.metadata_mut().insert_bin(key, value.clone());
                }
            });
    }
}
