use crate::tonic_gen::validator_client::ValidatorClient;
use anyhow::anyhow;
use async_trait::async_trait;
use eyre::Result;
use std::{collections::BTreeMap, net::SocketAddr};
use tonic::transport::Channel;
use tonic::{metadata::KeyAndValueRef, IntoRequest};
use tracing::info;
use types::{
    base::AuthorityName,
    client::{connect, connect_lazy, Config},
    committee::CommitteeWithNetworkMetadata,
    error::{SomaError, SomaResult},
    grpc::{HandleCertificateRequest, HandleCertificateResponse, HandleTransactionResponse},
    multiaddr::Multiaddr,
    transaction::Transaction,
};

#[async_trait]
pub trait AuthorityAPI {
    /// Initiate a new transaction to a  Primary account.
    async fn handle_transaction(
        &self,
        transaction: Transaction,
        client_addr: Option<SocketAddr>,
    ) -> Result<HandleTransactionResponse, SomaError>;

    /// Execute a certificate.
    async fn handle_certificate(
        &self,
        request: HandleCertificateRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<HandleCertificateResponse, SomaError>;
}

#[derive(Clone)]
pub struct NetworkAuthorityClient {
    client: SomaResult<ValidatorClient<Channel>>,
}

impl NetworkAuthorityClient {
    pub async fn connect(address: &Multiaddr) -> anyhow::Result<Self> {
        let channel = connect(address)
            .await
            .map_err(|err| anyhow!(err.to_string()))?;
        Ok(Self::new(channel))
    }

    pub fn connect_lazy(address: &Multiaddr) -> Self {
        let client: SomaResult<_> = connect_lazy(address)
            .map(ValidatorClient::new)
            .map_err(|err| err.to_string().into());
        Self { client }
    }

    pub fn new(channel: Channel) -> Self {
        Self {
            client: Ok(ValidatorClient::new(channel)),
        }
    }

    fn new_lazy(client: SomaResult<Channel>) -> Self {
        Self {
            client: client.map(ValidatorClient::new),
        }
    }

    fn client(&self) -> SomaResult<ValidatorClient<Channel>> {
        self.client.clone()
    }
}

#[async_trait]
impl AuthorityAPI for NetworkAuthorityClient {
    /// Initiate a new transfer to a Primary account.
    async fn handle_transaction(
        &self,
        transaction: Transaction,
        client_addr: Option<SocketAddr>,
    ) -> Result<HandleTransactionResponse, SomaError> {
        let mut request = transaction.into_request();
        insert_metadata(&mut request, client_addr);

        self.client()?
            .transaction(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(Into::into)
    }

    async fn handle_certificate(
        &self,
        request: HandleCertificateRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<HandleCertificateResponse, SomaError> {
        let mut request = request.into_request();
        insert_metadata(&mut request, client_addr);

        let response = self
            .client()?
            .handle_certificate(request)
            .await
            .map(tonic::Response::into_inner);

        response.map_err(Into::into)
    }
}

pub fn make_network_authority_clients_with_network_config(
    committee: &CommitteeWithNetworkMetadata,
    network_config: &Config,
) -> BTreeMap<AuthorityName, NetworkAuthorityClient> {
    let mut authority_clients = BTreeMap::new();
    for (name, (_state, network_metadata)) in committee.validators() {
        let address = network_metadata.consensus_address.clone();
        let address = address.rewrite_udp_to_tcp();
        let maybe_channel = network_config.connect_lazy(&address).map_err(|e| {
            tracing::error!(
                address = %address,
                name = %name,
                "unable to create authority client: {e}"
            );
            e.to_string().into()
        });
        let client = NetworkAuthorityClient::new_lazy(maybe_channel);
        authority_clients.insert(*name, client);
    }
    authority_clients
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
