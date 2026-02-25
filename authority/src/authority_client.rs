// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::tonic_gen::validator_client::ValidatorClient;
use anyhow::anyhow;
use async_trait::async_trait;
use eyre::Result;
use std::time::Duration;
use std::{collections::BTreeMap, net::SocketAddr};
use tap::TapFallible as _;
use tonic::transport::Channel;
use tonic::{IntoRequest, metadata::KeyAndValueRef};
use tracing::info;
use types::crypto::NetworkPublicKey;
use types::messages_grpc::{
    ObjectInfoRequest, ObjectInfoResponse, RawValidatorHealthRequest, RawWaitForEffectsRequest,
    SubmitTxRequest, SubmitTxResponse, SystemStateRequest, TransactionInfoRequest,
    TransactionInfoResponse, ValidatorHealthRequest, ValidatorHealthResponse,
    WaitForEffectsRequest, WaitForEffectsResponse,
};
use types::system_state::SystemState;
use types::{
    base::AuthorityName,
    checkpoints::{CheckpointRequest, CheckpointResponse},
    client::{Config, connect, connect_lazy},
    committee::CommitteeWithNetworkMetadata,
    error::{SomaError, SomaResult},
    messages_grpc::{
        HandleCertificateRequest, HandleCertificateResponse, HandleTransactionResponse,
    },
    multiaddr::Multiaddr,
    transaction::Transaction,
};
#[async_trait]
pub trait AuthorityAPI {
    /// Submits a transaction to validators for sequencing and execution.
    async fn submit_transaction(
        &self,
        request: SubmitTxRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<SubmitTxResponse, SomaError>;

    /// Waits for effects of a transaction that has been submitted to the network
    /// through the `submit_transaction` API.
    async fn wait_for_effects(
        &self,
        request: WaitForEffectsRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<WaitForEffectsResponse, SomaError>;

    /// Initiate a new transaction to a SOMA or Primary account.
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

    /// Handle Object information requests for this account.
    async fn handle_object_info_request(
        &self,
        request: ObjectInfoRequest,
    ) -> Result<ObjectInfoResponse, SomaError>;

    /// Handle Object information requests for this account.
    async fn handle_transaction_info_request(
        &self,
        request: TransactionInfoRequest,
    ) -> Result<TransactionInfoResponse, SomaError>;

    async fn handle_checkpoint(
        &self,
        request: CheckpointRequest,
    ) -> Result<CheckpointResponse, SomaError>;

    // This API is exclusively used by the benchmark code.
    // Hence it's OK to return a fixed system state type.
    async fn handle_system_state_object(
        &self,
        request: SystemStateRequest,
    ) -> Result<SystemState, SomaError>;

    /// Get validator health metrics (for latency measurement)
    async fn validator_health(
        &self,
        request: ValidatorHealthRequest,
    ) -> Result<ValidatorHealthResponse, SomaError>;
}

#[derive(Clone)]
pub struct NetworkAuthorityClient {
    client: SomaResult<ValidatorClient<Channel>>,
}

impl NetworkAuthorityClient {
    pub async fn connect(
        address: &Multiaddr,
        tls_target: NetworkPublicKey,
    ) -> anyhow::Result<Self> {
        let tls_config = soma_tls::create_rustls_client_config(
            tls_target.into_inner(),
            soma_tls::SERVER_NAME.to_string(),
            None,
        );
        let channel = types::client::connect(address, tls_config)
            .await
            .map_err(|err| anyhow!(err.to_string()))?;
        Ok(Self::new(channel))
    }

    pub fn connect_lazy(address: &Multiaddr, tls_target: NetworkPublicKey) -> Self {
        let tls_config = soma_tls::create_rustls_client_config(
            tls_target.into_inner(),
            soma_tls::SERVER_NAME.to_string(),
            None,
        );
        let client: SomaResult<_> = types::client::connect_lazy(address, tls_config)
            .map(ValidatorClient::new)
            .map_err(|err| err.to_string().into());
        Self { client }
    }

    pub fn new(channel: Channel) -> Self {
        Self { client: Ok(ValidatorClient::new(channel)) }
    }

    fn new_lazy(client: SomaResult<Channel>) -> Self {
        Self { client: client.map(ValidatorClient::new) }
    }

    pub(crate) fn client(&self) -> SomaResult<ValidatorClient<Channel>> {
        self.client.clone()
    }

    pub fn get_client_for_testing(&self) -> SomaResult<ValidatorClient<Channel>> {
        self.client()
    }
}

#[async_trait]
impl AuthorityAPI for NetworkAuthorityClient {
    /// Submits a transaction to the SOMA network for certification and execution.
    async fn submit_transaction(
        &self,
        request: SubmitTxRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<SubmitTxResponse, SomaError> {
        let mut request = request.into_raw()?.into_request();
        insert_metadata(&mut request, client_addr);

        self.client()?
            .submit_transaction(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(Into::<SomaError>::into)?
            .try_into()
    }

    async fn wait_for_effects(
        &self,
        request: WaitForEffectsRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<WaitForEffectsResponse, SomaError> {
        let raw_request: RawWaitForEffectsRequest = request.try_into()?;
        let mut request = raw_request.into_request();
        insert_metadata(&mut request, client_addr);

        self.client()?
            .wait_for_effects(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(Into::<SomaError>::into)?
            .try_into()
    }

    /// Initiate a new transfer to a SOMA or Primary account.
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

        let response =
            self.client()?.handle_certificate(request).await.map(tonic::Response::into_inner);

        response.map_err(Into::into)
    }

    async fn handle_object_info_request(
        &self,
        request: ObjectInfoRequest,
    ) -> Result<ObjectInfoResponse, SomaError> {
        self.client()?
            .object_info(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(Into::into)
    }

    /// Handle Object information requests for this account.
    async fn handle_transaction_info_request(
        &self,
        request: TransactionInfoRequest,
    ) -> Result<TransactionInfoResponse, SomaError> {
        self.client()?
            .transaction_info(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(Into::into)
    }

    /// Handle Object information requests for this account.
    async fn handle_checkpoint(
        &self,
        request: CheckpointRequest,
    ) -> Result<CheckpointResponse, SomaError> {
        self.client()?
            .checkpoint(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(Into::into)
    }

    async fn handle_system_state_object(
        &self,
        request: SystemStateRequest,
    ) -> Result<SystemState, SomaError> {
        self.client()?
            .get_system_state_object(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(Into::into)
    }

    async fn validator_health(
        &self,
        request: ValidatorHealthRequest,
    ) -> Result<ValidatorHealthResponse, SomaError> {
        let raw_request: RawValidatorHealthRequest = request.try_into()?;

        self.client()?
            .validator_health(raw_request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(Into::<SomaError>::into)?
            .try_into()
    }
}

pub fn make_network_authority_clients_with_network_config(
    committee: &CommitteeWithNetworkMetadata,
    network_config: &Config,
) -> BTreeMap<AuthorityName, NetworkAuthorityClient> {
    let mut authority_clients = BTreeMap::new();
    for (name, (_state, network_metadata)) in committee.validators() {
        let address =
            network_metadata.network_address.clone().rewrite_udp_to_tcp().rewrite_http_to_https();
        let tls_config = soma_tls::create_rustls_client_config(
            network_metadata.network_key.clone().into_inner(),
            soma_tls::SERVER_NAME.to_string(),
            None,
        );
        let maybe_channel = network_config
            .connect_lazy(&address, tls_config)
            .map_err(|e| e.to_string().into())
            .tap_err(|e| {
                tracing::error!(
                    address = %address,
                    name = %name,
                    "unable to create authority client: {e}"
                )
            });
        let client = NetworkAuthorityClient::new_lazy(maybe_channel);
        authority_clients.insert(*name, client);
    }
    authority_clients
}

pub fn make_authority_clients_with_timeout_config(
    committee: &CommitteeWithNetworkMetadata,
    connect_timeout: Duration,
    request_timeout: Duration,
) -> BTreeMap<AuthorityName, NetworkAuthorityClient> {
    let mut network_config = types::client::Config::new();
    network_config.connect_timeout = Some(connect_timeout);
    network_config.request_timeout = Some(request_timeout);
    network_config.http2_keepalive_interval = Some(connect_timeout);
    network_config.http2_keepalive_timeout = Some(connect_timeout);
    make_network_authority_clients_with_network_config(committee, &network_config)
}

fn insert_metadata<T>(request: &mut tonic::Request<T>, client_addr: Option<SocketAddr>) {
    if let Some(client_addr) = client_addr {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert("x-forwarded-for", client_addr.to_string().parse().unwrap());
        metadata.iter().for_each(|key_and_value| match key_and_value {
            KeyAndValueRef::Ascii(key, value) => {
                request.metadata_mut().insert(key, value.clone());
            }
            KeyAndValueRef::Binary(key, value) => {
                request.metadata_mut().insert_bin(key, value.clone());
            }
        });
    }
}
