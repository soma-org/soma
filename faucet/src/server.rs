// Portions Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// This file is derived from the Sui project (https://github.com/MystenLabs/sui),
// specifically crates/sui-faucet/src/server.rs

use crate::app_state::AppState;
use crate::faucet_gen::faucet_server::{Faucet, FaucetServer};
use crate::faucet_types::{GasCoinInfo, GasRequest, GasResponse};
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;

pub struct FaucetService {
    state: Arc<AppState>,
}

impl FaucetService {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl Faucet for FaucetService {
    async fn request_gas(
        &self,
        request: Request<GasRequest>,
    ) -> Result<Response<GasResponse>, Status> {
        let recipient = request.into_inner().recipient;

        let address = recipient
            .parse::<types::base::SomaAddress>()
            .map_err(|e| Status::invalid_argument(format!("Invalid address: {e}")))?;

        match self.state.faucet.local_request_execute_tx(address).await {
            Ok(coins) => Ok(Response::new(GasResponse {
                status: "Success".to_string(),
                coins_sent: coins
                    .into_iter()
                    .map(|c| GasCoinInfo {
                        amount: c.amount,
                        id: c.id,
                        transfer_tx_digest: c.transfer_tx_digest,
                    })
                    .collect(),
            })),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }
}

/// Start the faucet gRPC server with the given app state.
pub async fn start_faucet(app_state: Arc<AppState>) -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = format!("{}:{}", app_state.config.host_ip, app_state.config.port)
        .parse()
        .map_err(|e| format!("Invalid faucet address: {e}"))?;

    let svc = FaucetService::new(app_state);

    info!("Faucet gRPC server listening on {}", addr);

    tonic::transport::Server::builder().add_service(FaucetServer::new(svc)).serve(addr).await?;

    Ok(())
}
