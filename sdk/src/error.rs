// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

pub use rpc::api::error::RpcError;
use thiserror::Error;
use types::base::SomaAddress;
use types::digests::TransactionDigest;

pub type SomaRpcResult<T = ()> = Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    RpcError(RpcError),
    #[error(transparent)]
    BcsSerialisationError(#[from] bcs::Error),
    #[error("Failed to confirm tx status for {0:?} within {1} seconds.")]
    FailToConfirmTransactionStatus(TransactionDigest, u64),
    #[error("Data error: {0}")]
    DataError(String),
    #[error(
        "Client/Server api version mismatch, client api version : {client_version}, server api version : {server_version}"
    )]
    ServerVersionMismatch { client_version: String, server_version: String },
    #[error("Insufficient fund for address [{address}], requested amount: {amount}")]
    InsufficientFund { address: SomaAddress, amount: u128 },
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("Invalid Header key-value pair: {0}")]
    CustomHeadersError(String),
    #[error("Error initializing RPC client: {0}")]
    ClientInitError(String),
    #[error("Key error: {0}")]
    KeyError(String),
    #[error("Service not configured: {0}")]
    ServiceNotConfigured(String),
    #[error("gRPC error: {0}")]
    GrpcError(String),
    #[error("Transaction failed: {0}")]
    TransactionFailed(String),
}
