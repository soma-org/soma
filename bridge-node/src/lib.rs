pub mod checkpoint_watcher;
pub mod config;
pub mod error;
pub mod eth_client;
pub mod eth_syncer;
pub mod node;
pub mod retry;
pub mod server;
pub mod types;

/// Proto types for bridge gRPC service.
/// Hand-written to avoid protoc dependency. Matches proto/bridge.proto.
pub mod proto {
    pub use crate::proto_generated::*;
}
mod proto_generated;
