/// Proto types matching proto/bridge.proto.
/// Hand-written to avoid protoc build dependency.
/// Full gRPC server/client generation deferred to when protoc is available.

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SignatureRequest {
    #[prost(bytes = "vec", tag = "1")]
    pub action_digest: ::prost::alloc::vec::Vec<u8>,
    #[prost(bytes = "vec", tag = "2")]
    pub signature: ::prost::alloc::vec::Vec<u8>,
    #[prost(uint32, tag = "3")]
    pub signer_index: u32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SignatureResponse {
    #[prost(bool, tag = "1")]
    pub accepted: bool,
    #[prost(uint32, tag = "2")]
    pub total_signatures: u32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetSignaturesRequest {
    #[prost(bytes = "vec", tag = "1")]
    pub action_digest: ::prost::alloc::vec::Vec<u8>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetSignaturesResponse {
    #[prost(message, repeated, tag = "1")]
    pub signatures: ::prost::alloc::vec::Vec<CollectedSignature>,
    #[prost(uint64, tag = "2")]
    pub total_stake: u64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CollectedSignature {
    #[prost(uint32, tag = "1")]
    pub signer_index: u32,
    #[prost(bytes = "vec", tag = "2")]
    pub signature: ::prost::alloc::vec::Vec<u8>,
}

/// Server trait for bridge signature exchange.
/// Implementors handle signature collection from peer bridge nodes.
pub mod bridge_node_server {
    use super::*;

    #[tonic::async_trait]
    pub trait BridgeNode: Send + Sync + 'static {
        async fn submit_signature(
            &self,
            request: tonic::Request<SignatureRequest>,
        ) -> Result<tonic::Response<SignatureResponse>, tonic::Status>;

        async fn get_signatures(
            &self,
            request: tonic::Request<GetSignaturesRequest>,
        ) -> Result<tonic::Response<GetSignaturesResponse>, tonic::Status>;
    }

    // Full tonic gRPC server codegen deferred — requires protoc or tonic-prost-build.
    // The trait above is used directly in the orchestrator and tests.
    // When protoc is available, replace this file with generated output.
}
