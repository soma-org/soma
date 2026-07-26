//! Aggregator trait + cert envelope.
//!
//! The peer-broadcast implementation lives in
//! [`crate::peer_aggregator::PeerBroadcastAggregator`]. This module
//! keeps only the trait surface + the result type so the action
//! executor's signature is decoupled from the concrete fan-out
//! strategy.
//!
//! Historical note: an in-process `LocalAggregator` lived here that
//! read from a co-located gRPC sig cache. After the switch to the
//! HTTP fetch-and-sign model in Phase 4b, that cache (and the gRPC
//! server, and the proto codegen) was retired. The trait remains so
//! tests can stub the aggregator without spinning up real peers, and
//! so a future strategy (preference-ordered fan-out, cached
//! responses, etc.) can drop in without touching the executor.

use std::collections::BTreeMap;

use types::bridge::{BridgePubkey, BridgeSignature};

use crate::error::BridgeResult;
use crate::types::BridgeAction;

/// A quorum-signed authorization to execute a `BridgeAction`. Once produced,
/// the executor constructs the appropriate on-chain tx
/// (`BridgeDeposit` for inbound, `BridgeAttachWithdrawalSignatures`
/// for outbound, etc.) and submits it.
///
/// The wire format is a labeled-pubkey envelope: each entry is a
/// `(BridgePubkey, 65-byte recoverable secp256k1 sig)` pair. The
/// on-chain verifier ecrecovers each sig and checks the recovered
/// pubkey matches the labeled one — no bitmap, signer identity falls
/// out of the recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertifiedBridgeAction {
    pub action: BridgeAction,
    pub signatures: BTreeMap<BridgePubkey, BridgeSignature>,
}

/// Pull-style sig aggregation: gather enough committee signatures to
/// meet `threshold` voting power, return the cert.
#[async_trait::async_trait]
pub trait BridgeAuthorityAggregator: Send + Sync {
    /// Block until either (a) signatures totaling `threshold` voting
    /// power have been collected — return a `CertifiedBridgeAction` —
    /// or (b) the aggregator can't reach the threshold (too many
    /// peers down or blocklisted) — return an error.
    async fn request_committee_signatures(
        &self,
        action: &BridgeAction,
        threshold: u64,
    ) -> BridgeResult<CertifiedBridgeAction>;
}
