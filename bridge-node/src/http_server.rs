//! HTTP REST sig-exchange server.
//!
//! Mirrors Sui's `sui-bridge/src/server/mod.rs`. Each validator's bridge
//! node hosts this server; peers fetch signatures by GETing the
//! action-specific route. All routes return `Json<SignedBridgeAction>`
//! on success or a sanitized `BridgeError` body on failure.
//!
//! The server delegates every signing decision to a
//! [`BridgeRequestHandlerTrait`] — see [`crate::handler`]. The handler
//! re-verifies the action against authoritative chain state (Eth tx
//! receipt or Soma `PendingWithdrawal`) or against the operator's
//! governance whitelist before producing a signature, so a malicious
//! peer that forges a request body (e.g. with a different recipient)
//! can't extract a sig for an unverified action.
//!
//! ## Sui parity notes
//!
//! - Token transfer endpoints are *pointers* (tx hash + log index,
//!   withdrawal nonce). Bodies aren't accepted; the server fetches the
//!   actual action data from chain.
//! - Governance endpoints encode the full action in the URL path. The
//!   handler then compares the reconstructed action's digest against
//!   the operator's pre-approved whitelist.
//! - Routes are GET-only — sig requests are idempotent reads of a
//!   pure function of (handler state, URL pointer). Caching can be
//!   layered in front of this server without protocol changes.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tokio::task::JoinHandle;
use tracing::{info, instrument};
use types::bridge::{BlocklistType, BridgeChainId, BridgePubkey};

use crate::error::BridgeError;
use crate::handler::BridgeRequestHandlerTrait;
use crate::types::{BridgeAction, SignedBridgeAction};

pub const APPLICATION_JSON: &str = "application/json";

pub const MAX_REQUEST_URI_SIZE: usize = 8 * 1024;
pub const MAX_REQUEST_BODY_SIZE: usize = 64 * 1024;

/// Hard cap on comma-separated list args in governance endpoints. A
/// rogue peer that supplies `keys=a,a,a,…` could otherwise force the
/// server through arbitrarily large allocations. Matches Sui's
/// `MAX_LIST_SIZE`.
pub const MAX_LIST_SIZE: usize = 255;

// Health + metadata
pub const PING_PATH: &str = "/ping";

// Token transfer (fetch-and-sign — Sui parity)
pub const ETH_TO_SOMA_TX_PATH: &str = "/sign/bridge_tx/eth/soma/{tx_hash}/{event_index}";
/// Soma→Eth transfers are nonce-keyed on the Soma side (the on-chain
/// `PendingWithdrawal` object id is derived from `nonce`), so unlike
/// Sui's `/sui/eth/{tx_digest}/{event_index}` we only need the nonce.
pub const SOMA_TO_ETH_WITHDRAWAL_PATH: &str = "/sign/bridge_action/soma/eth/{nonce}";

// Governance (operator pre-authorized — Sui parity)
pub const EMERGENCY_BUTTON_PATH: &str = "/sign/emergency_button/{nonce}/{type}";
pub const COMMITTEE_BLOCKLIST_UPDATE_PATH: &str =
    "/sign/update_committee_blocklist/{chain_id}/{nonce}/{type}/{keys}";
pub const LIMIT_UPDATE_PATH: &str =
    "/sign/update_limit/{chain_id}/{nonce}/{sending_chain_id}/{new_usd_limit}";
pub const EVM_CONTRACT_UPGRADE_PATH: &str =
    "/sign/upgrade_evm_contract/{chain_id}/{nonce}/{proxy_address}/{new_impl_address}";
pub const EVM_CONTRACT_UPGRADE_PATH_WITH_CALLDATA: &str =
    "/sign/upgrade_evm_contract/{chain_id}/{nonce}/{proxy_address}/{new_impl_address}/{calldata}";

/// Committee-update route. `members` is a comma-separated list of
/// `pubkey_hex:power` pairs (e.g.
/// `0x02ab…:5000,0x03cd…:5000`). Each pubkey is a 33-byte compressed
/// secp256k1 (66 hex chars + `0x`); each power is a `u64`. The handler
/// reconstructs the action and runs it through the governance
/// whitelist before signing — same security model as the other
/// governance endpoints.
pub const COMMITTEE_UPDATE_PATH: &str =
    "/sign/update_committee/{nonce}/{members}";

/// Public metadata exposed at [`PING_PATH`]. Mirrors Sui's
/// `BridgeNodePublicMetadata`. Currently minimal — extends as we add
/// metrics auth or runtime feature flags.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BridgeNodePublicMetadata {
    pub version: &'static str,
}

impl BridgeNodePublicMetadata {
    pub fn new(version: &'static str) -> Self {
        Self { version }
    }

    pub fn empty_for_testing() -> Self {
        Self { version: "testing" }
    }
}

/// Server router state. A trio so that the public-metadata endpoint
/// can be served without holding the handler in scope, mirroring
/// Sui's `(handler, metrics, metadata)` triple. Metrics are TODO.
type AppState<H> = (Arc<H>, Arc<BridgeNodePublicMetadata>);

/// Spawn the HTTP REST server. Returns the [`JoinHandle`]; the caller
/// keeps it alive for the lifetime of the bridge node and aborts on
/// shutdown.
pub fn run_server<H>(
    socket_address: &SocketAddr,
    handler: H,
    metadata: Arc<BridgeNodePublicMetadata>,
) -> JoinHandle<()>
where
    H: BridgeRequestHandlerTrait + 'static,
{
    let socket_address = *socket_address;
    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(socket_address)
            .await
            .expect("bridge HTTP server bind failed");
        info!(addr = %socket_address, "bridge HTTP server listening");
        axum::serve(
            listener,
            make_router(Arc::new(handler), metadata).into_make_service(),
        )
        .await
        .expect("bridge HTTP server crashed");
    })
}

/// Build the axum router with all sig-exchange routes wired to
/// `handler`. Exposed (rather than only `run_server`) so tests can
/// drive routes in-process via `tower::ServiceExt::oneshot`.
pub fn make_router<H>(
    handler: Arc<H>,
    metadata: Arc<BridgeNodePublicMetadata>,
) -> Router
where
    H: BridgeRequestHandlerTrait + 'static,
{
    Router::new()
        .route("/", get(health_check))
        .route(PING_PATH, get(ping::<H>))
        .route(ETH_TO_SOMA_TX_PATH, get(handle_eth_tx_hash::<H>))
        .route(
            SOMA_TO_ETH_WITHDRAWAL_PATH,
            get(handle_soma_withdrawal::<H>),
        )
        .route(EMERGENCY_BUTTON_PATH, get(handle_emergency::<H>))
        .route(
            COMMITTEE_BLOCKLIST_UPDATE_PATH,
            get(handle_update_committee_blocklist::<H>),
        )
        .route(LIMIT_UPDATE_PATH, get(handle_limit_update::<H>))
        .route(
            EVM_CONTRACT_UPGRADE_PATH,
            get(handle_evm_contract_upgrade::<H>),
        )
        .route(
            EVM_CONTRACT_UPGRADE_PATH_WITH_CALLDATA,
            get(handle_evm_contract_upgrade_with_calldata::<H>),
        )
        .route(COMMITTEE_UPDATE_PATH, get(handle_committee_update::<H>))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_SIZE))
        .layer(middleware::from_fn(reject_oversized_uri))
        .with_state((handler, metadata))
}

/// Reject requests with absurdly long URIs before dispatch. Matches
/// Sui's `reject_oversized_uri`; without this, the comma-separated-list
/// governance endpoints could be a DoS surface.
async fn reject_oversized_uri(req: Request, next: Next) -> Response {
    let uri_len = req
        .uri()
        .path_and_query()
        .map(|v| v.as_str().len())
        .unwrap_or(0);
    if uri_len > MAX_REQUEST_URI_SIZE {
        return StatusCode::URI_TOO_LONG.into_response();
    }
    next.run(req).await
}

/// Validate a comma-separated path argument doesn't exceed
/// [`MAX_LIST_SIZE`] entries. Prevents a malformed `keys=a,a,…,a` URL
/// from being parsed into an oversized allocation downstream.
fn validate_list_size(list_str: &str, field_name: &str) -> Result<(), BridgeError> {
    let count = list_str.split(',').count();
    if count > MAX_LIST_SIZE {
        return Err(BridgeError::InvalidBridgeClientRequest(format!(
            "{field_name} list size {count} exceeds maximum allowed size of {MAX_LIST_SIZE}"
        )));
    }
    Ok(())
}

fn parse_chain_id(byte: u8) -> Result<BridgeChainId, BridgeError> {
    BridgeChainId::from_u8(byte).ok_or_else(|| {
        BridgeError::InvalidBridgeClientRequest(format!("Invalid chain id byte: {byte}"))
    })
}

fn parse_blocklist_type(byte: u8) -> Result<BlocklistType, BridgeError> {
    match byte {
        0 => Ok(BlocklistType::Blocklist),
        1 => Ok(BlocklistType::Unblocklist),
        n => Err(BridgeError::InvalidBridgeClientRequest(format!(
            "Invalid blocklist type byte: {n}"
        ))),
    }
}

fn parse_eth_address(s: &str) -> Result<[u8; 20], BridgeError> {
    let stripped = s.strip_prefix("0x").unwrap_or(s);
    let raw = hex::decode(stripped)
        .map_err(|e| BridgeError::InvalidBridgeClientRequest(format!("Invalid hex: {e}")))?;
    if raw.len() != 20 {
        return Err(BridgeError::InvalidBridgeClientRequest(format!(
            "Eth address must be 20 bytes, got {}",
            raw.len()
        )));
    }
    let mut out = [0u8; 20];
    out.copy_from_slice(&raw);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async fn health_check() -> StatusCode {
    StatusCode::OK
}

async fn ping<H>(
    State((_, metadata)): State<AppState<H>>,
) -> Json<Arc<BridgeNodePublicMetadata>>
where
    H: BridgeRequestHandlerTrait,
{
    Json(metadata)
}

#[instrument(level = "info", skip_all, fields(tx_hash = %tx_hash_hex, event_idx))]
async fn handle_eth_tx_hash<H>(
    Path((tx_hash_hex, event_idx)): Path<(String, u16)>,
    State((handler, _)): State<AppState<H>>,
) -> Result<Json<SignedBridgeAction>, BridgeError>
where
    H: BridgeRequestHandlerTrait,
{
    let signed = handler.handle_eth_tx_hash(tx_hash_hex, event_idx).await?;
    Ok(Json(signed))
}

#[instrument(level = "info", skip_all, fields(nonce))]
async fn handle_soma_withdrawal<H>(
    Path(nonce): Path<u64>,
    State((handler, _)): State<AppState<H>>,
) -> Result<Json<SignedBridgeAction>, BridgeError>
where
    H: BridgeRequestHandlerTrait,
{
    let signed = handler.handle_soma_withdrawal(nonce).await?;
    Ok(Json(signed))
}

#[instrument(level = "info", skip_all, fields(nonce, action_type))]
async fn handle_emergency<H>(
    Path((nonce, action_type)): Path<(u64, u8)>,
    State((handler, _)): State<AppState<H>>,
) -> Result<Json<SignedBridgeAction>, BridgeError>
where
    H: BridgeRequestHandlerTrait,
{
    let action = match action_type {
        0 => BridgeAction::EmergencyPause { nonce },
        1 => BridgeAction::EmergencyUnpause { nonce },
        n => {
            return Err(BridgeError::InvalidBridgeClientRequest(format!(
                "Invalid emergency action type: {n} (0=Pause, 1=Unpause)"
            )));
        }
    };
    let signed = handler.handle_governance_action(action).await?;
    Ok(Json(signed))
}

#[instrument(level = "info", skip_all, fields(chain_id, nonce, blocklist_type, keys))]
async fn handle_update_committee_blocklist<H>(
    Path((chain_id, nonce, blocklist_type, keys)): Path<(u8, u64, u8, String)>,
    State((handler, _)): State<AppState<H>>,
) -> Result<Json<SignedBridgeAction>, BridgeError>
where
    H: BridgeRequestHandlerTrait,
{
    let chain_id = parse_chain_id(chain_id)?;
    let blocklist_type = parse_blocklist_type(blocklist_type)?;
    validate_list_size(&keys, "keys")?;
    let members = keys
        .split(',')
        .map(|s| {
            let stripped = s.strip_prefix("0x").unwrap_or(s);
            let bytes = hex::decode(stripped).map_err(|e| {
                BridgeError::InvalidBridgeClientRequest(format!("Invalid pubkey hex: {e}"))
            })?;
            BridgePubkey::from_bytes(&bytes).map_err(|e| {
                BridgeError::InvalidBridgeClientRequest(format!("Invalid pubkey: {e:?}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let action = BridgeAction::UpdateCommitteeBlocklist {
        nonce,
        chain_id,
        blocklist_type,
        members,
    };
    let signed = handler.handle_governance_action(action).await?;
    Ok(Json(signed))
}

#[instrument(
    level = "info",
    skip_all,
    fields(chain_id, nonce, sending_chain_id, new_usd_limit)
)]
async fn handle_limit_update<H>(
    Path((chain_id, nonce, sending_chain_id, new_usd_limit)): Path<(u8, u64, u8, u64)>,
    State((handler, _)): State<AppState<H>>,
) -> Result<Json<SignedBridgeAction>, BridgeError>
where
    H: BridgeRequestHandlerTrait,
{
    let chain_id = parse_chain_id(chain_id)?;
    let sending_chain_id = parse_chain_id(sending_chain_id)?;
    let action = BridgeAction::LimitUpdate {
        nonce,
        chain_id,
        sending_chain_id,
        new_usd_limit,
    };
    let signed = handler.handle_governance_action(action).await?;
    Ok(Json(signed))
}

#[instrument(level = "info", skip_all, fields(chain_id, nonce))]
async fn handle_evm_contract_upgrade<H>(
    Path((chain_id, nonce, proxy, new_impl)): Path<(u8, u64, String, String)>,
    State((handler, _)): State<AppState<H>>,
) -> Result<Json<SignedBridgeAction>, BridgeError>
where
    H: BridgeRequestHandlerTrait,
{
    upgrade_action(chain_id, nonce, &proxy, &new_impl, "")
        .and_then(|a| Ok((handler, a)))
        .map(|(h, a)| async move { h.handle_governance_action(a).await.map(Json) })?
        .await
}

#[instrument(level = "info", skip_all, fields(chain_id, nonce))]
async fn handle_evm_contract_upgrade_with_calldata<H>(
    Path((chain_id, nonce, proxy, new_impl, calldata)): Path<(u8, u64, String, String, String)>,
    State((handler, _)): State<AppState<H>>,
) -> Result<Json<SignedBridgeAction>, BridgeError>
where
    H: BridgeRequestHandlerTrait,
{
    upgrade_action(chain_id, nonce, &proxy, &new_impl, &calldata)
        .and_then(|a| Ok((handler, a)))
        .map(|(h, a)| async move { h.handle_governance_action(a).await.map(Json) })?
        .await
}

#[instrument(level = "info", skip_all, fields(nonce, member_count))]
async fn handle_committee_update<H>(
    Path((nonce, members)): Path<(u64, String)>,
    State((handler, _)): State<AppState<H>>,
) -> Result<Json<SignedBridgeAction>, BridgeError>
where
    H: BridgeRequestHandlerTrait,
{
    validate_list_size(&members, "members")?;
    let mut parsed: Vec<(BridgePubkey, u64)> = Vec::new();
    for entry in members.split(',') {
        let (pk_hex, power_str) = entry.split_once(':').ok_or_else(|| {
            BridgeError::InvalidBridgeClientRequest(format!(
                "committee member entry `{entry}` is not `pubkey_hex:power`"
            ))
        })?;
        let stripped = pk_hex.strip_prefix("0x").unwrap_or(pk_hex);
        let bytes = hex::decode(stripped).map_err(|e| {
            BridgeError::InvalidBridgeClientRequest(format!("bad pubkey hex: {e}"))
        })?;
        let pk = BridgePubkey::from_bytes(&bytes).map_err(|e| {
            BridgeError::InvalidBridgeClientRequest(format!("invalid pubkey: {e:?}"))
        })?;
        let power: u64 = power_str.parse().map_err(|e| {
            BridgeError::InvalidBridgeClientRequest(format!(
                "voting power `{power_str}` not a u64: {e}"
            ))
        })?;
        parsed.push((pk, power));
    }
    let action = BridgeAction::CommitteeUpdate {
        nonce,
        new_members: parsed,
    };
    let signed = handler.handle_governance_action(action).await?;
    Ok(Json(signed))
}

fn upgrade_action(
    chain_id: u8,
    nonce: u64,
    proxy_hex: &str,
    new_impl_hex: &str,
    calldata_hex: &str,
) -> Result<BridgeAction, BridgeError> {
    let chain_id = parse_chain_id(chain_id)?;
    let proxy = parse_eth_address(proxy_hex)?;
    let new_impl = parse_eth_address(new_impl_hex)?;
    let call_data = if calldata_hex.is_empty() {
        Vec::new()
    } else {
        let stripped = calldata_hex.strip_prefix("0x").unwrap_or(calldata_hex);
        hex::decode(stripped).map_err(|e| {
            BridgeError::InvalidBridgeClientRequest(format!("Invalid calldata hex: {e}"))
        })?
    };
    Ok(BridgeAction::EvmContractUpgrade {
        nonce,
        chain_id,
        proxy,
        new_impl,
        call_data,
    })
}

// ---------------------------------------------------------------------------
// Error → response mapping
// ---------------------------------------------------------------------------

impl IntoResponse for BridgeError {
    fn into_response(self) -> Response {
        let status = match &self {
            BridgeError::InvalidBridgeClientRequest(_)
            | BridgeError::ActionIsNotGovernanceAction
            | BridgeError::GovernanceActionIsNotApproved
            | BridgeError::MismatchedAction => StatusCode::BAD_REQUEST,
            BridgeError::TxNotFound(_)
            | BridgeError::DepositEventNotFound(_)
            | BridgeError::NoBridgeEventsInTxPosition
            | BridgeError::BridgeEventInUnrecognizedEthContract => StatusCode::NOT_FOUND,
            BridgeError::TxNotFinalized(_) => StatusCode::CONFLICT,
            BridgeError::TransientProviderError(_) => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        // Sanitize body — surface variant name only, not internal
        // details. Mirrors Sui's tag-only response: stack traces / RPC
        // URLs / etc. never reach the wire.
        let tag = match &self {
            BridgeError::InvalidBridgeClientRequest(_) => "InvalidBridgeClientRequest",
            BridgeError::ActionIsNotGovernanceAction => "ActionIsNotGovernanceAction",
            BridgeError::GovernanceActionIsNotApproved => "GovernanceActionIsNotApproved",
            BridgeError::MismatchedAction => "MismatchedAction",
            BridgeError::TxNotFound(_) => "TxNotFound",
            BridgeError::DepositEventNotFound(_) => "DepositEventNotFound",
            BridgeError::NoBridgeEventsInTxPosition => "NoBridgeEventsInTxPosition",
            BridgeError::BridgeEventInUnrecognizedEthContract => {
                "BridgeEventInUnrecognizedEthContract"
            }
            BridgeError::TxNotFinalized(_) => "TxNotFinalized",
            BridgeError::TransientProviderError(_) => "TransientProviderError",
            _ => "InternalError",
        };
        (status, format!("BridgeError::{tag}")).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::BridgeResult;
    use async_trait::async_trait;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;
    use types::base::SomaAddress;

    // ----------------------------------------------------------------------
    // A controllable stub handler for routing tests. We don't exercise the
    // real fetch-and-sign path here (covered in handler.rs) — just verify
    // the router demuxes paths to the right handler method with the right
    // parsed args.
    // ----------------------------------------------------------------------
    #[derive(Default)]
    struct StubHandler {
        /// What `handle_governance_action` last received. Tests inspect
        /// this to verify URL parsing.
        last_governance: std::sync::Mutex<Option<BridgeAction>>,
        last_eth: std::sync::Mutex<Option<(String, u16)>>,
        last_withdrawal: std::sync::Mutex<Option<u64>>,
    }

    impl StubHandler {
        fn stub_signed(action: BridgeAction) -> SignedBridgeAction {
            SignedBridgeAction {
                action,
                signer_pubkey: vec![0; 33],
                signature: vec![0; 65],
            }
        }
    }

    #[async_trait]
    impl BridgeRequestHandlerTrait for StubHandler {
        async fn handle_eth_tx_hash(
            &self,
            tx_hash_hex: String,
            event_idx: u16,
        ) -> BridgeResult<SignedBridgeAction> {
            *self.last_eth.lock().unwrap() = Some((tx_hash_hex.clone(), event_idx));
            Ok(Self::stub_signed(BridgeAction::Deposit {
                nonce: 0,
                eth_tx_hash: [0; 32],
                eth_event_idx: 0,
                sender_eth_address: [0; 20],
                target_chain: types::bridge::BridgeChainId::SomaCustom,
                recipient: SomaAddress::random(),
                token_type: types::bridge::USDC_TOKEN_TYPE,
                amount: 0,
                timestamp_ms: 0,
            }))
        }

        async fn handle_soma_withdrawal(
            &self,
            nonce: u64,
        ) -> BridgeResult<SignedBridgeAction> {
            *self.last_withdrawal.lock().unwrap() = Some(nonce);
            Ok(Self::stub_signed(BridgeAction::Withdrawal {
                nonce,
                sender: SomaAddress::random(),
                target_chain: types::bridge::BridgeChainId::EthCustom,
                recipient_eth_address: [0; 20],
                token_type: types::bridge::USDC_TOKEN_TYPE,
                amount: 0,
                timestamp_ms: 0,
            }))
        }

        async fn handle_governance_action(
            &self,
            action: BridgeAction,
        ) -> BridgeResult<SignedBridgeAction> {
            *self.last_governance.lock().unwrap() = Some(action.clone());
            Ok(Self::stub_signed(action))
        }
    }

    fn test_router() -> (Router, Arc<StubHandler>) {
        let handler = Arc::new(StubHandler::default());
        let metadata = Arc::new(BridgeNodePublicMetadata::empty_for_testing());
        let router = make_router(Arc::clone(&handler), metadata);
        (router, handler)
    }

    async fn body_string(resp: Response) -> String {
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        String::from_utf8(body.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn test_health_check_returns_200() {
        let (router, _) = test_router();
        let resp = router
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_ping_returns_metadata() {
        let (router, _) = test_router();
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/ping")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_string(resp).await;
        assert!(body.contains("\"version\":\"testing\""));
    }

    #[tokio::test]
    async fn test_eth_tx_hash_route_parses_args() {
        let (router, h) = test_router();
        let tx_hash = format!("0x{}", "ab".repeat(32));
        let resp = router
            .oneshot(
                Request::builder()
                    .uri(format!("/sign/bridge_tx/eth/soma/{tx_hash}/7"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let (got_hash, got_idx) = h.last_eth.lock().unwrap().clone().unwrap();
        assert_eq!(got_hash, tx_hash);
        assert_eq!(got_idx, 7);
    }

    #[tokio::test]
    async fn test_soma_withdrawal_route_parses_nonce() {
        let (router, h) = test_router();
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/sign/bridge_action/soma/eth/42")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(h.last_withdrawal.lock().unwrap().clone(), Some(42));
    }

    #[tokio::test]
    async fn test_emergency_pause_route() {
        let (router, h) = test_router();
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/sign/emergency_button/5/0")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        match h.last_governance.lock().unwrap().clone().unwrap() {
            BridgeAction::EmergencyPause { nonce } => assert_eq!(nonce, 5),
            other => panic!("expected EmergencyPause, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_emergency_unpause_route() {
        let (router, h) = test_router();
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/sign/emergency_button/9/1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        match h.last_governance.lock().unwrap().clone().unwrap() {
            BridgeAction::EmergencyUnpause { nonce } => assert_eq!(nonce, 9),
            other => panic!("expected EmergencyUnpause, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_emergency_invalid_type_rejected() {
        let (router, _) = test_router();
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/sign/emergency_button/1/99")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_limit_update_route() {
        let (router, h) = test_router();
        // chain_id=12 (EthCustom), nonce=3, sending=2 (SomaCustom), limit=1_000_000
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/sign/update_limit/12/3/2/1000000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        match h.last_governance.lock().unwrap().clone().unwrap() {
            BridgeAction::LimitUpdate {
                nonce,
                chain_id,
                sending_chain_id,
                new_usd_limit,
            } => {
                assert_eq!(nonce, 3);
                assert_eq!(chain_id, BridgeChainId::EthCustom);
                assert_eq!(sending_chain_id, BridgeChainId::SomaCustom);
                assert_eq!(new_usd_limit, 1_000_000);
            }
            other => panic!("expected LimitUpdate, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_evm_upgrade_no_calldata() {
        let (router, h) = test_router();
        let proxy = format!("0x{}", "11".repeat(20));
        let new_impl = format!("0x{}", "22".repeat(20));
        let resp = router
            .oneshot(
                Request::builder()
                    .uri(format!("/sign/upgrade_evm_contract/12/5/{proxy}/{new_impl}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        match h.last_governance.lock().unwrap().clone().unwrap() {
            BridgeAction::EvmContractUpgrade {
                nonce, call_data, ..
            } => {
                assert_eq!(nonce, 5);
                assert!(call_data.is_empty());
            }
            other => panic!("expected EvmContractUpgrade, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_committee_update_route_parses_members() {
        use fastcrypto::secp256k1::Secp256k1KeyPair;
        use fastcrypto::traits::KeyPair;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let (router, h) = test_router();
        let mut rng = StdRng::from_seed([7; 32]);
        let kp1 = Secp256k1KeyPair::generate(&mut rng);
        let kp2 = Secp256k1KeyPair::generate(&mut rng);
        let pk1 = BridgePubkey::from_keypair(&kp1);
        let pk2 = BridgePubkey::from_keypair(&kp2);
        let members_str = format!(
            "0x{}:5000,0x{}:5000",
            hex::encode(pk1.as_bytes()),
            hex::encode(pk2.as_bytes())
        );
        let resp = router
            .oneshot(
                Request::builder()
                    .uri(format!("/sign/update_committee/3/{members_str}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        match h.last_governance.lock().unwrap().clone().unwrap() {
            BridgeAction::CommitteeUpdate { nonce, new_members } => {
                assert_eq!(nonce, 3);
                assert_eq!(new_members.len(), 2);
                assert_eq!(new_members[0].0, pk1);
                assert_eq!(new_members[0].1, 5000);
                assert_eq!(new_members[1].0, pk2);
                assert_eq!(new_members[1].1, 5000);
            }
            other => panic!("expected CommitteeUpdate, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_committee_update_route_rejects_bad_format() {
        let (router, _) = test_router();
        // Missing `:power` part.
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/sign/update_committee/3/abcd_no_colon")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_evm_upgrade_with_calldata() {
        let (router, h) = test_router();
        let proxy = format!("0x{}", "11".repeat(20));
        let new_impl = format!("0x{}", "22".repeat(20));
        let calldata = "0xdeadbeef";
        let resp = router
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/sign/upgrade_evm_contract/12/5/{proxy}/{new_impl}/{calldata}"
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        match h.last_governance.lock().unwrap().clone().unwrap() {
            BridgeAction::EvmContractUpgrade { call_data, .. } => {
                assert_eq!(call_data, vec![0xde, 0xad, 0xbe, 0xef]);
            }
            other => panic!("expected EvmContractUpgrade, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_invalid_chain_id_returns_400() {
        let (router, _) = test_router();
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/sign/update_limit/99/1/2/1000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_oversized_uri_rejected() {
        let (router, _) = test_router();
        let big = "a".repeat(MAX_REQUEST_URI_SIZE + 1);
        let resp = router
            .oneshot(
                Request::builder()
                    .uri(format!("/ping?{big}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::URI_TOO_LONG);
    }

    #[tokio::test]
    async fn test_error_body_is_sanitized() {
        let resp = BridgeError::Internal("secret".to_string()).into_response();
        let status = resp.status();
        let body = body_string(resp).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body, "BridgeError::InternalError");
        assert!(!body.contains("secret"));
    }
}
