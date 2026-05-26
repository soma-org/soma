//! Anvil inbound end-to-end: prove the bridge node's Eth syncer
//! correctly parses a user's `deposit()` event into the same
//! `BridgeAction::Deposit` the off-chain committee will sign.
//!
//! ```text
//!   user calls SomaBridge.deposit(somaChain, somaRecipient, amount)
//!     │
//!     ▼
//!   Eth contract locks USDC into the vault + emits TokensDeposited
//!     │
//!     ▼
//!   bridge-node's EthClient::get_finalized_bridge_action_maybe
//!     fetches the receipt, parses the log at event_idx, builds
//!     the canonical Soma BridgeAction::Deposit
//!     │
//!     ▼
//!   the action's fields must match what the user paid for —
//!   nonce, sender, destination chain, recipient, token, amount,
//!   timestamp_ms. This is the inverse proof of `anvil_e2e.rs`:
//!   that one shows we can submit a signed cert; this one shows
//!   we can read what the Eth side emitted.
//! ```

use std::path::PathBuf;

use alloy::node_bindings::Anvil;
use alloy::primitives::{Address, U256};
use alloy::providers::{Provider, ProviderBuilder};
use alloy::signers::local::PrivateKeySigner;
use bridge_node::eth_client::EthClient;
use bridge_node::types::BridgeAction;

mod mock_usdc {
    alloy::sol!(
        #[allow(missing_docs)]
        #[sol(rpc)]
        MockUSDC,
        "../bridge/evm/out/MockUSDC.sol/MockUSDC.json"
    );
}
mod bridge_committee {
    alloy::sol!(
        #[allow(missing_docs)]
        #[sol(rpc)]
        BridgeCommittee,
        "../bridge/evm/out/BridgeCommittee.sol/BridgeCommittee.json"
    );
}
mod bridge_vault {
    alloy::sol!(
        #[allow(missing_docs)]
        #[sol(rpc)]
        BridgeVault,
        "../bridge/evm/out/BridgeVault.sol/BridgeVault.json"
    );
}
mod bridge_limiter {
    alloy::sol!(
        #[allow(missing_docs)]
        #[sol(rpc)]
        BridgeLimiter,
        "../bridge/evm/out/BridgeLimiter.sol/BridgeLimiter.json"
    );
}
mod soma_bridge {
    alloy::sol!(
        #[allow(missing_docs)]
        #[sol(rpc)]
        SomaBridge,
        "../bridge/evm/out/SomaBridge.sol/SomaBridge.json"
    );
}
mod erc1967_proxy {
    alloy::sol!(
        #[allow(missing_docs)]
        #[sol(rpc)]
        ERC1967Proxy,
        "../bridge/evm/out/ERC1967Proxy.sol/ERC1967Proxy.json"
    );
}
use bridge_committee::BridgeCommittee;
use bridge_limiter::BridgeLimiter;
use bridge_vault::BridgeVault;
use erc1967_proxy::ERC1967Proxy;
use mock_usdc::MockUSDC;
use soma_bridge::SomaBridge;

const ETH_CHAIN_ID: u8 = 12; // EthCustom
const SOMA_CHAIN_ID: u8 = 2; // SomaCustom

/// User calls `deposit()` on Ethereum; the bridge node's syncer
/// reads the resulting log and reconstructs the exact
/// `BridgeAction::Deposit` the committee will sign.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_deposit_event_is_parseable_by_bridge_node() {
    if let Err(skip) = ensure_forge_artifacts_exist() {
        eprintln!("[anvil_inbound] skipping: {skip}");
        return;
    }

    let anvil = Anvil::new().try_spawn().expect("anvil spawn");
    let rpc = anvil.endpoint();

    // Anvil auto-mines on tx; for `get_finalized_bridge_action_maybe`
    // to accept it we need the contract's tx to land in a block tagged
    // `finalized`. Anvil treats the latest block as finalized by
    // default on dev chains — good.

    // operator-as-deployer wallet (anvil dev key #0)
    let deployer_signer = PrivateKeySigner::from_slice(&anvil.keys()[0].to_bytes()).unwrap();
    let deployer =
        ProviderBuilder::new().wallet(deployer_signer.clone()).connect_http(rpc.parse().unwrap());

    // user wallet (anvil dev key #1) — separate identity that calls
    // deposit() so the captured event's `sender` is unambiguously the
    // user, not the deployer.
    let user_signer = PrivateKeySigner::from_slice(&anvil.keys()[1].to_bytes()).unwrap();
    let user_address = user_signer.address();
    let user_provider =
        ProviderBuilder::new().wallet(user_signer).connect_http(rpc.parse().unwrap());

    // ---- deploy ----
    let usdc = MockUSDC::deploy(&deployer).await.unwrap();

    // Proxy-based deployment (impls have _disableInitializers in
    // their constructors, so initialize must run via delegatecall
    // through a proxy).
    let committee_impl = BridgeCommittee::deploy(&deployer).await.unwrap();
    let members: Vec<Address> =
        (0..4).map(|i| Address::from_slice(&[i as u8 + 0x10; 20])).collect();
    let stake: Vec<u16> = vec![2500, 2500, 2500, 2500];
    let committee_init = committee_impl.initialize(members, stake, ETH_CHAIN_ID).calldata().clone();
    let committee_proxy =
        ERC1967Proxy::deploy(&deployer, *committee_impl.address(), committee_init).await.unwrap();
    let committee = BridgeCommittee::new(*committee_proxy.address(), &deployer);

    let vault = BridgeVault::deploy(&deployer, *usdc.address()).await.unwrap();

    let limiter_impl = BridgeLimiter::deploy(&deployer).await.unwrap();
    let limiter_init =
        limiter_impl.initialize(*committee.address(), 1_000_000_000_000u64).calldata().clone();
    let limiter_proxy =
        ERC1967Proxy::deploy(&deployer, *limiter_impl.address(), limiter_init).await.unwrap();
    let limiter = BridgeLimiter::new(*limiter_proxy.address(), &deployer);

    let bridge_impl = SomaBridge::deploy(&deployer).await.unwrap();
    let supported: Vec<u8> = vec![SOMA_CHAIN_ID];
    let bridge_init = bridge_impl
        .initialize(
            *committee.address(),
            *usdc.address(),
            *vault.address(),
            *limiter.address(),
            supported,
        )
        .calldata()
        .clone();
    let bridge_proxy =
        ERC1967Proxy::deploy(&deployer, *bridge_impl.address(), bridge_init).await.unwrap();
    let bridge = SomaBridge::new(*bridge_proxy.address(), &deployer);
    vault.transferOwnership(*bridge.address()).send().await.unwrap().watch().await.unwrap();
    limiter.transferOwnership(*bridge.address()).send().await.unwrap().watch().await.unwrap();

    let bridge_contract_str = format!("{:?}", bridge.address());
    println!("[anvil_inbound] SomaBridge @ {bridge_contract_str}");

    // ---- user mints + approves + deposits ----
    let amount: u64 = 1_500_000;
    let soma_recipient_bytes: [u8; 32] = [
        0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, // 8
        0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // 16
        0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, // 24
        0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // 32
    ];

    // Deployer mints USDC into the user. User approves the bridge.
    usdc.mint(user_address, U256::from(amount)).send().await.unwrap().watch().await.unwrap();

    let usdc_as_user = MockUSDC::new(*usdc.address(), &user_provider);
    let bridge_as_user = SomaBridge::new(*bridge.address(), &user_provider);

    usdc_as_user
        .approve(*bridge.address(), U256::from(amount))
        .send()
        .await
        .unwrap()
        .watch()
        .await
        .unwrap();

    let pending = bridge_as_user
        .deposit(SOMA_CHAIN_ID, alloy::primitives::FixedBytes::from(soma_recipient_bytes), amount)
        .send()
        .await
        .unwrap();
    let tx_hash = *pending.tx_hash();
    let receipt = pending.get_receipt().await.unwrap();
    assert!(receipt.status(), "deposit reverted: {receipt:?}");

    // Find the TokensDeposited log among the tx's logs. There may be
    // multiple — USDC transfer events fire too — so we filter by
    // address.
    let mut event_idx: u16 = 0;
    let bridge_addr = *bridge.address();
    let mut found = false;
    for log in receipt.inner.logs() {
        if log.address() == bridge_addr {
            event_idx = log.log_index.unwrap_or(0) as u16;
            found = true;
            break;
        }
    }
    assert!(found, "no TokensDeposited log emitted by bridge");
    println!("[anvil_inbound] deposit tx={:?} event_idx={}", tx_hash, event_idx);

    // ---- bridge-node side: parse the event ----
    // Use `get_deposit_events_in_range` (the eth_syncer's production
    // path) rather than `get_finalized_bridge_action_maybe` — the
    // latter requires `eth_getBlockByNumber("finalized")` to advance
    // past the receipt block, which doesn't happen on anvil without
    // a beacon chain. The parsing logic is identical (both go
    // through `parse_deposit_log`); only the finalization gate
    // differs. Finalization is exercised in unit tests with mocked
    // `eth_getBlockByNumber` responses.
    let eth_client =
        EthClient::new(vec![rpc], &bridge_contract_str, "finalized".to_string()).await.unwrap();
    let receipt_block = receipt.block_number.expect("receipt has a block number");
    let events = eth_client
        .get_deposit_events_in_range(receipt_block, receipt_block)
        .await
        .expect("get_deposit_events_in_range");
    let event = events
        .iter()
        .find(|e| e.tx_hash == tx_hash.0)
        .expect("our deposit event was missing from the range");
    let action = event.to_bridge_action();

    // ---- assert the action matches what the user submitted ----
    match action {
        BridgeAction::Deposit {
            nonce,
            eth_tx_hash,
            eth_event_idx,
            sender_eth_address,
            target_chain,
            recipient,
            token_type,
            amount: got_amount,
            timestamp_ms,
        } => {
            assert_eq!(nonce, 0, "first deposit on a fresh contract is nonce 0");
            assert_eq!(eth_tx_hash, tx_hash.0);
            assert_eq!(eth_event_idx, event_idx);
            assert_eq!(sender_eth_address, user_address.into_array());
            assert_eq!(target_chain.as_u8(), SOMA_CHAIN_ID);
            assert_eq!(recipient.as_ref(), &soma_recipient_bytes);
            assert_eq!(token_type, types::bridge::USDC_TOKEN_TYPE);
            assert_eq!(got_amount, amount);
            assert!(timestamp_ms > 0, "block-time timestamp_ms should have been emitted (got 0)");
        }
        other => panic!("expected Deposit, got {other:?}"),
    }

    println!("[anvil_inbound] ✓ inbound parse round-trips end-to-end");
}

fn ensure_forge_artifacts_exist() -> Result<(), String> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for rel in [
        "../bridge/evm/out/MockUSDC.sol/MockUSDC.json",
        "../bridge/evm/out/BridgeCommittee.sol/BridgeCommittee.json",
        "../bridge/evm/out/BridgeVault.sol/BridgeVault.json",
        "../bridge/evm/out/BridgeLimiter.sol/BridgeLimiter.json",
        "../bridge/evm/out/SomaBridge.sol/SomaBridge.json",
    ] {
        let p = manifest.join(rel);
        if !p.exists() {
            return Err(format!(
                "forge artifact missing: {} — run `forge build` in bridge/evm/ first",
                p.display()
            ));
        }
    }
    Ok(())
}
