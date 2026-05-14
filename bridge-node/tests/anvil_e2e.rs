//! Anvil end-to-end: prove the bridge node's outbound relayer
//! actually moves USDC on Ethereum.
//!
//! ```text
//!   anvil (local Eth chain)
//!     │
//!     ▼
//!   Foundry-built artifacts deployed via alloy (MockUSDC,
//!   BridgeCommittee, BridgeVault, BridgeLimiter, SomaBridge)
//!     │
//!     ▼
//!   pre-fund the vault with USDC; ownership transferred to bridge
//!     │
//!     ▼
//!   build a quorum-signed BridgeAction::Withdrawal using the same
//!   ECDSA scheme the off-chain bridge node uses
//!     │
//!     ▼
//!   EthSubmitter::submit_withdrawal → eth_sendRawTransaction
//!     │
//!     ▼
//!   tx receipt is success + USDC balance of the recipient grew by `amount`
//! ```
//!
//! This is the "evm side actually accepts what soma side signs" proof.
//! If this passes, the wire format, ABI encoding, signing, and on-chain
//! verification all agree end-to-end.

use std::path::PathBuf;

use alloy::node_bindings::Anvil;
use alloy::primitives::{Address, Bytes, U256};
use alloy::providers::{Provider, ProviderBuilder};
use alloy::signers::SignerSync;
use alloy::signers::local::PrivateKeySigner;
use alloy::sol;
use bridge_node::eth_submitter::EthSubmitter;
use bridge_node::eth_wallet::EthWallet;
use bridge_node::outbound_relayer::OutboundWithdrawal;
use bridge_node::types::BridgeAction;
use types::base::SomaAddress;
use types::bridge::{
    BridgeChainId, BridgePubkey, BridgeSignature, USDC_TOKEN_TYPE, WithdrawalCertificate,
    sign_bridge_message,
};

// ---------------------------------------------------------------------------
// Contract bindings loaded from forge artifacts.
//
// `sol!` with a JSON path lifts the ABI + bytecode at compile time so
// `MockUSDC::deploy(provider).await?` is just a function call. Relative
// paths are rooted at CARGO_MANIFEST_DIR (= `bridge-node/`).
// ---------------------------------------------------------------------------

// Wrap each `sol!` in a private module so the shared library types
// (`BridgeMessage`, `IBridgeCommittee`, …) live in disjoint namespaces.
// Without this the macro re-emits library types in the test module's
// scope and collides at the second invocation.
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

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/// Sentinel chain ids matching `types/src/bridge.rs::BridgeChainId`.
const ETH_CHAIN_ID: u8 = 12; // EthCustom — anvil
const SOMA_CHAIN_ID: u8 = 2; // SomaCustom

/// Full path through the bridge: deploy contracts, sign a withdrawal
/// with two test committee members (5000 BPS, clears the 3334
/// transfer threshold), submit via the bridge-node's EthSubmitter,
/// assert USDC moved.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_withdrawal_releases_usdc_on_anvil() {
    let pre = ensure_forge_artifacts_exist();
    if let Err(skip) = pre {
        eprintln!("[anvil_e2e] skipping: {skip}");
        return;
    }

    // ---- spin up anvil ----
    let anvil = Anvil::new().try_spawn().expect("anvil spawn");
    let rpc = anvil.endpoint();

    // ---- test signers: 4 committee members @ 2500 BPS each ----
    //
    // Generated deterministically so the test is reproducible. Each
    // member signs the canonical Soma bridge message bytes with
    // recoverable secp256k1 + Keccak256, matching what the off-chain
    // peer-broadcast aggregator produces. The Eth contract ecrecovers
    // the same digest.
    let committee_signers: Vec<fastcrypto::secp256k1::Secp256k1KeyPair> = (0..4)
        .map(|i| {
            use fastcrypto::traits::KeyPair;
            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::from_seed([i as u8 + 1; 32]);
            fastcrypto::secp256k1::Secp256k1KeyPair::generate(&mut rng)
        })
        .collect();

    // Derive 20-byte Eth addresses for each so the on-chain
    // BridgeCommittee can recognize their signatures via ecrecover.
    let committee_eth_addresses: Vec<Address> = committee_signers
        .iter()
        .enumerate()
        .map(|(i, kp)| {
            let pk = BridgePubkey::from_keypair(kp);
            let eth20 = types::bridge::derive_eth_address(&pk);
            let addr = Address::from_slice(&eth20);
            println!("[anvil_e2e] committee[{i}] derived addr = {:?}", addr);
            addr
        })
        .collect();

    // ---- operator (relayer) wallet — Anvil's first dev account ----
    let operator_signer = PrivateKeySigner::from_slice(&anvil.keys()[0].to_bytes()).unwrap();
    let operator = EthWallet::from_hex(&format!("0x{}", hex::encode(operator_signer.to_bytes())))
        .expect("operator wallet");
    let operator_address = operator.address();

    // Provider that signs as the operator. Used for deployment + the
    // pre-flight USDC mint + the final balance read.
    let provider = ProviderBuilder::new()
        .wallet(operator.clone().into_alloy_wallet())
        .connect_http(rpc.parse().unwrap());

    // ---- deploy contracts ----
    let usdc = MockUSDC::deploy(&provider).await.expect("deploy MockUSDC");
    println!("[anvil_e2e] MockUSDC @ {:?}", usdc.address());

    // BridgeCommittee — proxy-deployed. The impl's constructor calls
    // _disableInitializers, so direct calls to initialize on the impl
    // would revert; the proxy delegatecalls initialize into its own
    // storage, which is what we want.
    let committee_impl = BridgeCommittee::deploy(&provider)
        .await
        .expect("deploy BridgeCommittee impl");
    let stake: Vec<u16> = vec![2500, 2500, 2500, 2500];
    let committee_init_calldata = committee_impl
        .initialize(committee_eth_addresses.clone(), stake, ETH_CHAIN_ID)
        .calldata()
        .clone();
    let committee_proxy = ERC1967Proxy::deploy(
        &provider,
        *committee_impl.address(),
        committee_init_calldata,
    )
    .await
    .expect("deploy committee proxy");
    let committee = BridgeCommittee::new(*committee_proxy.address(), &provider);
    println!("[anvil_e2e] BridgeCommittee proxy @ {:?}", committee.address());

    // BridgeVault — non-upgradeable, takes USDC in constructor.
    let vault = BridgeVault::deploy(&provider, *usdc.address())
        .await
        .expect("deploy BridgeVault");
    println!("[anvil_e2e] BridgeVault @ {:?}", vault.address());

    // BridgeLimiter — proxy-deployed.
    let limiter_impl = BridgeLimiter::deploy(&provider)
        .await
        .expect("deploy BridgeLimiter impl");
    let limiter_init_calldata = limiter_impl
        .initialize(*committee.address(), 1_000_000_000_000u64)
        .calldata()
        .clone();
    let limiter_proxy = ERC1967Proxy::deploy(
        &provider,
        *limiter_impl.address(),
        limiter_init_calldata,
    )
    .await
    .expect("deploy limiter proxy");
    let limiter = BridgeLimiter::new(*limiter_proxy.address(), &provider);
    println!("[anvil_e2e] BridgeLimiter proxy @ {:?}", limiter.address());

    // SomaBridge — proxy-deployed.
    let bridge_impl = SomaBridge::deploy(&provider)
        .await
        .expect("deploy SomaBridge impl");
    let supported_chains: Vec<u8> = vec![SOMA_CHAIN_ID];
    let bridge_init_calldata = bridge_impl
        .initialize(
            *committee.address(),
            *usdc.address(),
            *vault.address(),
            *limiter.address(),
            supported_chains,
        )
        .calldata()
        .clone();
    let bridge_proxy = ERC1967Proxy::deploy(
        &provider,
        *bridge_impl.address(),
        bridge_init_calldata,
    )
    .await
    .expect("deploy bridge proxy");
    let bridge = SomaBridge::new(*bridge_proxy.address(), &provider);
    println!("[anvil_e2e] SomaBridge proxy @ {:?}", bridge.address());

    vault
        .transferOwnership(*bridge.address())
        .send()
        .await
        .unwrap()
        .watch()
        .await
        .unwrap();
    limiter
        .transferOwnership(*bridge.address())
        .send()
        .await
        .unwrap()
        .watch()
        .await
        .unwrap();

    // ---- fund the vault with USDC so there's something to release ----
    let amount: u64 = 2_000_000; // 2 USDC raw
    usdc.mint(*vault.address(), U256::from(amount))
        .send()
        .await
        .unwrap()
        .watch()
        .await
        .unwrap();

    // ---- build a quorum-signed withdrawal cert off-chain ----
    let recipient: [u8; 20] = [0xCA; 20];
    let action = BridgeAction::Withdrawal {
        nonce: 1,
        sender: SomaAddress::from([0xAA; 32]),
        target_chain: BridgeChainId::EthCustom,
        recipient_eth_address: recipient,
        token_type: USDC_TOKEN_TYPE,
        amount,
        // Use a stale timestamp so the limiter bypass fires; the
        // limiter's window is 24h so 48h+ in the past is safe.
        timestamp_ms: 1_000, // ~1970-ish — definitely mature
    };
    let msg_bytes = action.to_message_bytes();
    println!(
        "[anvil_e2e] off-chain canonical msg len={}",
        msg_bytes.len()
    );

    // Sign via the production path: same `sign_bridge_message` the
    // off-chain peer-broadcast aggregator uses. Two signers
    // (2500+2500=5000 BPS) clear the 3334 transfer threshold.
    use fastcrypto::traits::ToFromBytes;
    let mut cert = WithdrawalCertificate {
        signatures: std::collections::BTreeMap::new(),
        attached_at_epoch: 0,
    };
    for kp in &committee_signers[..2] {
        let sig = sign_bridge_message(kp, &msg_bytes);
        let bridge_sig = BridgeSignature::from_bytes(sig.as_ref()).unwrap();
        let pk = BridgePubkey::from_keypair(kp);
        cert.signatures.insert(pk, bridge_sig);
    }

    // ---- submit via the bridge-node's EthSubmitter ----
    let submitter = EthSubmitter::new(&rpc, *bridge.address(), operator)
        .expect("EthSubmitter::new");
    let withdrawal = OutboundWithdrawal {
        nonce: 1,
        recipient_eth_address: recipient,
        amount,
        created_at_ms: 1_000,
        message_bytes: msg_bytes,
        certificate: cert,
    };
    let tx_hash = submitter
        .submit_withdrawal(&withdrawal)
        .await
        .expect("submit_withdrawal");
    println!("[anvil_e2e] release tx submitted: {:?}", tx_hash);

    // Wait for receipt + assert success. EthSubmitter returns as soon
    // as the tx is accepted into the mempool; anvil normally auto-mines
    // within a few ms but there is still a tiny race, so poll briefly.
    let receipt = {
        let mut attempt = 0;
        loop {
            match provider.get_transaction_receipt(tx_hash).await.unwrap() {
                Some(r) => break r,
                None if attempt < 50 => {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    attempt += 1;
                }
                None => panic!("receipt never appeared after 2.5s of polling"),
            }
        }
    };
    assert!(receipt.status(), "release tx reverted: {:?}", receipt);

    // ---- assert USDC moved ----
    let recipient_addr = Address::from(recipient);
    let bal = usdc.balanceOf(recipient_addr).call().await.unwrap();
    assert_eq!(
        bal,
        U256::from(amount),
        "recipient should have received exactly the released amount"
    );

    let vault_bal = usdc.balanceOf(*vault.address()).call().await.unwrap();
    assert_eq!(
        vault_bal,
        U256::ZERO,
        "vault should be drained by the release"
    );

    println!("[anvil_e2e] ✓ end-to-end withdrawal completed");
}

/// Catch the common "forgot to forge build" precondition. The
/// integration test loads forge artifacts at compile time via the
/// `sol!` macro file paths, but if `forge build` has never been run
/// the JSON files don't exist and the macro errors out. This helper
/// turns that into a soft skip so `cargo test` keeps working on a
/// fresh checkout (CI runs `forge build` before tests).
fn ensure_forge_artifacts_exist() -> Result<(), String> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let required = [
        "../bridge/evm/out/MockUSDC.sol/MockUSDC.json",
        "../bridge/evm/out/BridgeCommittee.sol/BridgeCommittee.json",
        "../bridge/evm/out/BridgeVault.sol/BridgeVault.json",
        "../bridge/evm/out/BridgeLimiter.sol/BridgeLimiter.json",
        "../bridge/evm/out/SomaBridge.sol/SomaBridge.json",
    ];
    for rel in required {
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
