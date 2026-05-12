//! Operator's Eth wallet for outbound relayer tx signing.
//!
//! Wraps an `alloy::signers::local::PrivateKeySigner` — the operator's
//! secp256k1 key that pays gas for + signs EIP-1559 transactions
//! submitted to the Eth bridge contract. **Distinct from the
//! validator's bridge committee key**: the latter signs bridge
//! messages whose ecrecovered pubkeys are inside the cert; this one
//! only authorizes the wrapper tx that delivers the cert on chain.
//!
//! Anyone with the cert can submit (the on-chain contract doesn't
//! gate the submitter's identity), so a bridge node losing access to
//! this key is recoverable — the operator can hand the cert to a
//! manual relayer.

use std::str::FromStr;

use alloy::network::EthereumWallet;
use alloy::signers::local::PrivateKeySigner;

use crate::error::{BridgeError, BridgeResult};

/// The operator's signing identity for outbound Eth txs.
///
/// Constructed once at bridge-node startup from a config-specified
/// private key. Cheaply cloneable — the inner `PrivateKeySigner`
/// holds an `Arc`-wrapped key.
#[derive(Debug, Clone)]
pub struct EthWallet {
    signer: PrivateKeySigner,
}

impl EthWallet {
    /// Parse a 32-byte secp256k1 private key (hex, with or without
    /// `0x` prefix) into a wallet.
    pub fn from_hex(key_hex: &str) -> BridgeResult<Self> {
        let stripped = key_hex.strip_prefix("0x").unwrap_or(key_hex);
        let signer = PrivateKeySigner::from_str(stripped)
            .map_err(|e| BridgeError::ConfigError(format!("Invalid Eth private key: {e}")))?;
        Ok(Self { signer })
    }

    /// The Eth address derived from the wallet's public key — the
    /// `from` of every tx this wallet signs.
    pub fn address(&self) -> alloy::primitives::Address {
        self.signer.address()
    }

    /// Convert into the `alloy::network::EthereumWallet` shape that
    /// `ProviderBuilder::wallet(...)` expects.
    pub fn into_alloy_wallet(self) -> EthereumWallet {
        EthereumWallet::from(self.signer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Anvil's canonical first dev account — same key Foundry's tests
    /// + this crate's integration tests use. The address is the
    /// standard `0xf39F...` that any Ethereum developer recognizes
    /// on sight.
    const ANVIL_DEV_KEY: &str =
        "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
    const ANVIL_DEV_ADDRESS: &str = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";

    #[test]
    fn wallet_address_matches_known_anvil_dev_account() {
        let w = EthWallet::from_hex(ANVIL_DEV_KEY).unwrap();
        assert_eq!(format!("{:?}", w.address()), ANVIL_DEV_ADDRESS.to_lowercase());
    }

    #[test]
    fn wallet_accepts_key_without_0x_prefix() {
        let bare = ANVIL_DEV_KEY.trim_start_matches("0x");
        let w = EthWallet::from_hex(bare).unwrap();
        assert_eq!(format!("{:?}", w.address()), ANVIL_DEV_ADDRESS.to_lowercase());
    }

    #[test]
    fn wallet_rejects_invalid_key() {
        assert!(matches!(
            EthWallet::from_hex("not-a-key"),
            Err(BridgeError::ConfigError(_))
        ));
    }
}
