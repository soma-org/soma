//! RocksDB-backed write-ahead log for the bridge node.
//!
//! Mirrors Sui's `BridgeOrchestratorTables`. Three concerns:
//!
//! 1. **Pending actions**: every `BridgeAction` we observed but haven't yet
//!    submitted to its destination chain lives here, keyed by its keccak256
//!    digest. On startup the orchestrator re-loads all entries and resubmits
//!    them. Without this, an off-chain crash mid-flight loses the action.
//!
//! 2. **Eth syncer cursor**: the highest finalized block we've already
//!    scanned for each watched bridge contract. Without this, every restart
//!    re-scans from genesis.
//!
//! 3. **Soma syncer cursor**: the highest checkpoint we've already processed.
//!    Same reasoning.
//!
//! All writes go through `typed_store`'s batch API so cursor advances and
//! action insertions can land atomically together — this is the
//! crash-consistency invariant that production correctness depends on.

use std::path::Path;
use std::sync::Arc;

use store::DBMapUtils;
use store::rocks::DBMap;
use store::traits::Map;

use crate::error::{BridgeError, BridgeResult};
use crate::types::BridgeAction;

/// 32-byte keccak256 digest of the canonical signed message bytes for a
/// `BridgeAction`. Doubles as the WAL key and as the gRPC sig-cache key.
pub type BridgeActionDigest = [u8; 32];

/// 20-byte Eth contract address (matches what `eth_client` parses from logs).
pub type EthContractAddress = [u8; 20];

#[derive(DBMapUtils)]
pub struct BridgeOrchestratorTables {
    /// `BridgeAction`s the orchestrator has accepted but not yet resolved.
    /// Cleared per-action after final on-chain confirmation. Keyed by the
    /// signed-message digest so we can dedupe when the eth-syncer
    /// re-emits an event after a restart.
    pub pending_actions: DBMap<BridgeActionDigest, BridgeAction>,

    /// Per-contract Eth-side cursor: the highest block we've fully scanned
    /// + processed. Restart picks up at `cursor + 1`. Without this, a
    /// restart re-scans from genesis (or `start_block_fallback`).
    pub eth_cursor: DBMap<EthContractAddress, u64>,

    /// Soma-side cursor: highest checkpoint we've processed for outbound
    /// withdrawal observation. Single global value (no contract sharding).
    pub soma_cursor: DBMap<(), u64>,
}

impl BridgeOrchestratorTables {
    /// Open (or create) the WAL at `path`. The directory is created if missing.
    /// Re-opening with the same path returns a handle on the existing data —
    /// that's the whole durability story.
    pub fn open(path: &Path) -> BridgeResult<Arc<Self>> {
        std::fs::create_dir_all(path).map_err(|e| {
            BridgeError::Internal(format!(
                "Failed to create bridge WAL directory at {}: {e}",
                path.display()
            ))
        })?;
        // None+None: default opening options + default per-table tuning.
        Ok(Arc::new(Self::open_tables_read_write(
            path.to_path_buf(),
            None,
            None,
        )))
    }

    /// Insert an action into the WAL. Idempotent: re-inserting the same
    /// digest is a no-op (typed_store's `insert` overwrites with identical
    /// bytes). Mirrors Sui's `insert_pending_actions`.
    pub fn insert_pending_action(&self, action: &BridgeAction) -> BridgeResult<()> {
        let digest = action.digest();
        self.pending_actions
            .insert(&digest, action)
            .map_err(|e| BridgeError::Internal(format!("WAL insert_pending_action: {e}")))
    }

    /// Remove an action after it's been confirmed on the destination chain.
    /// Mirrors Sui's `remove_pending_actions`.
    pub fn remove_pending_action(&self, digest: &BridgeActionDigest) -> BridgeResult<()> {
        self.pending_actions
            .remove(digest)
            .map_err(|e| BridgeError::Internal(format!("WAL remove_pending_action: {e}")))
    }

    /// Reload all WAL entries on startup so the orchestrator can resubmit
    /// them. Mirrors Sui's `get_all_pending_actions`.
    pub fn get_all_pending_actions(&self) -> BridgeResult<Vec<BridgeAction>> {
        let mut actions = Vec::new();
        for entry in self
            .pending_actions
            .safe_iter()
        {
            let (_digest, action) = entry
                .map_err(|e| BridgeError::Internal(format!("WAL iter: {e}")))?;
            actions.push(action);
        }
        Ok(actions)
    }

    /// Update the Eth-syncer cursor for a specific contract. Cursor advance
    /// must happen **after** all actions in the batch are persisted — that's
    /// the crash-consistency invariant the orchestrator depends on (Sui's
    /// `update_eth_event_cursor` works the same way).
    pub fn update_eth_cursor(
        &self,
        contract: EthContractAddress,
        block_number: u64,
    ) -> BridgeResult<()> {
        self.eth_cursor
            .insert(&contract, &block_number)
            .map_err(|e| BridgeError::Internal(format!("WAL update_eth_cursor: {e}")))
    }

    /// Read the persisted cursor for `contract`. Returns None if the
    /// contract has never been processed (first run). The orchestrator
    /// translates None → `start_block_fallback`.
    pub fn get_eth_cursor(
        &self,
        contract: &EthContractAddress,
    ) -> BridgeResult<Option<u64>> {
        self.eth_cursor
            .get(contract)
            .map_err(|e| BridgeError::Internal(format!("WAL get_eth_cursor: {e}")))
    }

    pub fn update_soma_cursor(&self, checkpoint_seq: u64) -> BridgeResult<()> {
        self.soma_cursor
            .insert(&(), &checkpoint_seq)
            .map_err(|e| BridgeError::Internal(format!("WAL update_soma_cursor: {e}")))
    }

    pub fn get_soma_cursor(&self) -> BridgeResult<Option<u64>> {
        self.soma_cursor
            .get(&())
            .map_err(|e| BridgeError::Internal(format!("WAL get_soma_cursor: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::base::SomaAddress;

    fn temp_path() -> std::path::PathBuf {
        tempfile::tempdir().expect("tempdir").keep()
    }

    fn deposit_action(nonce: u64, amount: u64) -> BridgeAction {
        BridgeAction::Deposit {
            nonce,
            eth_tx_hash: [nonce as u8; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::from([0xAA; 32]),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount,
            timestamp_ms: 0,
        }
    }

    #[tokio::test]
    async fn test_pending_actions_round_trip() {
        let path = temp_path();
        let tables = BridgeOrchestratorTables::open(&path).expect("open");

        let action = deposit_action(0, 1_000_000);
        tables.insert_pending_action(&action).unwrap();
        let loaded = tables.get_all_pending_actions().unwrap();
        assert_eq!(loaded, vec![action.clone()]);

        // Removal clears the entry.
        tables.remove_pending_action(&action.digest()).unwrap();
        assert!(tables.get_all_pending_actions().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_pending_actions_survive_reopen() {
        // The whole point of the WAL: a restart must see the same entries.
        let path = temp_path();
        let action = deposit_action(7, 5_000_000);
        {
            let t = BridgeOrchestratorTables::open(&path).unwrap();
            t.insert_pending_action(&action).unwrap();
            // drop t — closes the DB handle
        }
        let t2 = BridgeOrchestratorTables::open(&path).unwrap();
        assert_eq!(t2.get_all_pending_actions().unwrap(), vec![action]);
    }

    #[tokio::test]
    async fn test_eth_cursor_round_trip() {
        let path = temp_path();
        let t = BridgeOrchestratorTables::open(&path).unwrap();
        let contract = [0xBBu8; 20];
        assert!(t.get_eth_cursor(&contract).unwrap().is_none());
        t.update_eth_cursor(contract, 12345).unwrap();
        assert_eq!(t.get_eth_cursor(&contract).unwrap(), Some(12345));
        // Monotonic advance is the orchestrator's responsibility, not the
        // store's — the store accepts arbitrary updates.
        t.update_eth_cursor(contract, 99999).unwrap();
        assert_eq!(t.get_eth_cursor(&contract).unwrap(), Some(99999));
    }

    #[tokio::test]
    async fn test_soma_cursor_round_trip() {
        let path = temp_path();
        let t = BridgeOrchestratorTables::open(&path).unwrap();
        assert!(t.get_soma_cursor().unwrap().is_none());
        t.update_soma_cursor(42).unwrap();
        assert_eq!(t.get_soma_cursor().unwrap(), Some(42));
    }
}

/// Helper on `BridgeAction` so the WAL key derivation matches the gRPC
/// sig-cache key derivation (both are `keccak256(canonical_message_bytes)`).
impl BridgeAction {
    /// 32-byte digest of the canonical signed message bytes for this action.
    /// Doubles as both the WAL key and the gRPC sig-cache key — keeping
    /// them aligned so the two layers can be looked up by the same handle.
    pub fn digest(&self) -> BridgeActionDigest {
        use fastcrypto::hash::{HashFunction, Keccak256};
        let bytes = self.to_message_bytes();
        Keccak256::digest(&bytes).digest
    }
}
