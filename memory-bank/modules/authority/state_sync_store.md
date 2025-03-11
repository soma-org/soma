# State Sync Store

## Purpose and Scope

This document details the State Sync Store component in the Soma blockchain's Authority module. The StateSyncStore serves as a bridge between the P2P state synchronization system and the authority's persistent storage, providing a unified interface for accessing and storing synchronized state data. It enables nodes to efficiently synchronize their state from peers, particularly when recovering from crashes or joining the network as new validators.

## Key Components

### StateSyncStore

Core component that integrates state synchronization with persistent storage:

- Provides object retrieval for state sync
- Tracks commit watermarks for sync progress
- Manages committee state during sync
- Implements consensus store interface for block/commit storage

```rust
// VERIFIED-CODE: authority/src/state_sync_store.rs:21-33
#[derive(Clone)]
pub struct StateSyncStore {
    cache_traits: ExecutionCacheTraitPointers,

    committee_store: Arc<CommitteeStore>,
    commit_store: Arc<CommitStore>,
    consensus_store: Arc<dyn ConsensusStore>,
    // in memory commit watermark sequence numbers
    highest_synced_commit: Arc<Mutex<Option<CommitIndex>>>,
}
```

## Interface Implementation

### Constructor

The StateSyncStore is initialized with references to core storage components:

```rust
// VERIFIED-CODE: authority/src/state_sync_store.rs:35-45
pub fn new(
    cache_traits: ExecutionCacheTraitPointers,
    committee_store: Arc<CommitteeStore>,
    commit_store: Arc<CommitStore>,
    consensus_store: Arc<dyn ConsensusStore>,
) -> Self {
    Self {
        cache_traits,
        committee_store,
        commit_store,
        consensus_store,
        highest_synced_commit: Arc::new(Mutex::new(None)),
    }
}
```

### Object Access Methods

The StateSyncStore provides methods to access objects from storage:

```rust
// VERIFIED-CODE: authority/src/state_sync_store.rs:47-52
pub fn get_objects(&self, object_keys: &[ObjectKey]) -> Result<Vec<Option<Object>>> {
    self.cache_traits
        .object_cache_reader
        .multi_get_objects_by_key(object_keys)
        .map_err(Into::into)
}
```

### Commit Access Methods

Methods for accessing commit information:

```rust
// VERIFIED-CODE: authority/src/state_sync_store.rs:54-59
pub fn get_last_executed_commit(&self) -> Option<CommittedSubDag> {
    self.commit_store
        .get_highest_executed_commit()
        .expect("db error")
}
```

## ReadStore Implementation

The StateSyncStore implements the ReadStore trait, allowing it to serve as a source for state synchronization:

```rust
// VERIFIED-CODE: authority/src/state_sync_store.rs:62-117
impl ReadStore for StateSyncStore {
    fn get_commit_by_digest(&self, digest: &CommitDigest) -> Option<CommittedSubDag> {
        self.commit_store
            .get_commit_by_digest(digest)
            .expect("db error")
    }

    fn get_commit_by_index(&self, index: CommitIndex) -> Option<CommittedSubDag> {
        self.commit_store
            .get_commit_by_index(index)
            .expect("db error")
    }

    fn get_highest_synced_commit(&self) -> Result<CommittedSubDag, StorageError> {
        self.commit_store
            .get_highest_synced_commit()
            .map(|maybe_commit| {
                maybe_commit.expect("storage should have been initialized with genesis commit")
            })
            .map_err(Into::into)
    }

    fn get_last_commit_index_of_epoch(&self, epoch: EpochId) -> Option<CommitIndex> {
        self.commit_store.get_last_commit_index_of_epoch(epoch)
    }

    fn get_lowest_available_commit(&self) -> Result<CommitIndex, StorageError> {
        // Always start at 0 (genesis commit)
        Ok(0)
    }

    fn get_transaction(
        &self,
        digest: &TransactionDigest,
    ) -> Result<Option<Arc<VerifiedTransaction>>> {
        self.cache_traits
            .transaction_cache_reader
            .get_transaction_block(digest)
            .map_err(Into::into)
    }

    fn get_transaction_effects(
        &self,
        digest: &TransactionDigest,
    ) -> Result<Option<TransactionEffects>> {
        self.cache_traits
            .transaction_cache_reader
            .get_executed_effects(digest)
            .map_err(Into::into)
    }
}
```

## ObjectStore Implementation

The StateSyncStore implements the ObjectStore trait for retrieving objects:

```rust
// VERIFIED-CODE: authority/src/state_sync_store.rs:119-131
impl ObjectStore for StateSyncStore {
    fn get_object(&self, object_id: &ObjectID) -> Result<Option<Object>> {
        self.cache_traits.object_store.get_object(object_id)
    }

    fn get_object_by_key(&self, object_id: &ObjectID, version: Version) -> Result<Option<Object>> {
        self.cache_traits
            .object_store
            .get_object_by_key(object_id, version)
    }
}
```

## ReadCommitteeStore Implementation

Provides access to committee information for different epochs:

```rust
// VERIFIED-CODE: authority/src/state_sync_store.rs:133-138
impl ReadCommitteeStore for StateSyncStore {
    fn get_committee(&self, epoch: EpochId) -> Result<Option<Arc<Committee>>> {
        self.committee_store
            .get_committee(&epoch)
            .map_err(Into::into)
    }
}
```

## WriteStore Implementation

Enables writing synchronized data to persistent storage:

```rust
// VERIFIED-CODE: authority/src/state_sync_store.rs:140-183
impl WriteStore for StateSyncStore {
    fn insert_commit(
        &self,
        commit: CommittedSubDag,
    ) -> Result<(), types::storage::storage_error::Error> {
        // Handle next epoch committee if present
        if let Some(Some(EndOfEpochData {
            next_validator_set, ..
        })) = commit
            .get_end_of_epoch_block()
            .map(|b| b.end_of_epoch_data())
        {
            if let Some(next_validator_set) = next_validator_set {
                // Extract validator information
                let voting_rights: BTreeMap<_, _> = next_validator_set
                    .0
                    .iter()
                    .map(|(name, stake, _)| (*name, *stake))
                    .collect();

                let authorities = next_validator_set
                    .0
                    .iter()
                    .map(|(name, stake, meta)| {
                        (
                            *name,
                            Authority {
                                stake: *stake,
                                address: meta.consensus_address.clone(),
                                hostname: meta.hostname.clone(),
                                protocol_key: meta.protocol_key.clone(),
                                network_key: meta.network_key.clone(),
                                authority_key: meta.authority_key.clone(),
                            },
                        )
                    })
                    .collect();
                
                // Create new committee for next epoch
                let committee = Committee::new(
                    commit
                        .blocks
                        .last()
                        .unwrap()
                        .epoch()
                        .checked_add(1)
                        .unwrap(),
                    voting_rights,
                    authorities,
                );
                self.insert_committee(committee)?;
            }
        }

        self.commit_store.insert_commit(commit).map_err(Into::into)
    }

    fn insert_committee(&self, new_committee: Committee) -> Result<()> {
        self.committee_store
            .insert_new_committee(new_committee)
            .unwrap();
        Ok(())
    }

    fn update_highest_synced_commit(
        &self,
        commit: &CommittedSubDag,
    ) -> Result<(), types::storage::storage_error::Error> {
        let mut locked = self.highest_synced_commit.lock();
        if locked.is_some() && locked.unwrap() >= commit.commit_ref.index {
            return Ok(());
        }
        self.commit_store
            .update_highest_synced_commit(commit)
            .map_err(types::storage::storage_error::Error::custom)?;
        *locked = Some(commit.commit_ref.index);
        Ok(())
    }
}
```

## ConsensusStore Implementation

The StateSyncStore also implements the ConsensusStore trait, serving as a delegate to the underlying consensus store:

```rust
// VERIFIED-CODE: authority/src/state_sync_store.rs:185-286
impl ConsensusStore for StateSyncStore {
    fn write(
        &self,
        write_batch: types::storage::consensus::WriteBatch,
    ) -> types::error::ConsensusResult<()> {
        self.consensus_store.write(write_batch)
    }

    fn read_blocks(
        &self,
        refs: &[types::consensus::block::BlockRef],
    ) -> types::error::ConsensusResult<Vec<Option<types::consensus::block::VerifiedBlock>>> {
        self.consensus_store.read_blocks(refs)
    }

    fn contains_blocks(
        &self,
        refs: &[types::consensus::block::BlockRef],
    ) -> types::error::ConsensusResult<Vec<bool>> {
        self.consensus_store.contains_blocks(refs)
    }

    // Additional delegated methods omitted for brevity
}
```

## AccumulatorStore Implementation

Provides access to the state accumulator for cryptographic verification:

```rust
// VERIFIED-CODE: authority/src/state_sync_store.rs:288-317
impl AccumulatorStore for StateSyncStore {
    fn get_root_state_accumulator_for_epoch(
        &self,
        epoch: EpochId,
    ) -> types::error::SomaResult<Option<(CommitIndex, Accumulator)>> {
        self.cache_traits
            .accumulator_store
            .get_root_state_accumulator_for_epoch(epoch)
    }

    fn get_root_state_accumulator_for_highest_epoch(
        &self,
    ) -> types::error::SomaResult<Option<(EpochId, (CommitIndex, Accumulator))>> {
        self.cache_traits
            .accumulator_store
            .get_root_state_accumulator_for_highest_epoch()
    }

    fn insert_state_accumulator_for_epoch(
        &self,
        epoch: EpochId,
        commit: &CommitIndex,
        acc: &Accumulator,
    ) -> types::error::SomaResult {
        self.cache_traits
            .accumulator_store
            .insert_state_accumulator_for_epoch(epoch, commit, acc)
    }

    fn iter_live_object_set(&self) -> Box<dyn Iterator<Item = types::object::LiveObject> + '_> {
        self.cache_traits.accumulator_store.iter_live_object_set()
    }
}
```

## Integration with P2P State Sync

The StateSyncStore is a critical component in the P2P state synchronization system:

```mermaid
flowchart TD
    P2P[P2P Module] <-->|CommittedSubDag| SSS[StateSyncStore]
    SSS -->|insert_commit| CS[CommitStore]
    SSS -->|update_highest_synced_commit| CS
    SSS -->|insert_committee| CmS[CommitteeStore]
    SSS -->|access objects| ObjCache[Object Cache]
    SSS -->|access transactions| TxCache[Transaction Cache]
    SSS -->|access state digests| Acc[Accumulator Store]
    
    subgraph "Storage Interfaces"
        CS
        CmS
        ObjCache
        TxCache
        Acc
    end
    
    subgraph "Cache Access"
        ObjCache --> Storage[Persistent Storage]
        TxCache --> Storage
    end
```

## Key Workflows

### State Synchronization Flow

When a node is synchronizing state from peers:

1. The P2P module receives CommittedSubDag objects from peers
2. CommittedSubDags are verified and passed to StateSyncStore
3. StateSyncStore inserts the commits into the CommitStore
4. Updates the highest synced commit watermark
5. Extracts and stores committee information for epoch changes
6. Provides access to objects, transactions, and effects

### End of Epoch Handling

The StateSyncStore plays a key role in epoch transitions:

```rust
// VERIFIED-CODE: authority/src/state_sync_store.rs:142-169
if let Some(Some(EndOfEpochData {
    next_validator_set, ..
})) = commit
    .get_end_of_epoch_block()
    .map(|b| b.end_of_epoch_data())
{
    if let Some(next_validator_set) = next_validator_set {
        // Extract committee information and create new committee
        let committee = Committee::new(
            commit
                .blocks
                .last()
                .unwrap()
                .epoch()
                .checked_add(1)
                .unwrap(),
            voting_rights,
            authorities,
        );
        self.insert_committee(committee)?;
    }
}
```

This ensures that:
1. The node has the correct committee information for the next epoch
2. State synchronization can continue across epoch boundaries
3. The node can properly validate transactions in the new epoch

## Thread Safety

The StateSyncStore is designed for thread safety:

1. **Cloneable Interface**: Derives Clone for sharing across threads
2. **Arc References**: Uses Arc for shared component references
3. **Internal Mutex**: Uses a mutex for protecting the highest_synced_commit
4. **Delegate Threading**: Relies on the thread safety of underlying components

## Verification Status

| Component | Status | Confidence |
|-----------|--------|------------|
| StateSyncStore definition | Verified-Code | 9/10 |
| ReadStore implementation | Verified-Code | 9/10 |
| WriteStore implementation | Verified-Code | 9/10 |
| ObjectStore implementation | Verified-Code | 9/10 |
| ConsensusStore implementation | Verified-Code | 9/10 |
| AccumulatorStore implementation | Verified-Code | 9/10 |

## Confidence: 9/10

This documentation provides a comprehensive and accurate description of the State Sync Store component based on direct code inspection. The component interfaces, workflows, and integration with the P2P module are accurately represented with evidence from the codebase.

## Last Updated: 3/8/2025
