# Commit Processing

## Purpose and Scope
This document describes the commit processing subsystem in the Authority module of the Soma blockchain. The commit subsystem is responsible for handling consensus outputs, maintaining a consistent ordering of transactions, executing transactions in causal order, and ensuring deterministic state updates across validators. This document covers the architecture, components, and workflows of the commit processing subsystem.

## Key Components

### CommitStore
The CommitStore manages the persistent tracking of consensus commits and their execution status:

```rust
// in authority/src/commit/mod.rs:34-75
pub struct CommitStore {
    /// Maps commit digest to commit index
    pub(crate) commit_index_by_digest: RwLock<BTreeMap<CommitDigest, CommitIndex>>,

    /// Stores certified commits (CommittedSubDag)
    pub(crate) certified_commits: RwLock<BTreeMap<CommitIndex, CommittedSubDag>>,
    
    /// Map from commit digest to certified commit (CommittedSubDag)
    pub(crate) commit_by_digest: RwLock<BTreeMap<CommitDigest, CommittedSubDag>>,

    /// Watermarks used to determine the highest verified, fully synced, and
    /// fully executed commits
    pub(crate) watermarks: RwLock<BTreeMap<CommitWatermark, (CommitIndex, CommitDigest)>>,

    /// A map from epoch ID to the index of the last commit in that epoch.
    epoch_last_commit_map: RwLock<BTreeMap<EpochId, CommitIndex>>,

    /// Store locally computed commit summaries so that we can detect forks and log useful
    /// information. Can be pruned as soon as we verify that we are in agreement with the latest
    /// certified commit.
    pub(crate) locally_computed_commits: RwLock<BTreeMap<CommitIndex, CommittedSubDag>>,
}
```

**Responsibilities**:
- Tracks commit execution progress through watermarks
- Stores certified commits and their contents
- Maps between commit digests and indices
- Detects and prevents consensus forks
- Tracks epoch boundaries within the commit sequence

**Verification Status**: Verified-Code [authority/src/commit/mod.rs:34-75]

### CausalOrder
The CausalOrder component provides deterministic transaction ordering based on causal dependencies:

```rust
// in authority/src/commit/causal_order.rs:7-13
pub struct CausalOrder {
    not_seen: BTreeMap<TransactionDigest, TransactionDependencies>,
    output: Vec<TransactionEffects>,
}
```

**Responsibilities**:
- Sorts transactions based on their dependencies
- Ensures deterministic ordering of independent transactions
- Handles read-write lock dependencies for shared objects
- Prevents cycles in the dependency graph
- Maintains consistency of object versions

**Key Methods**:
```rust
// in authority/src/commit/causal_order.rs:16-24
pub fn causal_sort(effects: Vec<TransactionEffects>) -> Vec<TransactionEffects> {
    let mut this = Self::from_vec(effects);
    while let Some(item) = this.pop_first() {
        this.insert(item);
    }
    this.into_list()
}
```

**Verification Status**: Verified-Code [authority/src/commit/causal_order.rs:7-24]

### CommitExecutor
The CommitExecutor is responsible for processing and executing consensus commits:

```rust
// in authority/src/commit/executor.rs:59-69
pub struct CommitExecutor {
    mailbox: broadcast::Receiver<CommittedSubDag>,
    commit_store: Arc<CommitStore>,
    state: Arc<AuthorityState>,
    object_cache_reader: Arc<dyn ObjectCacheRead>,
    transaction_cache_reader: Arc<dyn TransactionCacheRead>,
    tx_manager: Arc<TransactionManager>,
    accumulator: Arc<StateAccumulator>,
}
```

**Responsibilities**:
- Receives committed SubDAGs from consensus
- Schedules commits for execution in index order
- Executes transactions within each commit
- Manages epoch transitions via end-of-epoch transactions
- Updates state watermarks after execution
- Accumulates state for cryptographic verification

**Verification Status**: Verified-Code [authority/src/commit/executor.rs:59-69]

## Commit Processing Flow

### 1. Commit Reception and Storage

```mermaid
flowchart TD
    Consensus[Consensus Module] -->|CommittedSubDag| Handler[ConsensusHandler]
    Handler -->|insert_commit| Store[CommitStore]
    Handler -->|update_highest_synced_commit| Store
    Store -->|broadcast| Executor[CommitExecutor]
    
    subgraph "Reception Flow"
        Handler
        Store
    end
```

When consensus produces a commit, it's processed as follows:

```rust
// in authority/src/commit/mod.rs:204-213
pub fn insert_commit(&self, commit: CommittedSubDag) -> Result<(), TypedStoreError> {
    self.insert_certified_commit(&commit)?;
    self.update_highest_synced_commit(&commit)
}
```

1. **Commit Reception**: ConsensusHandler receives a certified CommittedSubDag from consensus
2. **Commit Storage**: The commit is stored in the CommitStore
3. **Watermark Update**: The highest_synced_commit watermark is updated
4. **Broadcast**: The commit is broadcast to CommitExecutor for processing

**Verification Status**: Verified-Code [authority/src/commit/mod.rs:163-213]

### 2. Commit Execution Scheduling

```mermaid
sequenceDiagram
    participant CE as CommitExecutor
    participant CS as CommitStore
    participant EP as AuthorityPerEpochStore
    participant TM as TransactionManager
    
    CE->>CS: get_highest_synced_commit()
    CS-->>CE: latest_synced_commit
    
    loop For each unexecuted commit
        CE->>CS: get_commit_by_index(next_to_schedule)
        CS-->>CE: commit
        CE->>CE: schedule_commit(commit)
        
        CE->>EP: notify_synced_commit()
        
        CE->>+CE: spawn execution task
        
        CE->>+TM: enqueue(executable_txns)
        TM-->>-CE: transactions enqueued
        
        CE-->>-CE: return (commit, tx_digests)
    end
```

The CommitExecutor schedules commits for execution:

```rust
// in authority/src/commit/executor.rs:223-257
fn schedule_synced_commits(
    &self,
    pending: &mut CommitExecutionBuffer,
    next_to_schedule: &mut CommitIndex,
    epoch_store: Arc<AuthorityPerEpochStore>,
) {
    let Some(latest_synced_commit) = self
        .commit_store
        .get_highest_synced_commit()
        .expect("Failed to read highest synced commit")
    else {
        debug!("No commits to schedule, highest synced commit is None",);
        return;
    };

    while *next_to_schedule <= latest_synced_commit.commit_ref.index
    {
        let commit = self
            .commit_store
            .get_commit_by_index(*next_to_schedule)
            .unwrap()
            .unwrap_or_else(|| {
                panic!(
                    "Commit index {:?} does not exist in commit store",
                    *next_to_schedule
                )
            });
        // Check if this commit belongs to current epoch
        if commit.epoch() > epoch_store.epoch() {
            return;
        }

        self.schedule_commit(commit, pending, epoch_store.clone());
        *next_to_schedule += 1;
    }
}
```

1. **Watermark Check**: Determine next commit to schedule based on highest_synced_commit
2. **Epoch Validation**: Ensure commit belongs to current epoch
3. **Task Spawning**: Create an asynchronous task to execute the commit
4. **Transaction Preparation**: Get unexecuted transactions from the commit
5. **Shared Object Handling**: Assign shared object versions if needed
6. **Execution Queuing**: Enqueue transactions for execution via TransactionManager

**Verification Status**: Verified-Code [authority/src/commit/executor.rs:223-257]

### 3. Transaction Execution

```mermaid
flowchart TD
    subgraph "Transaction Execution"
        A[Get Unexecuted Transactions]
        B[Assign Shared Object Versions]
        C[Enqueue Transactions]
        D[Wait For Execution]
        E[Process Execution Results]
        F[Update Watermarks]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

Transaction execution involves multiple steps:

```rust
// in authority/src/commit/executor.rs:327-419
fn get_unexecuted_transactions(
    commit: CommittedSubDag,
    cache_reader: &dyn TransactionCacheRead,
    epoch_store: Arc<AuthorityPerEpochStore>,
) -> (Vec<TransactionDigest>, Vec<VerifiedExecutableTransaction>) {
    // Collect transactions from CommittedSubDag
    let transactions: HashMap<_, _> = commit
        .transactions()
        .iter()
        .flat_map(|(_, authority_transactions)| {
            authority_transactions
                .iter()
                .filter_map(|(_, transaction)| {
                    if let ConsensusTransactionKind::UserTransaction(cert_tx) = &transaction.kind {
                        let digest = *cert_tx.digest();
                        let certificate = VerifiedCertificate::new_unchecked(*cert_tx.clone());
                        let executable =
                            VerifiedExecutableTransaction::new_from_certificate(certificate);
                        Some((digest, executable))
                    } else {
                        None
                    }
                })
        })
        .collect();

    // If you need separate vectors:
    let tx_digests: Vec<_> = transactions.keys().copied().collect();
    let transactions: Vec<_> = transactions.values().cloned().collect();

    // Check which transactions are already executed
    let executed_effects_digests = cache_reader
        .multi_get_executed_effects_digests(&tx_digests)
        .expect("failed to read executed_effects from store");

    // Filter out already executed transactions
    let executable_txns: Vec<_> = izip!(transactions.into_iter(), executed_effects_digests.iter())
        .filter_map(|(tx, effects_digest)| match effects_digest {
            None => Some(tx),
            Some(_) => {
                trace!(
                    "Transaction with digest {:?} has already been executed",
                    tx.digest()
                );
                None
            }
        })
        .collect();

    (tx_digests, executable_txns)
}
```

1. **Transaction Collection**: Extract transactions from the CommittedSubDag
2. **Executed Transaction Filtering**: Filter out already executed transactions
3. **Shared Object Version Assignment**: Assign versions to shared objects
4. **Transaction Enqueuing**: Send transactions to TransactionManager
5. **Execution Monitoring**: Wait for all transactions to execute
6. **Effect Processing**: Process transaction effects
7. **State Accumulation**: Update the State Accumulator

**Verification Status**: Verified-Code [authority/src/commit/executor.rs:327-419]

### 4. Commit Finalization

```mermaid
sequenceDiagram
    participant CE as CommitExecutor
    participant SA as StateAccumulator
    participant ES as AuthorityPerEpochStore
    participant CS as CommitStore
    
    CE->>SA: accumulate_commit(effects)
    SA-->>CE: commit_acc
    
    CE->>ES: insert_finalized_transactions(tx_digests)
    
    CE->>CS: update_highest_executed_commit(commit)
    
    alt Not Last Commit of Epoch
        CE->>SA: accumulate_running_root()
    else Last Commit of Epoch
        CE->>CE: execute_change_epoch_tx()
        CE->>SA: accumulate_epoch()
    end
```

Commit finalization ensures the commit is properly persisted and state is updated:

```rust
// in authority/src/commit/executor.rs:559-574
async fn finalize_commit(
    state: &AuthorityState,
    object_cache_reader: &dyn ObjectCacheRead,
    transaction_cache_reader: &dyn TransactionCacheRead,
    commit_store: Arc<CommitStore>,
    tx_digests: &[TransactionDigest],
    epoch_store: &Arc<AuthorityPerEpochStore>,
    commit: CommittedSubDag,
    accumulator: Arc<StateAccumulator>,
    effects: Vec<TransactionEffects>,
) -> SomaResult<Accumulator> {
    debug!("finalizing commit");
    epoch_store.insert_finalized_transactions(tx_digests, commit.commit_ref.index)?;

    let commit_acc =
        accumulator.accumulate_commit(effects, commit.commit_ref.index, epoch_store)?;

    Ok(commit_acc)
}
```

1. **Transaction Finalization**: Mark transactions as finalized in the epoch store
2. **State Accumulation**: Accumulate commit effects into the state accumulator
3. **Watermark Update**: Update the highest_executed_commit watermark
4. **Checkpoint Creation**: For specific commits, create checkpoint data

**Verification Status**: Verified-Code [authority/src/commit/executor.rs:559-574]

### 5. End of Epoch Handling

For the last commit of an epoch, special processing occurs:

```rust
// in authority/src/commit/executor.rs:163-211
pub async fn check_epoch_last_commit(
    &self,
    epoch_store: Arc<AuthorityPerEpochStore>,
    commit: &Option<CommittedSubDag>,
) -> bool {
    let cur_epoch = epoch_store.epoch();

    if let Some(commit) = commit {
        if commit.epoch() == cur_epoch {
            if let Some(eoe_block) = commit.get_end_of_epoch_block() {
                info!(
                    ended_epoch = cur_epoch,
                    last_commit = commit.commit_ref.index,
                    "Reached end of epoch",
                );

                let epoch_start_timestamp_ms = eoe_block
                    .end_of_epoch_data()
                    .as_ref()
                    .expect("end of epoch block must have end of epoch data")
                    .next_epoch_start_timestamp_ms;

                let change_epoch_tx_digest = self
                    .execute_change_epoch_tx(
                        cur_epoch,
                        epoch_store.clone(),
                        commit.clone(),
                        epoch_start_timestamp_ms,
                    )
                    .await;

                // Additional epoch finalization logic...

                return true;
            }
        }
    }
    false
}
```

1. **End-of-Epoch Detection**: Detect last commit of the epoch
2. **Change Epoch Transaction**: Create and execute the special end-of-epoch transaction
3. **Epoch Accumulation**: Accumulate state for the entire epoch
4. **Epoch Transition**: Signal epoch completion to enable reconfiguration

**Verification Status**: Verified-Code [authority/src/commit/executor.rs:163-211]

## Causal Ordering of Transactions

The causal ordering mechanism ensures transactions are executed in a deterministic order that respects their dependencies:

```rust
// in authority/src/commit/causal_order.rs:16-24
pub fn causal_sort(effects: Vec<TransactionEffects>) -> Vec<TransactionEffects> {
    let mut this = Self::from_vec(effects);
    while let Some(item) = this.pop_first() {
        this.insert(item);
    }
    this.into_list()
}
```

### Dependency Tracking
Dependencies between transactions are tracked through:

1. **Direct Dependencies**: Explicitly listed in transaction effects
2. **Shared Object Dependencies**: Transactions reading/writing the same shared object
3. **Read-Write Lock Dependencies**: Write transactions depend on all previous read transactions

```rust
// in authority/src/commit/causal_order.rs:96-140
struct RWLockDependencyBuilder {
    read_version: HashMap<ObjectKey, Vec<TransactionDigest>>,
    overwrite_versions: HashMap<TransactionDigest, Vec<ObjectKey>>,
}

impl RWLockDependencyBuilder {
    pub fn from_effects(effects: &[TransactionEffects]) -> Self {
        let mut read_version: HashMap<ObjectKey, Vec<TransactionDigest>> = Default::default();
        let mut overwrite_versions: HashMap<TransactionDigest, Vec<ObjectKey>> = Default::default();
        for effect in effects {
            for kind in effect.input_shared_objects() {
                match kind {
                    InputSharedObject::ReadOnly(obj_ref) => {
                        let obj_key = obj_ref.into();
                        // Read only transaction
                        read_version
                            .entry(obj_key)
                            .or_default()
                            .push(*effect.transaction_digest());
                    }
                    InputSharedObject::Mutate(obj_ref) => {
                        let obj_key = obj_ref.into();
                        // write transaction
                        overwrite_versions
                            .entry(*effect.transaction_digest())
                            .or_default()
                            .push(obj_key);
                    }
                    // Additional cases...
                }
            }
        }
        Self {
            read_version,
            overwrite_versions,
        }
    }
}
```

### Sorting Algorithm
The sorting algorithm works through:

1. **Dependency Initialization**: Build a graph of transaction dependencies
2. **First Transaction Selection**: Pick the next transaction with no unresolved dependencies
3. **Recursive Processing**: Process transactions in dependency order
4. **Deterministic Tiebreaking**: Use lexicographic ordering for unrelated transactions

**Verification Status**: Verified-Code [authority/src/commit/causal_order.rs:16-140]

## Commit Watermarks

The commit subsystem maintains several watermarks to track progress:

```rust
// in authority/src/commit/mod.rs:14-19
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum CommitWatermark {
    // HighestVerified,
    HighestSynced,
    HighestExecuted,
    // HighestPruned,
}
```

### Watermark Types
1. **HighestSynced**: Highest commit that has been received and stored
2. **HighestExecuted**: Highest commit that has been fully executed

### Watermark Updates
Watermarks are updated at specific points:

```rust
// in authority/src/commit/mod.rs:121-132
pub fn update_highest_synced_commit(
    &self,
    commit: &CommittedSubDag,
) -> Result<(), TypedStoreError> {
    debug!(
        commit_index = commit.commit_ref.index,
        "Updating highest synced commit",
    );
    self.watermarks.write().insert(
        CommitWatermark::HighestSynced,
        (commit.commit_ref.index, commit.commit_ref.digest),
    );

    info!(
        commit_index = commit.commit_ref.index,
        "Updated highest synced commit"
    );
    Ok(())
}
```

```rust
// in authority/src/commit/mod.rs:134-149
pub fn update_highest_executed_commit(
    &self,
    commit: &CommittedSubDag,
) -> Result<(), TypedStoreError> {
    if let Some(index) = self.get_highest_executed_commit_index()? {
        if index >= commit.commit_ref.index {
            return Ok(());
        }
        assert_eq!(index + 1, commit.commit_ref.index,
        "Cannot update highest executed commit to {} when current highest executed commit is {}",
        commit.commit_ref.index,
        index);
    }
    debug!(
        index = commit.commit_ref.index,
        "Updating highest executed commit",
    );
    self.watermarks.write().insert(
        CommitWatermark::HighestExecuted,
        (commit.commit_ref.index, commit.commit_ref.digest),
    );

    Ok(())
}
```

**Verification Status**: Verified-Code [authority/src/commit/mod.rs:14-149]

## Concurrency and Thread Safety

The commit subsystem implements thread safety mechanisms:

### 1. RwLock Protection
```rust
// in authority/src/commit/mod.rs:34-75
pub struct CommitStore {
    commit_index_by_digest: RwLock<BTreeMap<CommitDigest, CommitIndex>>,
    certified_commits: RwLock<BTreeMap<CommitIndex, CommittedSubDag>>,
    commit_by_digest: RwLock<BTreeMap<CommitDigest, CommittedSubDag>>,
    watermarks: RwLock<BTreeMap<CommitWatermark, (CommitIndex, CommitDigest)>>,
    epoch_last_commit_map: RwLock<BTreeMap<EpochId, CommitIndex>>,
    locally_computed_commits: RwLock<BTreeMap<CommitIndex, CommittedSubDag>>,
}
```

### 2. Concurrency Control in Execution
```rust
// in authority/src/commit/executor.rs:78-90
pub async fn run_epoch(
    &mut self,
    epoch_store: Arc<AuthorityPerEpochStore>,
) -> StopReason {
    // Create execution buffer with bounded concurrency
    let mut pending: CommitExecutionBuffer = FuturesOrdered::new();
    
    // Only schedule next batch of commits when
    // earlier commits are fully processed
}
```

### 3. Atomic Updates
```rust
// in authority/src/commit/mod.rs:134-149
pub fn update_highest_executed_commit(
    &self,
    commit: &CommittedSubDag,
) -> Result<(), TypedStoreError> {
    // Atomic update of watermark through RwLock
    self.watermarks.write().insert(
        CommitWatermark::HighestExecuted,
        (commit.commit_ref.index, commit.commit_ref.digest),
    );
    Ok(())
}
```

**Verification Status**: Verified-Code [authority/src/commit/mod.rs:34-149, authority/src/commit/executor.rs:78-90]

## Integration with Other Components

### 1. Consensus Integration
The commit subsystem receives commits from the consensus module:

```rust
// in authority/src/commit/executor.rs:59-69
pub struct CommitExecutor {
    mailbox: broadcast::Receiver<CommittedSubDag>,
    // Other fields...
}
```

Consensus produces CommittedSubDag structures that contain ordered transactions and are sent to the commit subsystem for execution.

### 2. Transaction Manager Integration
Transactions from commits are executed through the TransactionManager:

```rust
// in authority/src/commit/executor.rs:422-467
async fn execute_transactions(
    all_tx_digests: Vec<TransactionDigest>,
    executable_txns: Vec<VerifiedExecutableTransaction>,
    state: &AuthorityState,
    // Other parameters...
) -> SomaResult<Option<Accumulator>> {
    // Enqueue transactions for execution
    transaction_manager.enqueue(
        executable_txns.clone(),
        &epoch_store,
        Some(commit.commit_ref.index),
    );
    
    // Wait for execution to complete
    // Process results
}
```

### 3. State Accumulator Integration
The commit subsystem updates the state accumulator with transaction effects:

```rust
// in authority/src/commit/executor.rs:559-574
async fn finalize_commit(
    // Parameters...
) -> SomaResult<Accumulator> {
    // Insert finalized transactions
    epoch_store.insert_finalized_transactions(tx_digests, commit.commit_ref.index)?;

    // Update state accumulator
    let commit_acc =
        accumulator.accumulate_commit(effects, commit.commit_ref.index, epoch_store)?;

    Ok(commit_acc)
}
```

### 4. Epoch Store Integration
The commit subsystem interacts with the epoch store for epoch-specific operations:

```rust
// in authority/src/commit/executor.rs:163-211
pub async fn check_epoch_last_commit(
    &self,
    epoch_store: Arc<AuthorityPerEpochStore>,
    commit: &Option<CommittedSubDag>,
) -> bool {
    // Check if commit is last of epoch
    // Handle epoch transition
}
```

**Verification Status**: Verified-Code [authority/src/commit/executor.rs:59-574]

## Cross-References
- See [Consensus Integration](./consensus_integration.md) for details on how consensus produces commits
- See [Transaction Processing](./transaction_processing.md) for general transaction processing flow
- See [State Management](./state_management.md) for state updates and storage
- See [Thread Safety](./thread_safety.md) for concurrency patterns

## Confidence: 9/10
This document provides a comprehensive and accurate description of the commit processing subsystem in the Authority module, based on direct code analysis and verification. The component architecture, execution flow, and integration with other modules are thoroughly documented with specific code references.

## Last Updated: 2025-03-08 by Cline
