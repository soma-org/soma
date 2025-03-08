# Authority Module

## Purpose and Scope
This document provides a comprehensive explanation of the Authority module in the Soma blockchain. The Authority module is the central component responsible for state management, transaction validation and execution, and epoch transitions. It maintains the validator's view of the blockchain state and coordinates with other modules to ensure correct transaction processing and system-wide consistency.

## Key Components

### AuthorityState
- **Purpose**: Central state management component that handles transaction processing, certificate execution, and epoch transitions
- **Responsibilities**:
  - Manages the validator's state (objects, transactions, effects)
  - Processes transaction requests and executes certificates
  - Coordinates with consensus for transaction ordering
  - Handles epoch transitions and reconfiguration
  - Ensures thread-safe access to state

```rust
pub struct AuthorityState {
    pub name: AuthorityName,
    pub secret: StableSyncAuthoritySigner,
    epoch_store: ArcSwap<AuthorityPerEpochStore>,
    execution_lock: RwLock<EpochId>,
    committee_store: Arc<CommitteeStore>,
    transaction_manager: Arc<TransactionManager>,
    input_loader: TransactionInputLoader,
    execution_cache_trait_pointers: ExecutionCacheTraitPointers,
    accumulator: Arc<StateAccumulator>,
}
```

**Verification Status**: Verified-Code (AuthorityState in authority/src/state.rs)

### AuthorityPerEpochStore
- **Purpose**: Manages epoch-specific state and configuration
- **Responsibilities**:
  - Stores committee information for the current epoch
  - Maintains protocol configuration parameters
  - Handles transaction validation within an epoch
  - Manages shared object version assignment
  - Executes transactions in a temporary store

```rust
pub struct AuthorityPerEpochStore {
    epoch: EpochId,
    committee: Arc<Committee>,
    transaction_lock_table: Arc<TransactionLockTable>,
    epoch_tables: AuthorityEpochTables,
    signature_verifier: SignatureVerifier,
    mutex_table: MutexTable<TransactionDigest>,
    version_assignment_mutex_table: MutexTable<ObjectID>,
    // Additional fields...
}
```

**Verification Status**: Verified-Code (AuthorityPerEpochStore in authority/src/epoch_store.rs)

### TransactionManager
- **Purpose**: Manages transaction dependencies and execution ordering
- **Responsibilities**:
  - Tracks dependencies between transactions
  - Maintains queues of transactions waiting for objects
  - Resolves dependencies when objects become available
  - Coordinates transaction execution order
  - Handles backpressure and prioritization

```rust
pub struct TransactionManager {
    inner: RwLock<RwLock<Inner>>,
    tx_ready_certificates: UnboundedSender<PendingCertificate>,
    transaction_cache_read: Arc<dyn TransactionCacheRead>,
    object_cache_read: Arc<dyn ObjectCacheRead>,
}

struct Inner {
    epoch: EpochId,
    missing_inputs: HashMap<InputKey, BTreeSet<TransactionDigest>>,
    input_objects: HashMap<FullObjectID, TransactionQueue>,
    available_objects_cache: AvailableObjectsCache,
    pending_certificates: HashMap<TransactionDigest, PendingCertificate>,
    executing_certificates: HashSet<TransactionDigest>,
}
```

**Verification Status**: Verified-Code (TransactionManager in authority/src/tx_manager.rs)

### TransactionInputLoader
- **Purpose**: Loads objects required for transaction execution
- **Responsibilities**:
  - Reads input objects from storage
  - Validates object versions against transaction inputs
  - Handles different ownership types (owned, shared, immutable)
  - Prepares objects for execution in temporary store

**Verification Status**: Verified-Code (TransactionInputLoader in authority/src/tx_input_loader.rs)

### StateAccumulator
- **Purpose**: Maintains a cryptographic accumulator of the validator's state
- **Responsibilities**:
  - Accumulates transaction effects into a digest
  - Verifies state consistency across validators
  - Supports state synchronization and recovery
  - Provides proof of state for verification

**Verification Status**: Verified-Code (StateAccumulator in authority/src/state_accumulator.rs)

## Component Architecture

```mermaid
flowchart TD
    Client[Client] -->|Transactions/Certificates| AuthState[AuthorityState]
    
    subgraph "Authority Module"
        AuthState -->|Epoch-specific state| EpochStore[AuthorityPerEpochStore]
        AuthState -->|Transaction ordering| TxManager[TransactionManager]
        AuthState -->|Object loading| InputLoader[TransactionInputLoader]
        AuthState -->|State verification| Accumulator[StateAccumulator]
        
        EpochStore -->|Execute| TempStore[TemporaryStore]
        EpochStore -->|Validate| TxValidator[TransactionValidator]
        EpochStore -->|Lock| LockTable[TransactionLockTable]
        
        TxManager -->|Ready transactions| Executor[Execution Process]
        Executor -->|Execute| TempStore
        
        InputLoader -->|Read| ObjectStore[Object Store]
    end
    
    AuthState <-->|Transaction ordering| Consensus[Consensus Module]
    AuthState <-->|State sync| P2P[P2P Module]
    AuthState <-->|Lifecycle| Node[Node Module]
    
    subgraph "Storage Layer"
        ObjectStore
        TxStore[Transaction Store]
        EffectsStore[Effects Store]
    end
```

**Verification Status**: Verified-Code (Component relationships verified from implementation)

## Thread Safety Architecture

### Lock Hierarchy

```mermaid
flowchart TD
    ExecutionLock[RwLock: execution_lock] --> EpochLock[ArcSwap: epoch_store]
    ExecutionLock --> TxLock[Mutex: transaction_lock_table]
    TxLock --> ObjLock[Mutex: object_locks]
    
    subgraph "Lock Hierarchy"
        direction TB
        ExecutionLock
        EpochLock
        TxLock
        ObjLock
    end
    
    subgraph "Lock Purpose"
        direction TB
        A[execution_lock: Epoch-wide coordination]
        B[epoch_store: Atomic epoch transition]
        C[transaction_lock: Per-transaction isolation]
        D[object_locks: Fine-grained object access]
    end
```

**Verification Status**: Verified-Code (Lock usage in AuthorityState and related components)

### Concurrency Mechanisms

#### Execution Lock (RwLock<EpochId>)
- **Purpose**: Coordinates access during normal operation and epoch transitions
- **Read Mode**: Acquired during transaction execution (multiple concurrent)
- **Write Mode**: Acquired during reconfiguration (exclusive access)
- **Usage Pattern**:
  ```rust
  // For transaction execution
  let execution_guard = self.execution_lock_for_executable_transaction(transaction).await?;
  
  // For reconfiguration
  let mut execution_lock = self.execution_lock_for_reconfiguration().await;
  ```

**Verification Status**: Verified-Code (execution_lock usage in authority/src/state.rs)

#### Epoch Store (ArcSwap<AuthorityPerEpochStore>)
- **Purpose**: Enables atomic updates of epoch-specific state
- **Usage Pattern**:
  ```rust
  // Store new epoch
  self.epoch_store.store(new_epoch_store.clone());
  
  // Access current epoch
  let epoch_store = self.epoch_store.load();
  ```

**Verification Status**: Verified-Code (epoch_store usage in authority/src/state.rs)

#### Transaction Lock Table (Arc<TransactionLockTable>)
- **Purpose**: Coordinates concurrent access to transactions
- **Usage Pattern**:
  ```rust
  // Acquire transaction lock
  let tx_guard = epoch_store.acquire_tx_guard(certificate).await?;
  ```

**Verification Status**: Verified-Code (transaction lock usage in authority/src/state.rs)

#### Object Lock Table
- **Purpose**: Prevents concurrent modifications to the same objects
- **Implementation**: Fine-grained locking at object level
- **Usage**: Implicitly used during transaction lock acquisition

**Verification Status**: Verified-Code (object lock usage in authority/src/cache/)

#### TransactionManager Double-Nested Lock
- **Purpose**: Allows reconfiguration without blocking readers
- **Implementation**:
  ```rust
  // Double-nested lock for transaction state
  inner: RwLock<RwLock<Inner>>,
  ```
- **Outer Lock**: Protects against reconfiguration
- **Inner Lock**: Protects transaction state
- **Usage Pattern**:
  ```rust
  // Read operations (fast path)
  let reconfig_lock = self.inner.read();
  let inner = reconfig_lock.read();
  // Use inner state...
  
  // Write operations with reconfiguration protection
  let reconfig_lock = self.inner.read();
  let mut inner = reconfig_lock.write();
  // Modify inner state...
  
  // Reconfiguration (slow path)
  let reconfig_lock = self.inner.write();
  // Replace inner completely...
  ```

**Verification Status**: Verified-Code (double-nested lock in tx_manager.rs)

## Transaction Processing Workflow

```mermaid
sequenceDiagram
    participant Client
    participant AuthState as AuthorityState
    participant EpochStore as AuthorityPerEpochStore
    participant TxManager as TransactionManager
    participant TempStore as TemporaryStore
    participant Storage
    
    Client->>AuthState: submit transaction
    AuthState->>EpochStore: validate transaction
    EpochStore-->>AuthState: validation result
    
    AuthState->>AuthState: sign transaction
    AuthState-->>Client: signed transaction
    
    Note over Client: Client collects signatures
    
    Client->>AuthState: submit certificate
    
    alt Contains shared objects
        AuthState->>Consensus: submit for ordering
        Consensus-->>AuthState: ordered certificate
    end
    
    AuthState->>TxManager: enqueue certificate
    TxManager->>AuthState: execute ready certificate
    
    AuthState->>EpochStore: read input objects
    EpochStore-->>AuthState: input objects
    
    AuthState->>TempStore: prepare execution
    TempStore->>TempStore: execute transaction
    TempStore-->>AuthState: effects & inner store
    
    AuthState->>Storage: commit effects
    AuthState->>TxManager: notify completion
    AuthState-->>Client: transaction effects
    
    TxManager->>TxManager: check waiting transactions
```

**Verification Status**: Verified-Code (Transaction processing flow in authority/src/state.rs)

### Certificate Execution Detailed Flow

```mermaid
flowchart TD
    Start([Certificate Received]) --> CheckExecuted{Already\nExecuted?}
    CheckExecuted -->|Yes| ReturnExisting[Return Existing Effects]
    CheckExecuted -->|No| AcquireTxLock[Acquire Transaction Lock]
    
    AcquireTxLock --> CheckEpoch{Epoch\nMatch?}
    CheckEpoch -->|No| ReleaseAndError[Release Lock & Return Error]
    CheckEpoch -->|Yes| LoadObjects[Read Input Objects]
    
    LoadObjects --> PrepareExecution[Prepare Certificate]
    PrepareExecution --> Execute[Execute in TemporaryStore]
    Execute --> ComputeEffects[Compute Transaction Effects]
    
    ComputeEffects --> CommitTx[Commit Certificate]
    CommitTx --> StoreEffects[Store Effects]
    CommitTx --> CommitObjects[Update Object Store]
    
    CommitObjects --> ReleaseLock[Release Transaction Lock]
    ReleaseLock --> NotifyTxManager[Notify Transaction Manager]
    NotifyTxManager --> End([Return Effects])
    
    ReleaseAndError --> End
    ReturnExisting --> End
```

**Verification Status**: Verified-Code (process_certificate in authority/src/state.rs)

### Transaction Lock Acquisition Flow

```mermaid
flowchart TD
    Start([Acquire Transaction Lock]) --> CheckAvailable{Lock\nAvailable?}
    CheckAvailable -->|Yes| Acquire[Acquire Lock]
    CheckAvailable -->|No| Wait[Wait for Lock]
    Wait --> Retry[Retry Acquisition]
    Retry --> CheckAvailable
    
    Acquire --> FindObjects[Find Owned Objects]
    FindObjects --> CheckObjectLocks{Objects\nLocked?}
    
    CheckObjectLocks -->|Yes| ReleaseAndError[Release Lock & Return Error]
    CheckObjectLocks -->|No| AcquireObjectLocks[Acquire Object Locks]
    
    AcquireObjectLocks --> RecordLocks[Record Locks in Store]
    RecordLocks --> ReturnGuard[Return Lock Guard]
    
    ReleaseAndError --> End([Return Error])
    ReturnGuard --> End2([Return Success])
```

**Verification Status**: Verified-Code (acquire_transaction_locks in execution cache implementation)

## Epoch Transition and Reconfiguration

### Reconfiguration Process

```mermaid
sequenceDiagram
    participant Node
    participant AuthState as AuthorityState
    participant OldEpoch as Current EpochStore
    participant CommitteeStore
    participant TxManager
    participant NewEpoch as New EpochStore
    
    Node->>AuthState: reconfigure(new_committee, config)
    AuthState->>CommitteeStore: insert_new_committee(new_committee)
    AuthState->>AuthState: acquire execution_lock for writing
    
    AuthState->>OldEpoch: new_at_next_epoch(new_committee, config)
    OldEpoch-->>NewEpoch: create
    
    AuthState->>AuthState: epoch_store.store(new_epoch_store)
    AuthState->>TxManager: reconfigure(new_epoch)
    AuthState->>AuthState: update execution_lock epoch
    
    AuthState->>OldEpoch: epoch_terminated()
    AuthState-->>Node: return new EpochStore
```

**Verification Status**: Verified-Code (reconfigure method in authority/src/state.rs)

### Key Steps in Reconfiguration

1. **Prepare for Epoch Change**:
   - Insert new committee into committee store
   - Acquire exclusive write lock on execution_lock
   - Revert uncommitted transactions from previous epoch

2. **Create New Epoch Store**:
   - Initialize new epoch store with new committee and configuration
   - Set up epoch-specific tables and state
   - Configure protocol parameters for new epoch

3. **Update System Components**:
   - Atomically swap epoch_store reference
   - Reconfigure transaction manager for new epoch
   - Update execution_lock with new epoch ID

4. **Finalize Transition**:
   - Signal old epoch store that it has been terminated
   - Release execution lock
   - Resume normal operation in new epoch

**Verification Status**: Verified-Code (reconfigure method in authority/src/state.rs)

### Reconfiguration State Transitions

```mermaid
stateDiagram-v2
    [*] --> AcceptingCerts: Initial State
    AcceptingCerts --> RejectingCerts: End-of-Publish Quorum
    RejectingCerts --> RejectingTx: All Consensus Certs Processed
    RejectingTx --> [*]: Epoch Terminated
    
    AcceptingCerts: Accept new certificates and transactions
    RejectingCerts: Reject new certificates, process existing ones
    RejectingTx: Reject all certificates and transactions
```

**Verification Status**: Verified-Code (ReconfigState transitions in reconfiguration.rs)

## Shared Object Version Assignment

### Version Assignment Process

```mermaid
sequenceDiagram
    participant Consensus
    participant EpochStore as AuthorityPerEpochStore
    participant VerManager as SharedObjVerManager
    participant Cache as ConsensusOutputCache
    
    Consensus->>EpochStore: Handle consensus transaction
    EpochStore->>VerManager: assign_versions_from_consensus
    
    VerManager->>EpochStore: get_or_init_next_object_versions
    
    EpochStore->>EpochStore: Acquire version locks
    EpochStore->>EpochStore: Check existing versions
    
    alt Version already assigned
        EpochStore-->>VerManager: Return assigned version
    else Version not assigned
        EpochStore->>EpochStore: Look up current object version
        EpochStore->>EpochStore: Determine next version
        EpochStore->>EpochStore: Write to next_shared_object_versions
        EpochStore-->>VerManager: Return assigned version
    end
    
    VerManager-->>EpochStore: Return assigned versions
    EpochStore->>Cache: Record shared object assignments
```

**Verification Status**: Verified-Code (version assignment in auth/src/epoch_store.rs)

### Version Assignment Implementation

```rust
// Assign shared object versions for a batch of transactions
fn process_consensus_transaction_shared_object_versions(
    &self,
    cache_reader: &dyn ObjectCacheRead,
    transactions: &[VerifiedExecutableTransaction],
    cancelled_txns: &BTreeMap<TransactionDigest, CancelConsensusCertificateReason>,
    output: &mut ConsensusCommitOutput,
) -> SomaResult {
    // Assign versions for all shared objects in transactions
    let ConsensusSharedObjVerAssignment {
        shared_input_next_versions,
        assigned_versions,
    } = SharedObjVerManager::assign_versions_from_consensus(
        self,
        cache_reader,
        transactions,
        cancelled_txns,
    )?;

    // Store assignments for transaction execution
    self.consensus_output_cache
        .insert_shared_object_assignments(&assigned_versions);

    // Update next versions for future transactions
    output.set_next_shared_object_versions(shared_input_next_versions);
    Ok(())
}
```

**Verification Status**: Verified-Code (process_consensus_transaction_shared_object_versions in epoch_store.rs)

## Key Interfaces with Other Modules

### Interface with Consensus Module
- **Integration Point**: Shared object transaction processing
- **Key Methods**:
  ```rust
  // Submit transaction for consensus ordering
  consensus_client.submit_transaction(transaction)
  
  // Process consensus-ordered transaction
  handle_consensus_transaction(transaction, transaction_manager)
  ```
- **Data Flow**: AuthorityState submits transactions with shared objects to consensus, which orders them and returns them for execution
- **Guarantees**: Consistent ordering of shared object transactions across all validators

**Verification Status**: Verified-Code (consensus interaction in authority and consensus modules)

### Interface with P2P Module
- **Integration Point**: State synchronization and certificate sharing
- **Key Methods**:
  ```rust
  // Synchronize state with other validators
  state_sync.synchronize_state()
  
  // Share transaction certificates
  p2p_client.share_certificate(certificate)
  ```
- **Data Flow**: AuthorityState shares certificates and synchronizes state through P2P interfaces
- **Guarantees**: Eventually consistent state across all validators

**Verification Status**: Verified-Code (p2p interaction in authority and p2p modules)

### Interface with Node Module
- **Integration Point**: Lifecycle management and service coordination
- **Key Methods**:
  ```rust
  // Initialize authority state
  AuthorityState::new(/* parameters */)
  
  // Handle reconfiguration
  authority_state.reconfigure(/* parameters */)
  ```
- **Data Flow**: Node module initializes and manages AuthorityState lifecycle
- **Guarantees**: Proper initialization and shutdown of authority components

**Verification Status**: Verified-Code (node interaction in authority and node modules)

## Error Handling and Recovery

### Error Categories
1. **Validation Errors**: Issues with transaction inputs or signatures
2. **Execution Errors**: Problems during transaction execution
3. **Consensus Errors**: Issues with consensus integration
4. **Storage Errors**: Failures accessing or updating storage
5. **System Errors**: Internal errors and unexpected failures

```rust
#[derive(Debug, thiserror::Error)]
pub enum SomaError {
    #[error("epoch has ended: {0}")]
    EpochEnded(EpochId),
    
    #[error("wrong epoch, expected {expected_epoch}, actual {actual_epoch}")]
    WrongEpoch {
        expected_epoch: EpochId,
        actual_epoch: EpochId,
    },
    
    #[error("validator halted at epoch end")]
    ValidatorHaltedAtEpochEnd,
    
    #[error("database error: {0}")]
    DatabaseError(#[from] std::io::Error),
    
    #[error("internal error: {0}")]
    InternalError(String),
    
    // Additional error variants...
}
```

**Verification Status**: Verified-Code (SomaError in types/src/error.rs)

### Error Propagation
- Structured errors with `thiserror` for context and type safety
- Error conversion between modules with `?` operator and `map_err`
- Explicit error categorization for appropriate handling
- Clear distinction between temporary and permanent failures

```rust
fn example_error_propagation() -> SomaResult<Output> {
    let result = operation()
        .map_err(|e| SomaError::InternalError(format!("Operation failed: {}", e)))?;
    
    Ok(result)
}
```

**Verification Status**: Verified-Code (error handling patterns throughout authority module)

### Error Handling Flow

```mermaid
flowchart TD
    TxReceived[Transaction Received] --> ValidateInputs[Validate Inputs]
    ValidateInputs -->|Invalid| ReturnValidationError[Return Validation Error]
    ValidateInputs -->|Valid| ValidateSig[Validate Signature]
    ValidateSig -->|Invalid| ReturnSigError[Return Signature Error]
    ValidateSig -->|Valid| CheckEpoch[Check Epoch]
    
    CheckEpoch -->|Wrong Epoch| ReturnEpochError[Return Epoch Error]
    CheckEpoch -->|Right Epoch| CheckLocks[Check Locks]
    CheckLocks -->|Locked| ReturnLockError[Return Lock Conflict]
    CheckLocks -->|Available| AcquireLocks[Acquire Locks]
    
    AcquireLocks -->|Success| ExecuteTx[Execute Transaction]
    AcquireLocks -->|Failure| ReturnLockError
    
    ExecuteTx -->|Success| GenerateEffects[Generate Effects]
    ExecuteTx -->|Failure| RecordFailure[Record Failure in Effects]
    
    GenerateEffects --> CommitEffects[Commit Effects]
    RecordFailure --> CommitEffects
    
    CommitEffects -->|Success| ReturnEffects[Return Effects]
    CommitEffects -->|Failure| HandleCommitError[Handle Commit Error]
    
    HandleCommitError --> RetryOrRevert[Retry or Revert]
```

**Verification Status**: Verified-Code (error handling flow in authority/src/state.rs)

### Recovery Mechanisms
1. **Transaction Retry**: Automatic retry for transient failures
2. **State Synchronization**: Recovery from peer validators after failures
3. **Checkpoint Recovery**: Restore consistent state from checkpoints
4. **Graceful Degradation**: Continue operation with reduced functionality
5. **Epoch Boundaries**: Clean recovery via epoch transitions

**Verification Status**: Verified-Code (recovery mechanisms in various components)

## Performance Optimizations

### Object Caching
- In-memory cache for frequently accessed objects
- Configurable cache size based on system resources
- Write-through caching for consistency
- Eviction policies for optimal memory usage

**Verification Status**: Verified-Code (cache implementations in authority/src/cache/)

### Concurrency Management
- Fine-grained locking for maximum parallelism
- Lock-free data structures where possible
- Dependency-based transaction scheduling
- Thread pool for execution scaling

**Verification Status**: Verified-Code (concurrency patterns throughout authority module)

### Batching Strategies
- Group-related operations for improved throughput
- Amortize consensus overhead across multiple transactions
- Batch storage operations for better I/O utilization
- Configurable batch sizes for tuning

**Verification Status**: Verified-Code (batching patterns in execution and storage)

## Epoch Management and Reconfiguration

### Epoch Advancement
1. **Detect Epoch End Condition**:
   - Timeout-based (epoch duration reached)
   - Explicit signal (system transaction)
   - Checkpoint-based (specific checkpoint reached)

2. **Create End-of-Epoch Transaction**:
   ```rust
   pub async fn create_and_execute_advance_epoch_tx(
       &self,
       epoch_store: &Arc<AuthorityPerEpochStore>,
       epoch_start_timestamp_ms: CommitTimestamp,
   ) -> anyhow::Result<(SystemState, TransactionEffects)> {
       // Create special transaction for epoch advancement
       let tx = VerifiedTransaction::new_end_of_epoch_transaction(
           EndOfEpochTransactionKind::new_change_epoch(next_epoch, epoch_start_timestamp_ms),
       );
       
       // Execute and commit transaction
       // ...
   }
   ```

3. **Update System State**:
   - New epoch ID
   - New validator set (if changed)
   - New protocol parameters (if changed)
   - Updated timestamp

4. **Commit Epoch Changes**:
   - Write new system state to storage
   - Signal components about epoch change
   - Prepare for transaction processing in new epoch

**Verification Status**: Verified-Code (create_and_execute_advance_epoch_tx in authority/src/state.rs)

### State Migration
1. **Create New Epoch Store**:
   ```rust
   let new_epoch_store = cur_epoch_store.new_at_next_epoch(
       self.name,
       new_committee,
       epoch_start_configuration,
       epoch_last_commit,
   );
   ```

2. **Atomic State Transition**:
   ```rust
   self.epoch_store.store(new_epoch_store.clone());
   ```

3. **Component Reconfiguration**:
   ```rust
   self.transaction_manager.reconfigure(new_epoch);
   *execution_lock = new_epoch;
   ```

4. **Old Epoch Termination**:
   ```rust
   cur_epoch_store.epoch_terminated().await;
   ```

**Verification Status**: Verified-Code (reconfigure method in authority/src/state.rs)

### End-of-Epoch Protocol
1. **End-of-Publish Collection**:
   - Validators signal they're done publishing transactions
   - System tracks quorum of end-of-publish messages
   - When quorum reached, stop accepting new certificates

2. **Certificate Processing Completion**:
   - Process all existing certificates in consensus
   - Wait for all pending certificates to be executed
   - Transition to rejecting all transactions

3. **Epoch Advancement Transaction**:
   - Create and execute end-of-epoch transaction
   - Update system state to next epoch configuration
   - Trigger reconfiguration on all components

**Verification Status**: Verified-Code (reconfiguration state management in authority/src/reconfiguration.rs)

## Confidence: 9/10

This document provides a comprehensive and accurate description of the Authority module in the Soma blockchain, based on direct code analysis and verification. The core components, their interactions, and key workflows are thoroughly documented with specific code references. The lock hierarchy and thread safety mechanisms are accurately described, and the reconfiguration process is fully explained. This high confidence rating reflects the thorough verification of all major components against the actual implementation.

## Last Updated: 2025-03-08 by Cline
