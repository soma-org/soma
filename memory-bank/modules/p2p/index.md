# P2P Module

## Purpose and Scope
This document provides a comprehensive overview of the P2P (Peer-to-Peer) module in the Soma blockchain. The P2P module is responsible for network discovery, state synchronization, and facilitating communication between nodes in the network. This document explains its architecture, components, and how they work together to enable reliable peer-to-peer communication.

## Key Components

### P2P Server
- Core network service implementation
- Handles gRPC protocol implementation
- Processes incoming and outgoing messages
- Manages state synchronization requests
- Routes network requests to appropriate handlers
- Enforces fetch size limits and resource constraints

### Discovery Event Loop
- Implements peer discovery protocol
- Manages node information exchange
- Maintains peer metadata and capabilities
- Handles peer ranking and selection
- Establishes connections with new peers
- Validates node information signatures

### State Sync Event Loop
- Implements state synchronization protocol
- Manages blocks and commits synchronization
- Handles checkpoint-based recovery
- Verifies fetched blocks and commits
- Coordinates incremental state updates
- Broadcasts CommittedSubDag objects to execute transactions

### P2P Builder
- Constructs and configures P2P components
- Initializes discovery and sync subsystems
- Sets up network interfaces and handles
- Provides service access points for other modules
- Creates broadcast channel for CommittedSubDag objects

### Transaction Verifier
- Verifies transactions during sync
- Validates certificates against committee
- Ensures transaction integrity during sync
- Maintains consistent verification standards
- Works with block verifier to ensure data consistency

## Module Structure

This documentation is organized into several subdocuments, each focusing on a specific aspect of the P2P module:

1. [Module Structure](./module_structure.md) - Detailed component architecture and relationships
2. [Discovery](./discovery.md) - Peer discovery and network connectivity
3. [State Sync](./state_sync.md) - State synchronization mechanisms
4. [Thread Safety](./thread_safety.md) - Concurrency controls and safety guarantees

## Primary Interfaces

### With Authority Module
- Provides object and transaction synchronization
- Requests missing objects during validation
- Verifies incoming transaction certificates
- Submits synchronized transactions to authority
- Broadcasts CommittedSubDag objects to CommitExecutor for execution
- Ensures synchronized transactions are executed correctly

### With Consensus Module
- Propagates consensus messages
- Distributes committed blocks
- Shares certificates and votes
- Synchronizes consensus state
- Receives newly verified commits

### With Node Module
- Lifecycle management and graceful startup/shutdown
- Configuration coordination
- Network service registration
- Component supervision

## Transaction Execution Flow

The P2P module plays a crucial role in the transaction execution pipeline:

1. State sync system fetches blocks and commits from peers
2. Verification ensures data integrity and correctness
3. CommittedSubDag objects are created representing complete commits
4. These objects are broadcast via a channel to the CommitExecutor
5. CommitExecutor in the authority module processes these objects to:
   - Extract and filter transactions (remove already executed ones)
   - Assign shared object versions
   - Enqueue transactions for execution
   - Process transaction effects
   - Commit outputs to storage
6. This flow ensures that all synchronized transactions are properly executed
7. Special handling is provided for epoch boundaries

## Network Protocol
The P2P module implements a custom network protocol over gRPC with the following key message types:
- Peer discovery messages (GetKnownPeers)
- Block fetch messages (FetchBlocks)
- Commit fetch messages (FetchCommits)
- Commit propagation messages (PushCommit)
- Commit availability queries (GetCommitAvailability)

## Error Handling
The module implements comprehensive error handling for:
- Network failures and timeouts
- Data validation failures
- Peer disconnections
- Resource limitations
- Protocol violations
- Epoch boundary conditions

## Confidence: 9/10
This overview document provides a comprehensive high-level understanding of the P2P module's structure and responsibilities. The detailed implementation specifics are covered in the referenced subdocuments.

## Last Updated: 2025-03-08 by Cline
