# Soma Blockchain Project Brief

## Project Purpose
Soma is a Rust-based blockchain node software that provides a Byzantine Fault Tolerant consensus mechanism with validator-based authority. The system processes transactions and maintains state across a distributed network, with clear epoch boundaries for reconfiguration.


## Core Requirements
- Process data transactions from users
- Distribute transactions to encoder shards
- Enable encoders to embed data and self-evaluate their embeddings
- Assess data usefulness and manage rewards
- Register new encoders and manage committee membership
- Handle escrow payments and reward distributions based on stake
- Serve proofs of validator/encoder committees at epoch transitions
- Use Mysticeti for consensus and state transitions on object storage


## Technical Architecture

### Component Architecture
The system is organized into primary modules:
- **Authority**: State management, transaction validation and execution
- **Consensus**: BFT agreement protocol implementation (Mysticeti)
- **Node**: Lifecycle and orchestration
- **P2P**: Network discovery and state synchronization

### Key Components
1. **AuthorityState**: Manages transaction processing and execution
2. **AuthorityPerEpochStore**: Holds epoch-specific data and state
3. **ConsensusAuthority**: Handles Byzantine fault-tolerant agreement
4. **SomaNode**: Coordinates lifecycle and component interaction
5. **StateSyncEventLoop**: Synchronizes state between nodes
6. **DiscoveryEventLoop**: Manages peer discovery and networking

### Transaction Flow
1. Transaction received by validator node
2. Transaction validated and signed by validator
3. Transaction certificate created with validator signatures
4. Certificate processed through consensus
5. Consensus orders transactions and resolves conflicts
6. Transactions executed and effects committed to state
7. State changes propagated through the network

### Epoch Management
- Epochs define validator set lifetimes
- End-of-epoch determined by time threshold or explicit signals
- Validator set changes implemented at epoch boundaries
- System state accumulated and verified at epoch transitions

## Development Milestones
Based on the Soma Milestones document:
- State sync and shard end-to-end workflow
- Transaction types implementation (balances, staking, shards)
- Node and shard integration
- Fee structure and tokenomics
- Storage, epoch upgradability, and RPC implementation
- Archival store and indexer completion

## Development Approach
This project utilizes an AI agent workflow with Cursor and Cline, focusing on:
- Context efficiency through structured documentation
- Modular development with clear handoffs
- Rust async best practices and thorough testing
- Thorough testing at unit and integration levels
- Comprehensive error handling and recovery
- Component-based architecture with clear interfaces
- Progressive stability improvements


## Project Philosophy
Soma transforms the growth of knowledge into a natural process by:
- Creating an incentive for knowledge contributions
- Evaluating contributions objectively through probe models
- Rewarding the most valuable additions to our collective understanding
- Building a self-improving world model without central control

## Implementation Philosophy
- Safety-first approach with Byzantine fault tolerance
- Deterministic execution for predictable results
- Performance optimization without compromising safety
- Clean recovery from crashes and network partitions

