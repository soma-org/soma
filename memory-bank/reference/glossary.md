# Terminology Glossary

## Purpose and Scope
This glossary provides definitions for key terms and concepts used in the Soma blockchain documentation. It serves as a reference resource to ensure consistent understanding and usage of terminology across the project.

## Core Blockchain Concepts

### Authority
A validator node with the authority to validate transactions and participate in consensus.

### AuthorityName
The unique identifier for a validator authority, typically a public key.

### Byzantine Fault Tolerance (BFT)
A property of a system that can continue to operate correctly even if some nodes fail or behave maliciously, up to a certain threshold (typically less than 1/3 of nodes).

### Certificate
A collection of signatures from a quorum of validators attesting to the validity of a transaction.

### Committee
The set of validators participating in consensus during a specific epoch, along with their voting weights.

### Consensus
The process by which validators agree on the ordering and validity of transactions.

### Epoch
A period during which the validator set (committee) remains unchanged. Epoch changes may involve reconfiguration of the validator set.

### Fullnode
A node that maintains a copy of the blockchain state but does not participate in consensus or sign transactions.

### Object
The fundamental unit of state in the Soma blockchain, with a unique identifier, version, owner, and content.

### Owner
The entity that has control over an object, which can be an address, shared access, or immutable status.

### Sequence Number
A monotonically increasing number used to track the sequence of operations, particularly for object versions.

### Transaction
An operation that modifies the blockchain state, including inputs, operations, and signatures.

### Transaction Effects
The changes to blockchain state resulting from a transaction, including created, modified, and deleted objects.

### Validator
A node that participates in consensus, validates transactions, and maintains the blockchain state.

## Soma-Specific Concepts

### Data Contributor
An entity that submits knowledge or data to the Soma network for evaluation and potential reward.

### Encoder
A specialized node responsible for evaluating and embedding contributed data.

### Encoder Shard
A subset of encoders responsible for a specific domain or type of data evaluation.

### Probe Model
A model used to evaluate the usefulness and novelty of contributed data.

### Usefulness Score
A metric indicating the value of a piece of data to the collective knowledge system.

## Technical Terms

### ArcSwap
A concurrency primitive for atomically replacing shared components, particularly used for epoch transitions.

### BCS (Binary Canonical Serialization)
The serialization format used in Soma for deterministic representation of data structures.

### Commit
The process of finalizing a transaction by applying its effects to the persistent state.

### Execution Status
The outcome of transaction execution, which can be success or failure with specific error details.

### JoinSet
A Tokio primitive for managing a set of tasks and tracking their completion.

### Merkle Tree
A tree data structure where each non-leaf node is a hash of its child nodes, used for efficient proof of membership.

### Object Lock
A concurrency mechanism to prevent conflicts when multiple transactions attempt to access the same object.

### RwLock
A reader-writer lock that allows multiple readers or a single writer to access a resource concurrently.

### TemporaryStore
An in-memory representation of object state used during transaction execution.

### Threshold Signature
A cryptographic signature scheme requiring a minimum number of parties to collaborate to produce a valid signature.

### Transaction Digest
A cryptographic hash uniquely identifying a transaction.

## Networking Terms

### Discovery Protocol
The mechanism by which nodes find and connect to peers in the network.

### Peer
Another node in the P2P network with which a node communicates.

### State Synchronization
The process by which nodes update their state to match the current blockchain state.

### View Change
A consensus protocol mechanism to handle leader failures by switching to a new leader.

## Governance Terms

### Stake
The amount of value locked by a validator as a security deposit, which determines their voting power.

### Slashing
The penalty mechanism for misbehaving validators, typically involving the loss of staked tokens.

### Voting Power
The influence a validator has in the consensus process, typically proportional to their stake.

## File Naming Conventions

### authority.rs
A file implementing the authority-related functionality.

### state.rs
A file implementing state management for a component.

### tx_manager.rs
A file implementing transaction management and scheduling.

## Confidence: 6/10
This glossary provides a good foundation of terminology but will need to be expanded as more implementation details and domain-specific terms are defined.

## Last Updated: 2025-03-08 by Cline
