# Project Status and Milestones

## Purpose and Scope
This document tracks the overall progress of the Soma blockchain project, including completed milestones, current development status, and upcoming priorities. It provides a high-level view of project advancement and serves as a reference for planning and coordination.

## Now 
- Have all reconfiguration tests pass
	- Just incorporated shared objects, which should help the stress reconfiguration test pass

## Next
- Transaction types
	- Balances
		- Revisit tx conflicts (object locks) and ordering txs by dependencies
	- Staking
		- Separation between Encoder and Validator Stake
		- Leader Scoring / Tally Rule
	- Shard
		- Escrow
		- Result
- E2E integrate node and shard
	- Add encoder committee and probe hashes into EndOfEpochData
		- Make those agg sigs servable to the shard
	- Serve tx proofs to the shard

## April
- Fees / Tokenomics
	- Gas (tax)
	- Network Rewards
- Probe + Learnability

## May
- RocksDB for long term storage
- Epoch upgradability for:
	- Probe architecture
	- Embedding dimensionality
- ChainIdentifier (to distinguish between testnets and mainnet)
- RPC
- Metrics

## June
- Archival Store (of Commits and Blocks)
- Indexer
- Final testing and refactoring

## Other
- Integrate mempool into Mysticeti
- Transaction Types
	- Payment channels (for data market and bridging)
- State Snapshotting

## Blockers and Challenges
This section will be updated as blockers and challenges are identified.

## Documentation Progress

### Core Documentation: 90% Complete
- **Strengths**: Comprehensive architecture overview, pattern documentation
- **Recent Updates**: Enhanced system patterns documentation, improved technical context

### Module Documentation: 90% Complete
- **Strengths**: Detailed component explanations, workflow documentation
- **Recent Updates**: Added consensus workflow documentation, improved authority module docs

### Knowledge Base: 95% Complete
- **Strengths**: Cross-cutting concerns well documented, concurrency patterns explained
- **Recent Updates**: Added security model, data flow documentation

## Confidence: 8/10
This progress assessment is based on thorough codebase review and comprehensive documentation verification. The percentages represent a careful estimate of completion, considering both implemented features and their stability. Recent documentation improvements have significantly increased our confidence in system understanding.

## Last Updated: 2025-03-10 by Cline
