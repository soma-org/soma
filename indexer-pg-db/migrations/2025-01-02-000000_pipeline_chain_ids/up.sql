-- Per-pipeline chain id, used to detect cross-network misconfiguration.
--
-- The framework calls `Connection::accepts_chain_id` once per pipeline at startup. The first
-- call for a `pipeline` records the chain id; subsequent calls compare against it. A mismatch
-- means the indexer has been pointed at data from a different network than it previously
-- processed, and processing must halt rather than silently mix networks.
--
-- `chain_id` is the 32-byte genesis checkpoint digest of the network, stored as raw bytes.
CREATE TABLE IF NOT EXISTS pipeline_chain_ids
(
    pipeline   TEXT        PRIMARY KEY,
    chain_id   BYTEA       NOT NULL
);
