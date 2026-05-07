-- Per-tx materialization of `BridgeDeposit` transactions. One row per
-- BridgeDeposit, populated by the `soma_bridge_deposits` indexer handler
-- from each tx's `TransactionKind::BridgeDeposit(BridgeDepositArgs)`.
--
-- `eth_tx_hash` is the Base-side L1 transaction proof carried in the args.
-- It enables off-chain joins back to Base (for e.g. funder-clustering and
-- audit), which is *not* possible from `soma_balance_deltas` alone (per-cp
-- aggregation loses per-tx provenance and drops the eth_tx_hash entirely).
CREATE TABLE soma_bridge_deposits (
    tx_sequence_number BIGINT NOT NULL PRIMARY KEY,
    cp_sequence_number BIGINT NOT NULL,
    -- Soma recipient that received the minted USDC.
    recipient          BYTEA  NOT NULL,
    -- USDC amount minted (micros — same scale as soma_balance_deltas.delta).
    amount             BIGINT NOT NULL,
    -- L1 nonce assigned by the bridge contract on Base.
    nonce              BIGINT NOT NULL,
    -- Base-side tx hash. Always 32 bytes.
    eth_tx_hash        BYTEA  NOT NULL,
    timestamp_ms       BIGINT NOT NULL
);

-- Recipient enumeration: "what bridge deposits did this address receive?"
CREATE INDEX soma_bridge_deposits_recipient
    ON soma_bridge_deposits (recipient, timestamp_ms);

-- Time-window scans (e.g. clustering deposits within a 5-min window).
CREATE INDEX soma_bridge_deposits_ts
    ON soma_bridge_deposits (timestamp_ms);

-- L1 lookup: "which Soma address(es) did this Base tx fund?"
CREATE INDEX soma_bridge_deposits_eth_hash
    ON soma_bridge_deposits (eth_tx_hash);
