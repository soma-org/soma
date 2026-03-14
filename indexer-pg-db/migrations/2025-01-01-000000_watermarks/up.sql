-- Watermarks table for tracking indexer pipeline progress.
--
-- `pipeline` identifies the specific indexer pipeline (e.g., "epochs", "checkpoints").
--
-- The following columns track the upper bounds of data that has been committed:
--
-- - `epoch_hi_inclusive` is the epoch of the latest committed checkpoint.
-- - `checkpoint_hi_inclusive` is the latest committed checkpoint sequence number.
-- - `tx_hi` is the exclusive upper bound on the transaction sequence number of the latest
--   committed checkpoint.
-- - `timestamp_ms_hi_inclusive` is the timestamp of the latest committed checkpoint, which
--   guarantees that all checkpoints up to and including `checkpoint_hi_inclusive` have been
--   persisted.
--
-- `reader_lo` is the inclusive low watermark that the pruner advances. Readers consider data below
-- this watermark as removed.
--
-- `pruner_timestamp` records when `reader_lo` was last updated. This allows the pruner to wait
-- for in-flight reads to complete before actually deleting data.
--
-- `pruner_hi` tracks the exclusive upper bound of data that has been pruned. Data below this
-- watermark may already be deleted.
CREATE TABLE IF NOT EXISTS watermarks
(
    pipeline                  TEXT        PRIMARY KEY,
    epoch_hi_inclusive        BIGINT      NOT NULL,
    checkpoint_hi_inclusive   BIGINT      NOT NULL,
    tx_hi                     BIGINT      NOT NULL,
    timestamp_ms_hi_inclusive BIGINT      NOT NULL,
    reader_lo                 BIGINT      NOT NULL,
    pruner_timestamp          TIMESTAMP   NOT NULL DEFAULT NOW(),
    pruner_hi                 BIGINT      NOT NULL
);
