-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Append-only event log for payment-channel ops. One row per channel
-- transaction. Powers reputation aggregates (volume_settled_30d,
-- distinct_buyers_30d, channel_renewal_rate, mean_channel_age) and
-- the "what happened to this channel" tail for clients reconstructing
-- their state after losing local files.
--
-- Ordering: by `(cp_sequence_number, tx_sequence_number)`. Settle
-- events carry the per-tx delta (settled_amount delta); TopUp carries
-- the deposit increase; OpenChannel carries the initial deposit;
-- everything else has delta=0.
CREATE TABLE soma_channel_events (
    tx_sequence_number BIGINT     NOT NULL PRIMARY KEY,
    cp_sequence_number BIGINT     NOT NULL,
    channel_id         BYTEA      NOT NULL,
    -- 'open' | 'settle' | 'top_up' | 'request_close' | 'withdraw'.
    -- Versioned at the indexer; chain-side label changes don't
    -- require a schema migration.
    kind               TEXT       NOT NULL,
    -- Signed by convention: positive amounts add to the channel
    -- (open / top_up) or to the payee (settle); zero for control ops
    -- (request_close / withdraw). BIGINT holds far more than any
    -- realistic delta (2^63 ≈ 9.2×10^18 USDC micros).
    delta              BIGINT     NOT NULL,
    timestamp_ms       BIGINT     NOT NULL
);

-- Per-channel scan: "show me everything that happened on this channel".
CREATE INDEX soma_channel_events_channel_cp
    ON soma_channel_events (channel_id, cp_sequence_number DESC);

-- Recency-bounded reputation queries (e.g. "last 30 days of Settles
-- for payee X").
CREATE INDEX soma_channel_events_kind_cp
    ON soma_channel_events (kind, cp_sequence_number DESC);
