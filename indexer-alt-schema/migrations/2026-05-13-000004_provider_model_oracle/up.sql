-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Per-(provider, model_id) reputation + price-oracle views. Composed
-- entirely from on-chain-mirrored tables (`soma_offerings`,
-- `soma_channels`, `soma_channel_events`, `soma_channel_ratings`) so
-- the view definition itself is the entire formula — bump `signal_version`
-- to indicate a definition change.

-- ---------------------------------------------------------------------------
-- `provider_model_offerings_ranked` — the *menu* surface. One row per
-- active offering, sorted by `effective_per_1k = prompt + completion`.
-- Discovery layer for the inference proxy router.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW provider_model_offerings_ranked AS
SELECT
    o.model_id,
    o.provider,
    p.endpoint,
    o.prompt_micros_per_1k,
    o.completion_micros_per_1k,
    o.cache_read_micros_per_1k,
    o.cache_write_micros_per_1k,
    o.request_micros,
    o.ttft_bound_ms,
    o.ttot_bound_ms,
    o.updated_at_ms,
    (o.prompt_micros_per_1k + o.completion_micros_per_1k) AS effective_per_1k
FROM soma_offerings o
LEFT JOIN soma_providers p ON p.address = o.provider
WHERE o.active = TRUE;

-- ---------------------------------------------------------------------------
-- `model_price_oracle` — the *ground-truth* surface. Stats from
-- channels that have actually transacted, not just providers'
-- aspirational menus. Power-tools for buyers and downstream
-- financial products that want to know real prices, not posted ones.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW model_price_oracle AS
SELECT
    model_id,
    COUNT(*)                                       AS open_channels,
    SUM(deposit)                                   AS total_tvl_micros,
    MIN(prompt_micros_per_1k)                      AS prompt_min,
    MAX(prompt_micros_per_1k)                      AS prompt_max,
    -- TVL-weighted average $-rate. Treat unfunded zero-tvl
    -- channels as zero-weight (filter to deposit > 0).
    CASE WHEN SUM(deposit) > 0 THEN
        SUM(prompt_micros_per_1k * deposit) / SUM(deposit)
    ELSE NULL END                                  AS prompt_vwap,
    CASE WHEN SUM(deposit) > 0 THEN
        SUM(completion_micros_per_1k * deposit) / SUM(deposit)
    ELSE NULL END                                  AS completion_vwap,
    MIN(ttft_bound_ms)                             AS ttft_bound_min_ms,
    MAX(ttft_bound_ms)                             AS ttft_bound_max_ms
FROM soma_channels
WHERE model_id <> '' AND status = 0
GROUP BY model_id;

-- ---------------------------------------------------------------------------
-- `provider_model_reputation` — per-(provider, model) extension of
-- the existing `provider_reputation` view. Rolls up the
-- offering-snapshotted channels + settle events for one model and
-- one provider into a single row.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW provider_model_reputation AS
WITH thirty_day_floor AS (
    SELECT (EXTRACT(EPOCH FROM now()) * 1000)::BIGINT - 30::BIGINT * 86400000 AS floor_ms
),
settles AS (
    SELECT
        c.payee     AS provider,
        c.model_id  AS model_id,
        e.delta,
        e.timestamp_ms
    FROM soma_channel_events e
    JOIN soma_channels c ON c.channel_id = e.channel_id
    CROSS JOIN thirty_day_floor f
    WHERE e.kind = 'settle' AND e.timestamp_ms >= f.floor_ms
)
SELECT
    c.payee    AS provider,
    c.model_id AS model_id,
    -- 30d settled volume on this (provider, model) channel set.
    COALESCE((SELECT SUM(delta) FROM settles s
              WHERE s.provider = c.payee AND s.model_id = c.model_id), 0)  AS volume_30d,
    -- 30d distinct payers.
    COALESCE((SELECT COUNT(DISTINCT c2.payer)
              FROM soma_channels c2
              CROSS JOIN thirty_day_floor f
              WHERE c2.payee = c.payee
                AND c2.model_id = c.model_id
                AND c2.last_update_cp IS NOT NULL
                AND c2.opened_at_cp IS NOT NULL), 0)                       AS distinct_buyers_30d,
    -- 30d negative-rate breakdown by reason_code.
    COALESCE((SELECT COUNT(*) FROM soma_channel_ratings r
              JOIN soma_channels rc ON rc.channel_id = r.channel_id
              WHERE r.payee = c.payee
                AND rc.model_id = c.model_id
                AND r.negative = TRUE), 0)                                 AS negative_rating_count_30d
FROM soma_channels c
WHERE c.model_id <> ''
GROUP BY c.payee, c.model_id;
