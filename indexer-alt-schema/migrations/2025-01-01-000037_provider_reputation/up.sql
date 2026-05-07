-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Reputation aggregates per provider, derived from `soma_channels`,
-- `soma_channel_events`, and `soma_channel_ratings`. A SQL view (not
-- a materialized table) — iterating reputation signal definitions
-- is then a schema migration only, no indexer rewire. Promote to a
-- periodic refresh table later if query latency becomes a problem.
--
-- Signals (versioned off-chain — bump `signal_version` if a formula
-- changes):
--   * volume_settled_30d   — sum of Settle deltas, last 30 days by
--                            checkpoint timestamp.
--   * distinct_buyers_30d  — count distinct payers, same window.
--   * channel_renewal_rate — TopUp count / OpenChannel count over all
--                            time. >0 means buyers chose to keep funding
--                            rather than walk away.
--   * mean_channel_span_cps — avg ((last_update_cp - opened_at_cp))
--                            per channel. Approximate.
--   * negative_rate_30d    — VOLUME-weighted fraction of rated channels
--                            this provider received that buyers flagged
--                            as negative, scoped to ratings issued in
--                            the last 30 days. Higher = worse. NULL
--                            when no in-window ratings on positive-
--                            volume channels (consumers should treat
--                            as "no signal", not as 0).
--   * rating_count_30d     — number of distinct channels rated in the
--                            window. Confidence proxy alongside
--                            `negative_rate_30d`.
--
-- All signals are observation-only: nothing on-chain, nothing
-- slashable. Buyers consume them as one routing input among many.
CREATE VIEW provider_reputation AS
WITH
  recent_settles AS (
    SELECT c.payee AS provider, e.delta, e.timestamp_ms, c.payer
    FROM soma_channel_events e
    JOIN soma_channels c ON c.channel_id = e.channel_id
    WHERE e.kind = 'settle'
      AND e.timestamp_ms >= (extract(epoch from now()) * 1000)::BIGINT - 30 * 24 * 3600 * 1000
  ),
  per_channel_age AS (
    SELECT payee AS provider,
           (last_update_cp - opened_at_cp) AS cp_span
    FROM soma_channels
    WHERE status <> 2
  ),
  open_count AS (
    SELECT c.payee AS provider, COUNT(*) AS n
    FROM soma_channel_events e
    JOIN soma_channels c ON c.channel_id = e.channel_id
    WHERE e.kind = 'open'
    GROUP BY c.payee
  ),
  topup_count AS (
    SELECT c.payee AS provider, COUNT(*) AS n
    FROM soma_channel_events e
    JOIN soma_channels c ON c.channel_id = e.channel_id
    WHERE e.kind = 'top_up'
    GROUP BY c.payee
  ),
  -- Volume-weighted ratings: each rating is paired with the
  -- channel's total settled volume, scoped to the last 30 days by
  -- `rated_at_ms`. We carry the `negative` bool through; the SELECT
  -- below takes the SUM of volume per (provider, negative).
  recent_rated_volume AS (
    SELECT
      r.payee AS provider,
      r.negative AS negative,
      c.settled_amount AS volume
    FROM soma_channel_ratings r
    JOIN soma_channels c ON c.channel_id = r.channel_id
    WHERE r.rated_at_ms >= (extract(epoch from now()) * 1000)::BIGINT - 30 * 24 * 3600 * 1000
      AND c.settled_amount > 0
  )
SELECT
  p.address AS address,
  COALESCE((SELECT SUM(delta) FROM recent_settles WHERE provider = p.address), 0)::BIGINT
    AS volume_settled_30d,
  COALESCE((SELECT COUNT(DISTINCT payer) FROM recent_settles WHERE provider = p.address), 0)::BIGINT
    AS distinct_buyers_30d,
  CASE
    WHEN COALESCE((SELECT n FROM open_count WHERE provider = p.address), 0) = 0 THEN 0::DOUBLE PRECISION
    ELSE COALESCE((SELECT n FROM topup_count WHERE provider = p.address), 0)::DOUBLE PRECISION
       / (SELECT n FROM open_count WHERE provider = p.address)::DOUBLE PRECISION
  END AS channel_renewal_rate,
  COALESCE((SELECT AVG(cp_span) FROM per_channel_age WHERE provider = p.address), 0)::BIGINT
    AS mean_channel_span_cps,
  -- Volume-weighted negative-rating rate in the 30d window.
  -- numerator: settled volume on negatively-rated channels.
  -- denominator: settled volume on all rated channels.
  -- NULL when denominator is 0 (no in-window ratings on positive
  -- volume).
  (SELECT
     CASE WHEN SUM(volume) > 0
          THEN SUM(volume) FILTER (WHERE negative)::DOUBLE PRECISION
               / SUM(volume)::DOUBLE PRECISION
          ELSE NULL
     END
   FROM recent_rated_volume WHERE provider = p.address)
    AS negative_rate_30d,
  COALESCE((SELECT COUNT(*) FROM recent_rated_volume WHERE provider = p.address), 0)::BIGINT
    AS rating_count_30d,
  -- Signal definition version. Bump on any formula change so
  -- consumers can detect the change without reading code.
  2::INT AS signal_version
FROM soma_providers p;
