-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Reputation aggregates per provider, derived from `soma_channels`
-- and `soma_channel_events`. A SQL view (not a materialized table) —
-- iterating reputation signal definitions is then a schema migration
-- only, no indexer rewire. Promote to a periodic refresh table later
-- if query latency becomes a problem.
--
-- Signals (versioned off-chain — bump `signal_version` if a formula
-- changes):
--   * volume_settled_30d  — sum of Settle deltas, last 30 days by
--                           checkpoint timestamp.
--   * distinct_buyers_30d — count distinct payers, same window.
--   * channel_renewal_rate — TopUp count / OpenChannel count over all
--                           time. >0 means buyers chose to keep funding
--                           rather than walk away.
--   * mean_channel_age_ms — avg ((last_update_cp - opened_at_cp) * cp_ms)
--                           per channel, weighted equally. Approximate —
--                           we use last-update as a proxy for "live until
--                           when". Good enough for a weak signal.
--
-- All three are observation-only: nothing on-chain, nothing slashable.
-- Buyers consume them as one routing input among many.
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
  -- Signal definition version. Bump on any formula change so
  -- consumers can detect the change without reading code.
  1::INT AS signal_version
FROM soma_providers p;
