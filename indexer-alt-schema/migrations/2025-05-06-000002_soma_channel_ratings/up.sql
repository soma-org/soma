-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Channel ratings: payer-issued binary (negative / positive) flags
-- per channel. Latest-wins: a subsequent `RateChannel` from the
-- same payer for the same channel UPSERTs the row with the new
-- flag.
--
-- Powers the volume-weighted `negative_rate_30d` signal in the
-- `provider_reputation` view (see migration 037 update). The
-- `payee` column is denormalized from `soma_channels` for cheap
-- per-provider aggregation.
CREATE TABLE soma_channel_ratings (
    channel_id      BYTEA   NOT NULL PRIMARY KEY,
    payer           BYTEA   NOT NULL,
    payee           BYTEA   NOT NULL,
    -- TRUE = negative rating (thumbs down), FALSE = positive.
    negative        BOOLEAN NOT NULL,
    rated_at_cp     BIGINT  NOT NULL,
    rated_at_ms     BIGINT  NOT NULL
);

-- Per-payee aggregation index for the view's join on
-- `soma_channel_ratings` keyed by provider address.
CREATE INDEX soma_channel_ratings_payee
    ON soma_channel_ratings (payee, rated_at_ms DESC);
