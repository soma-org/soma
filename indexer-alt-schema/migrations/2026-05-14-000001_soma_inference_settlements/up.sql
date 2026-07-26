-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Denormalized per-Settle row. Joins the Settle event with the
-- channel's snapshotted `model_id` + `payee` so callers don't have to
-- reconstruct that via `soma_channel_events JOIN soma_channels`.
--
-- Use cases:
--   * compute marketplace-wide token volume per (provider, model_id)
--     per epoch / per day from a single table scan
--   * power buyer-side dashboards that show "I spent X on model Y"
--   * feed the `model_price_oracle` view's realized-price stream
--     directly (one row = one realized price-and-volume sample)
--
-- One row per `Settle` tx. The handler reads the channel object out
-- of the tx's input set (its post-state if mutated, pre-state if
-- only read) to get model_id / payee — those don't change after open.
CREATE TABLE soma_inference_settlements (
    tx_sequence_number  BIGINT      NOT NULL PRIMARY KEY,
    cp_sequence_number  BIGINT      NOT NULL,
    channel_id          BYTEA       NOT NULL,
    payer               BYTEA       NOT NULL,
    payee               BYTEA       NOT NULL,
    model_id            TEXT        NOT NULL,
    -- Cumulative voucher fields (post-Settle values; the chain enforces
    -- monotonic-non-decreasing across Settles on the same channel).
    cumulative_amount         BIGINT  NOT NULL,
    cumulative_prompt_tokens  BIGINT  NOT NULL,
    cumulative_completion_tokens BIGINT NOT NULL,
    cumulative_cache_read_tokens BIGINT NOT NULL,
    cumulative_cache_write_tokens BIGINT NOT NULL,
    cumulative_requests       BIGINT  NOT NULL,
    -- Per-tx deltas computed from the pre-state Channel input.
    delta_amount              BIGINT  NOT NULL,
    timestamp_ms              BIGINT  NOT NULL
);

CREATE INDEX idx_soma_inference_settlements_model_ts
    ON soma_inference_settlements (model_id, timestamp_ms);

CREATE INDEX idx_soma_inference_settlements_payee_ts
    ON soma_inference_settlements (payee, timestamp_ms);

CREATE INDEX idx_soma_inference_settlements_payer_ts
    ON soma_inference_settlements (payer, timestamp_ms);
