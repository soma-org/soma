-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Per-(payer, model) channel snapshot columns. The on-chain
-- `ChannelV1` carries these directly; the indexer mirrors them so
-- queries that ask "what model is this channel for? what prices did
-- it open with?" answer in a single row read.
--
-- All channel rows opened pre-migration get sentinel values
-- (`model_id = ''`, prices = 0) because the on-chain schema didn't
-- include these fields before. The router treats empty `model_id` as
-- "legacy/unbound" and refuses to route requests through such a
-- channel.

ALTER TABLE soma_channels
    ADD COLUMN model_id                   TEXT      NOT NULL DEFAULT '',
    ADD COLUMN prompt_micros_per_1k       BIGINT    NOT NULL DEFAULT 0,
    ADD COLUMN completion_micros_per_1k   BIGINT    NOT NULL DEFAULT 0,
    ADD COLUMN cache_read_micros_per_1k   BIGINT    NOT NULL DEFAULT 0,
    ADD COLUMN cache_write_micros_per_1k  BIGINT    NOT NULL DEFAULT 0,
    ADD COLUMN request_micros             BIGINT    NOT NULL DEFAULT 0,
    ADD COLUMN ttft_bound_ms              INTEGER   NOT NULL DEFAULT 0,
    ADD COLUMN ttot_bound_ms              INTEGER   NOT NULL DEFAULT 0;

-- Discovery / oracle: scan all open channels for a given model.
CREATE INDEX idx_soma_channels_model_status_lastcp
    ON soma_channels (model_id, status, last_update_cp DESC);
