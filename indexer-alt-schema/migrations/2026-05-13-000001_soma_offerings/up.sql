-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Per-(provider, model_id) Offering rows mirroring the on-chain
-- `Offering` shared objects. Created/updated/deactivated by the
-- corresponding tx kinds; the row is the price + SLA menu a provider
-- publishes for a single model. Channels snapshot these values onto
-- their own row at OpenChannel time (see migration 000002) so that
-- mutating an offering does not retroactively change settlement math
-- for existing channels.
CREATE TABLE soma_offerings (
    provider                   BYTEA       NOT NULL,
    model_id                   TEXT        NOT NULL,
    prompt_micros_per_1k       BIGINT      NOT NULL,
    completion_micros_per_1k   BIGINT      NOT NULL,
    cache_read_micros_per_1k   BIGINT      NOT NULL,
    cache_write_micros_per_1k  BIGINT      NOT NULL,
    request_micros             BIGINT      NOT NULL,
    ttft_bound_ms              INTEGER     NOT NULL,
    ttot_bound_ms              INTEGER     NOT NULL,
    active                     BOOLEAN     NOT NULL,
    updated_at_cp              BIGINT      NOT NULL,
    updated_at_ms              BIGINT      NOT NULL,
    PRIMARY KEY (provider, model_id)
);

-- Discovery: cheapest active offerings per model. The router uses
-- this for routing decisions (e.g. ordered by `prompt_micros_per_1k`).
CREATE INDEX idx_soma_offerings_model_active_price
    ON soma_offerings (model_id, active, prompt_micros_per_1k);

-- Per-provider lookup: a provider's full menu in one shot.
CREATE INDEX idx_soma_offerings_provider_active
    ON soma_offerings (provider, active);
