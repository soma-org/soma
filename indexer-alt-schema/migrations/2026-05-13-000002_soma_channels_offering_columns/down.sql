-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

DROP INDEX IF EXISTS idx_soma_channels_model_status_lastcp;
ALTER TABLE soma_channels
    DROP COLUMN ttot_bound_ms,
    DROP COLUMN ttft_bound_ms,
    DROP COLUMN request_micros,
    DROP COLUMN cache_write_micros_per_1k,
    DROP COLUMN cache_read_micros_per_1k,
    DROP COLUMN completion_micros_per_1k,
    DROP COLUMN prompt_micros_per_1k,
    DROP COLUMN model_id;
