-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

ALTER TABLE soma_channel_events
    DROP COLUMN rating_reason_code,
    DROP COLUMN requests_delta,
    DROP COLUMN cache_write_delta,
    DROP COLUMN cache_read_delta,
    DROP COLUMN tokens_out_delta,
    DROP COLUMN tokens_in_delta;
