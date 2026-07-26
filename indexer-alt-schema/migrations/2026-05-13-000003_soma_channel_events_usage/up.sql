-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Voucher-side usage deltas on Settle events + the rating reason
-- code on Rate events. Cumulative-on-channel semantics means each
-- Settle's delta is `(cumulative_X - prior_cumulative_X)`. RateChannel
-- events carry the reason_code (0=Quality, 1=TtftBreach, 2=TtotBreach,
-- 3=NoResponse, 255=Other) so reputation aggregates can break out
-- "why is this provider rated negatively" instead of just "is it".
ALTER TABLE soma_channel_events
    ADD COLUMN tokens_in_delta        BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN tokens_out_delta       BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN cache_read_delta       BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN cache_write_delta      BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN requests_delta         BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN rating_reason_code     SMALLINT NULL;
