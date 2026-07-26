-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Per-channel state mirror, sourced from on-chain Channel objects
-- by the `soma_channels` indexer handler. One row per channel.
-- INSERT on OpenChannel; UPDATE on Settle / RequestClose / TopUp /
-- WithdrawAfterTimeout (the last keeps the row but flips status to
-- 'withdrawn' — the on-chain Channel is deleted but we retain the
-- record for client recovery and reputation).
CREATE TABLE soma_channels (
    channel_id            BYTEA      NOT NULL PRIMARY KEY,
    payer                 BYTEA      NOT NULL,
    payee                 BYTEA      NOT NULL,
    authorized_signer     BYTEA      NOT NULL,
    -- Coin denomination label: "USDC" / "SOMA". Same string form
    -- used elsewhere (e.g. soma_balance_deltas.coin_type).
    token                 TEXT       NOT NULL,
    -- Live deposit (decreases on Settle, increases on TopUp). 0 once
    -- WithdrawAfterTimeout has fired.
    deposit               BIGINT     NOT NULL,
    -- Highest cumulative_amount paid out so far. Strictly
    -- non-decreasing.
    settled_amount        BIGINT     NOT NULL,
    -- NULL = open; Some(ts) = RequestClose was called at that ms.
    -- Cleared back to NULL on a TopUp that overrides the close timer.
    close_requested_at_ms BIGINT,
    -- Status discriminator. 0 = open, 1 = closing, 2 = withdrawn.
    -- Cheap to filter on; mirrors the off-chain `ChannelStatus`
    -- enum in the inference crate.
    status                SMALLINT   NOT NULL,
    opened_at_cp          BIGINT     NOT NULL,
    -- Open-tx digest. Lets clients reconstruct the channel id
    -- via `predicted_channel_id(open_tx_digest)`.
    opened_tx_digest      BYTEA      NOT NULL,
    last_update_cp        BIGINT     NOT NULL
);

-- Buyer-side enumeration: "what channels do I have open?"
CREATE INDEX soma_channels_payer_status
    ON soma_channels (payer, status, last_update_cp DESC);

-- Provider-side enumeration: "what channels are settled to me?"
CREATE INDEX soma_channels_payee_status
    ON soma_channels (payee, status, last_update_cp DESC);

-- Reputation aggregates use this index for descending-time scans.
CREATE INDEX soma_channels_payee_recent
    ON soma_channels (payee, last_update_cp DESC);
