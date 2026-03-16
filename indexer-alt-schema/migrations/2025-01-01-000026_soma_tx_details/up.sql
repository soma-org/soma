-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- Detailed transaction information: kind label, sender, epoch, and optional
-- kind-specific metadata as JSON.  Enables labeled tx feeds and kind-filtered
-- queries (e.g. "show all SubmitData transactions").

CREATE TABLE IF NOT EXISTS soma_tx_details
(
    tx_sequence_number  BIGINT       NOT NULL PRIMARY KEY,
    tx_digest           BYTEA        NOT NULL,
    kind                TEXT         NOT NULL,
    sender              BYTEA        NOT NULL,
    epoch               BIGINT       NOT NULL,
    timestamp_ms        BIGINT       NOT NULL,
    metadata_json       TEXT
);

CREATE INDEX IF NOT EXISTS soma_tx_details_kind
ON soma_tx_details (kind);

CREATE INDEX IF NOT EXISTS soma_tx_details_sender
ON soma_tx_details (sender);

CREATE INDEX IF NOT EXISTS soma_tx_details_epoch
ON soma_tx_details (epoch);
