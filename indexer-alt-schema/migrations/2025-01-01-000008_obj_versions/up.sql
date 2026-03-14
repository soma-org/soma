-- Copyright (c) Mysten Labs, Inc.
-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS obj_versions
(
    object_id                   BYTEA         NOT NULL,
    object_version              BIGINT        NOT NULL,
    object_digest               BYTEA,
    cp_sequence_number          BIGINT        NOT NULL,
    PRIMARY KEY (object_id, object_version)
);

CREATE INDEX IF NOT EXISTS obj_versions_cp_sequence_number
ON obj_versions (cp_sequence_number);

CREATE INDEX IF NOT EXISTS obj_versions_id_cp_version
ON obj_versions (object_id, cp_sequence_number DESC, object_version DESC);
