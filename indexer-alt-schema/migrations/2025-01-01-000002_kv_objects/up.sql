-- Copyright (c) Mysten Labs, Inc.
-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS kv_objects
(
    object_id                   BYTEA         NOT NULL,
    object_version              BIGINT        NOT NULL,
    serialized_object           BYTEA,
    PRIMARY KEY (object_id, object_version)
);
