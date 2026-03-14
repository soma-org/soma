-- Copyright (c) Mysten Labs, Inc.
-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS obj_info
(
    object_id                   BYTEA         NOT NULL,
    cp_sequence_number          BIGINT        NOT NULL,
    owner_kind                  SMALLINT,
    owner_id                    BYTEA,
    package                     BYTEA,
    module                      TEXT,
    name                        TEXT,
    instantiation               BYTEA,
    PRIMARY KEY (object_id, cp_sequence_number)
);

CREATE INDEX IF NOT EXISTS obj_info_owner_object_id_desc
ON obj_info (owner_kind, owner_id, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_pkg_object_id_desc
ON obj_info (package, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_mod_object_id_desc
ON obj_info (package, module, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_name_object_id_desc
ON obj_info (package, module, name, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_inst_object_id_desc
ON obj_info (package, module, name, instantiation, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_owner_pkg_object_id_desc
ON obj_info (owner_kind, owner_id, package, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_owner_mod_object_id_desc
ON obj_info (owner_kind, owner_id, package, module, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_owner_name_object_id_desc
ON obj_info (owner_kind, owner_id, package, module, name, cp_sequence_number DESC, object_id DESC);

CREATE INDEX IF NOT EXISTS obj_info_owner_inst_object_id_desc
ON obj_info (owner_kind, owner_id, package, module, name, instantiation, cp_sequence_number DESC, object_id DESC);
