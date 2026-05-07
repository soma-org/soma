-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- On-chain provider registry mirror. One row per provider address
-- that currently has a `Provider` shared object on chain.
--
-- INSERT on RegisterProvider; UPDATE (endpoint, last_update_cp) on
-- UpdateProvider. Liveness is purely off-chain (the proxy probes the
-- advertised endpoint over HTTP), so this table answers "who is
-- *claiming* to provide", not "who is reachable right now".
CREATE TABLE soma_providers (
    address           BYTEA  NOT NULL PRIMARY KEY,
    endpoint          TEXT   NOT NULL,
    -- Last checkpoint at which this row changed (insert or endpoint
    -- update). Lets readers ask "who registered or changed endpoints
    -- in the last N checkpoints" cheaply.
    last_update_cp    BIGINT NOT NULL
);
