-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- On-chain provider registry mirror. One row per provider address.
-- INSERT on RegisterProvider; UPDATE (endpoint, registered_at_ms,
-- last_update_cp) on UpdateProvider. Off-chain liveness logic uses
-- `registered_at_ms` against a staleness threshold to decide whether
-- a provider is in the active set.
CREATE TABLE soma_providers (
    address           BYTEA  NOT NULL PRIMARY KEY,
    endpoint          TEXT   NOT NULL,
    registered_at_ms  BIGINT NOT NULL,
    -- Last checkpoint at which this row changed. Lets buyers ask
    -- "active in the last N checkpoints" cheaply.
    last_update_cp    BIGINT NOT NULL
);

CREATE INDEX soma_providers_active
    ON soma_providers (registered_at_ms DESC);
