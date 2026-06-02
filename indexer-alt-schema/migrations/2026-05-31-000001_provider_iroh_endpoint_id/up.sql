-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- The provider's iroh EndpointId (canonical z-base-32 string), mirrored
-- from the on-chain `Provider.iroh_endpoint_id`. Buyers dial this key over
-- iroh. Empty for providers with no iroh identity. Defaulted to '' so
-- existing rows backfill cleanly; the indexer overwrites it on the next
-- Register/UpdateProvider for each provider.
ALTER TABLE soma_providers
    ADD COLUMN iroh_endpoint_id TEXT NOT NULL DEFAULT '';
