// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Wire shape of a provider's `GET /health` response.
//!
//! Shared between the provider server (which serves it) and the proxy's
//! liveness probe (which parses it). Beyond plain liveness, the provider
//! publishes its current `saturation` so the proxy can steer demand away
//! from devices that are already at capacity — the same signal the
//! provider's own price controller uses to reprice on-chain.

use serde::{Deserialize, Serialize};

/// Provider `/health` payload. All non-`status` fields default so an
/// older provider that omits them still deserializes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHealth {
    pub status: String,
    /// Whether the provider's upstream model server answered its own
    /// health check.
    #[serde(default)]
    pub backend_healthy: bool,
    /// Device load in `[0.0, 1.0]` (`None` if the backend can't report
    /// it — e.g. a remote API). `1.0` means fully saturated.
    #[serde(default)]
    pub saturation: Option<f64>,
}
