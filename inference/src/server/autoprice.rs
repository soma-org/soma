// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Utilization-targeting price controller.
//!
//! A provider serving a local model wants its device near-saturated but
//! not overwhelmed: idle slots earn nothing, an over-long queue breaches
//! latency SLAs. This loop reads the backend's saturation each tick and
//! submits an `UpdateOffering` that nudges the on-chain price toward a
//! target utilization band — up when over-loaded (shed demand), down when
//! idle (attract demand), never below a configurable floor.
//!
//! Only the prompt and completion per-1k rates move; cache and per-request
//! rates plus SLA bounds pass through from the boot offering. New channels
//! snapshot whichever price is live at open time, so in-flight channels are
//! never repriced underneath a buyer.

use std::sync::Arc;

use sdk::offering::OfferingPrices;
use sdk::wallet_context::WalletContext;
use types::base::SomaAddress;

use crate::server::backend::Backend;
use crate::server::config::AutoPrice;

/// Per-model working price state: the boot offering (for the fields the
/// controller leaves fixed) plus the live prompt/completion rates it steps.
#[derive(Debug, Clone)]
pub struct ModelPrice {
    pub model_id: String,
    pub base: OfferingPrices,
    pub cur_prompt: u64,
    pub cur_completion: u64,
}

impl ModelPrice {
    /// Seed from a boot offering, lifting prompt/completion up to the
    /// configured floors so the controller starts inside its own bounds.
    pub fn new(model_id: String, base: OfferingPrices, cfg: &AutoPrice) -> Self {
        let cur_prompt = base.prompt_micros_per_1k.max(cfg.min_prompt_micros_per_1k);
        let cur_completion = base.completion_micros_per_1k.max(cfg.min_completion_micros_per_1k);
        Self { model_id, base, cur_prompt, cur_completion }
    }

    /// The `OfferingPrices` to submit, built from `base` with the live
    /// prompt/completion rates substituted in.
    pub fn prices(&self) -> OfferingPrices {
        OfferingPrices {
            prompt_micros_per_1k: self.cur_prompt,
            completion_micros_per_1k: self.cur_completion,
            ..self.base.clone()
        }
    }
}

/// Pure stepping rule. Given current rates and a saturation reading,
/// returns the next `(prompt, completion)` rates — or `None` when no
/// transaction should be sent: either saturation is inside the target band,
/// or the rates are already pinned at the bound we'd move toward.
pub fn step_prices(
    cur_prompt: u64,
    cur_completion: u64,
    saturation: f64,
    cfg: &AutoPrice,
) -> Option<(u64, u64)> {
    let factor = if saturation > cfg.target_high {
        1.0 + cfg.step_frac
    } else if saturation < cfg.target_low {
        1.0 - cfg.step_frac
    } else {
        return None;
    };

    let bump = |v: u64, min: u64, max: Option<u64>| -> u64 {
        let mut n = (v as f64 * factor).round().max(0.0) as u64;
        // A 1µ floor on the *delta* so low prices still move off small
        // bases (e.g. 1µ × 1.1 rounds back to 1µ otherwise).
        if factor > 1.0 && n <= v {
            n = v.saturating_add(1);
        } else if factor < 1.0 && n >= v && v > 0 {
            n = v.saturating_sub(1);
        }
        n = n.max(min);
        if let Some(m) = max {
            n = n.min(m);
        }
        n
    };

    let next_prompt = bump(cur_prompt, cfg.min_prompt_micros_per_1k, None);
    let next_completion =
        bump(cur_completion, cfg.min_completion_micros_per_1k, cfg.max_completion_micros_per_1k);

    if (next_prompt, next_completion) == (cur_prompt, cur_completion) {
        return None;
    }
    Some((next_prompt, next_completion))
}

/// Spawn the background price loop. No-op (logs and returns) when disabled,
/// when there are no offerings, or when `interval_secs == 0`.
pub fn spawn(
    wallet: Arc<WalletContext>,
    address: SomaAddress,
    backend: Arc<dyn Backend>,
    mut models: Vec<ModelPrice>,
    cfg: AutoPrice,
) {
    if !cfg.enabled {
        tracing::info!("autoprice disabled");
        return;
    }
    if models.is_empty() || cfg.interval_secs == 0 {
        tracing::info!("autoprice: nothing to price (no offerings or interval 0)");
        return;
    }
    tracing::info!(
        interval_secs = cfg.interval_secs,
        target_low = cfg.target_low,
        target_high = cfg.target_high,
        step_frac = cfg.step_frac,
        "autoprice armed",
    );
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(cfg.interval_secs));
        // Skip the immediate first tick — boot just registered offerings.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let Some(sat) = backend.saturation().await else {
                tracing::debug!("autoprice: no saturation signal this tick; holding price");
                continue;
            };
            for m in models.iter_mut() {
                let Some((np, nc)) = step_prices(m.cur_prompt, m.cur_completion, sat, &cfg) else {
                    continue;
                };
                m.cur_prompt = np;
                m.cur_completion = nc;
                let prices = m.prices();
                match sdk::offering::register_or_update_no_wait(
                    &wallet,
                    address,
                    m.model_id.clone(),
                    prices,
                )
                .await
                {
                    Ok(()) => tracing::info!(
                        model_id = %m.model_id,
                        saturation = sat,
                        prompt = np,
                        completion = nc,
                        "autoprice: offering repriced",
                    ),
                    Err(e) => tracing::warn!(
                        model_id = %m.model_id,
                        err = %e,
                        "autoprice: UpdateOffering failed (will retry next tick)",
                    ),
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> AutoPrice {
        AutoPrice {
            enabled: true,
            interval_secs: 120,
            target_low: 0.65,
            target_high: 0.85,
            step_frac: 0.10,
            min_prompt_micros_per_1k: 100,
            min_completion_micros_per_1k: 500,
            max_completion_micros_per_1k: Some(50_000),
        }
    }

    #[test]
    fn in_band_holds() {
        // 0.75 sits between low (0.65) and high (0.85) → no change.
        assert_eq!(step_prices(1_000, 5_000, 0.75, &cfg()), None);
    }

    #[test]
    fn over_saturated_raises() {
        let (p, c) = step_prices(1_000, 5_000, 0.95, &cfg()).unwrap();
        assert_eq!(p, 1_100);
        assert_eq!(c, 5_500);
    }

    #[test]
    fn under_saturated_lowers() {
        let (p, c) = step_prices(1_000, 5_000, 0.20, &cfg()).unwrap();
        assert_eq!(p, 900);
        assert_eq!(c, 4_500);
    }

    #[test]
    fn lowering_clamps_at_min_and_stops() {
        // Already at the floor; under-saturated → no further move, no tx.
        assert_eq!(step_prices(100, 500, 0.10, &cfg()), None);
    }

    #[test]
    fn raising_clamps_at_completion_max() {
        // Completion already at ceiling, prompt still free to rise → tx
        // (prompt changed) but completion pinned at max.
        let (p, c) = step_prices(1_000, 50_000, 0.99, &cfg()).unwrap();
        assert_eq!(p, 1_100);
        assert_eq!(c, 50_000);
    }

    #[test]
    fn small_base_still_moves_up() {
        // 1µ × 1.1 = 1.1 → rounds to 1; the delta floor forces it to 2.
        let mut c = cfg();
        c.min_prompt_micros_per_1k = 0;
        c.min_completion_micros_per_1k = 0;
        let (p, comp) = step_prices(1, 1, 0.99, &c).unwrap();
        assert_eq!(p, 2);
        assert_eq!(comp, 2);
    }
}
