// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Pricing math: per-request worst-case (signed in pre-flight) and realized
//! (charged in post-flight). Costs are integer micros (1 micro = $1e-6).

use crate::catalog::{ModelCard, Pricing};
use crate::openai::chat::Usage;
use crate::openai::ChatRequest;
use crate::tokenizer::count_messages;

/// Parse a fixed-point decimal string into a u128 with the given number of
/// fractional digits. Truncates excess fractional digits (toward zero).
pub fn parse_fixed(price: &str, frac_digits: u32) -> u128 {
    let s = price.trim();
    let (int_part, frac_part) = match s.split_once('.') {
        Some((a, b)) => (a, b),
        None => (s, ""),
    };
    let int_part = int_part.trim_start_matches('+');
    let int_val: u128 = int_part.parse().unwrap_or(0);
    let mut frac_val: u128 = 0;
    let mut count = 0u32;
    for c in frac_part.chars() {
        if !c.is_ascii_digit() {
            break;
        }
        if count >= frac_digits {
            break;
        }
        frac_val = frac_val * 10 + (c as u128 - '0' as u128);
        count += 1;
    }
    while count < frac_digits {
        frac_val *= 10;
        count += 1;
    }
    int_val
        .saturating_mul(10u128.pow(frac_digits))
        .saturating_add(frac_val)
}

/// `price_str` USD per unit, quantity. Result: `ceil(USD × 1e6)` micros.
pub fn cost_micros(price_str: &str, quantity: u64) -> u64 {
    let scaled = parse_fixed(price_str, 12);
    let cost = scaled.saturating_mul(quantity as u128);
    let micros = (cost + 999_999) / 1_000_000;
    micros.min(u128::from(u64::MAX)) as u64
}

#[derive(Default, Debug, Clone, Copy)]
pub struct UsageCounts {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub cache_write_tokens: u64,
}

pub fn realized_cost_micros(price: &Pricing, u: &UsageCounts) -> u64 {
    let uncached_input = u.prompt_tokens.saturating_sub(u.cached_tokens);
    cost_micros(&price.prompt, uncached_input)
        .saturating_add(cost_micros(&price.input_cache_read, u.cached_tokens))
        .saturating_add(cost_micros(&price.input_cache_write, u.cache_write_tokens))
        .saturating_add(cost_micros(&price.completion, u.completion_tokens))
        .saturating_add(cost_micros(&price.request, 1))
}

pub fn worst_case_cost_micros(card: &ModelCard, input_tokens: u64, max_output_tokens: u64) -> u64 {
    // 10% safety margin on input tokens (cl100k may under-count for non-GPT
    // architectures; the margin absorbs the gap).
    let input_with_margin = input_tokens
        .saturating_add(input_tokens / 10)
        .max(input_tokens);
    cost_micros(&card.pricing.prompt, input_with_margin)
        .saturating_add(cost_micros(&card.pricing.completion, max_output_tokens))
        .saturating_add(cost_micros(&card.pricing.request, 1))
}

pub fn worst_case_for_request(card: &ModelCard, req: &ChatRequest) -> u64 {
    let input_tokens = count_messages(&req.messages);
    let max_output = req
        .max_tokens
        .or(req.max_completion_tokens)
        .or(card.top_provider.max_completion_tokens)
        .unwrap_or(1024);
    worst_case_cost_micros(card, input_tokens, max_output)
}

pub fn realized_for_usage(card: &ModelCard, u: &Usage) -> u64 {
    let cached = u
        .prompt_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .unwrap_or(0);
    let cache_write = u
        .prompt_tokens_details
        .as_ref()
        .map(|d| d.cache_write_tokens)
        .unwrap_or(0);
    realized_cost_micros(
        &card.pricing,
        &UsageCounts {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            cached_tokens: cached,
            cache_write_tokens: cache_write,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fixed_simple() {
        assert_eq!(parse_fixed("1", 6), 1_000_000);
        assert_eq!(parse_fixed("0.5", 6), 500_000);
        assert_eq!(parse_fixed("0.000001", 6), 1);
        assert_eq!(parse_fixed("0.00000028", 12), 280_000);
    }
    #[test]
    fn cost_micros_basic() {
        // $0.00000028 per token × 1e6 tokens = $0.28 = 280_000 micros.
        assert_eq!(cost_micros("0.00000028", 1_000_000), 280_000);
        assert_eq!(cost_micros("0", 1_000_000), 0);
    }
}
