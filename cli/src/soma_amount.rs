// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::str::FromStr;

const SHANNONS_PER_SOMA: u64 = 1_000_000_000;

/// An amount of SOMA, stored internally as shannons (u64).
///
/// Parsed from user input as SOMA by default:
/// - `"1"` = 1 SOMA = 1,000,000,000 shannons
/// - `"0.5"` = 500,000,000 shannons
/// - `"0.000000001"` = 1 shannon (max 9 decimal places)
#[derive(Debug, Clone, Copy)]
pub struct SomaAmount(u64);

impl SomaAmount {
    /// Return the amount in shannons.
    pub fn shannons(self) -> u64 {
        self.0
    }
}

impl FromStr for SomaAmount {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('.').collect();
        match parts.len() {
            1 => {
                let whole: u64 = parts[0].parse().map_err(|_| format!("invalid amount: {}", s))?;
                whole
                    .checked_mul(SHANNONS_PER_SOMA)
                    .map(SomaAmount)
                    .ok_or_else(|| format!("amount overflow: {} SOMA", s))
            }
            2 => {
                let whole: u64 = if parts[0].is_empty() {
                    0
                } else {
                    parts[0].parse().map_err(|_| format!("invalid amount: {}", s))?
                };
                let frac_str = parts[1];
                if frac_str.is_empty() {
                    return whole
                        .checked_mul(SHANNONS_PER_SOMA)
                        .map(SomaAmount)
                        .ok_or_else(|| format!("amount overflow: {} SOMA", s));
                }
                if frac_str.len() > 9 {
                    return Err("precision beyond 1 shannon (max 9 decimal places)".into());
                }
                let frac_padded = format!("{:0<9}", frac_str);
                let frac: u64 =
                    frac_padded.parse().map_err(|_| format!("invalid fraction: {}", s))?;
                whole
                    .checked_mul(SHANNONS_PER_SOMA)
                    .and_then(|w| w.checked_add(frac))
                    .map(SomaAmount)
                    .ok_or_else(|| format!("amount overflow: {} SOMA", s))
            }
            _ => Err(format!("invalid amount format: {}", s)),
        }
    }
}

impl fmt::Display for SomaAmount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let whole = self.0 / SHANNONS_PER_SOMA;
        let frac = self.0 % SHANNONS_PER_SOMA;
        if frac == 0 {
            write!(f, "{} SOMA", whole)
        } else {
            let frac_str = format!("{:09}", frac);
            let trimmed = frac_str.trim_end_matches('0');
            write!(f, "{}.{} SOMA", whole, trimmed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_whole_soma() {
        assert_eq!(SomaAmount::from_str("1").unwrap().shannons(), 1_000_000_000);
        assert_eq!(SomaAmount::from_str("0").unwrap().shannons(), 0);
        assert_eq!(SomaAmount::from_str("100").unwrap().shannons(), 100_000_000_000);
    }

    #[test]
    fn test_parse_decimal_soma() {
        assert_eq!(SomaAmount::from_str("0.5").unwrap().shannons(), 500_000_000);
        assert_eq!(SomaAmount::from_str("1.5").unwrap().shannons(), 1_500_000_000);
        assert_eq!(SomaAmount::from_str("0.000000001").unwrap().shannons(), 1);
        assert_eq!(SomaAmount::from_str("1.23").unwrap().shannons(), 1_230_000_000);
    }

    #[test]
    fn test_parse_edge_cases() {
        // Trailing dot
        assert_eq!(SomaAmount::from_str("1.").unwrap().shannons(), 1_000_000_000);
        // Leading dot
        assert_eq!(SomaAmount::from_str(".5").unwrap().shannons(), 500_000_000);
    }

    #[test]
    fn test_parse_errors() {
        assert!(SomaAmount::from_str("abc").is_err());
        assert!(SomaAmount::from_str("1.2.3").is_err());
        assert!(SomaAmount::from_str("0.0000000001").is_err()); // 10 decimals
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", SomaAmount(1_000_000_000)), "1 SOMA");
        assert_eq!(format!("{}", SomaAmount(1_500_000_000)), "1.5 SOMA");
        assert_eq!(format!("{}", SomaAmount(1)), "0.000000001 SOMA");
        assert_eq!(format!("{}", SomaAmount(0)), "0 SOMA");
    }
}
