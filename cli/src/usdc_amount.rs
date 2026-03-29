// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::str::FromStr;

const MICRODOLLARS_PER_USDC: u64 = 1_000_000;

/// An amount of USDC, stored internally as microdollars (u64).
///
/// Parsed from user input as USDC by default:
/// - `"1"` = 1 USDC = 1,000,000 microdollars
/// - `"0.50"` = 500,000 microdollars
/// - `"0.000001"` = 1 microdollar (max 6 decimal places)
#[derive(Debug, Clone, Copy)]
pub struct UsdcAmount(u64);

impl UsdcAmount {
    /// Return the amount in microdollars.
    pub fn microdollars(self) -> u64 {
        self.0
    }
}

impl FromStr for UsdcAmount {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('.').collect();
        match parts.len() {
            1 => {
                let whole: u64 = parts[0].parse().map_err(|_| format!("invalid amount: {}", s))?;
                whole
                    .checked_mul(MICRODOLLARS_PER_USDC)
                    .map(UsdcAmount)
                    .ok_or_else(|| format!("amount overflow: {} USDC", s))
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
                        .checked_mul(MICRODOLLARS_PER_USDC)
                        .map(UsdcAmount)
                        .ok_or_else(|| format!("amount overflow: {} USDC", s));
                }
                if frac_str.len() > 6 {
                    return Err("precision beyond 1 microdollar (max 6 decimal places)".into());
                }
                let frac_padded = format!("{:0<6}", frac_str);
                let frac: u64 =
                    frac_padded.parse().map_err(|_| format!("invalid fraction: {}", s))?;
                whole
                    .checked_mul(MICRODOLLARS_PER_USDC)
                    .and_then(|w| w.checked_add(frac))
                    .map(UsdcAmount)
                    .ok_or_else(|| format!("amount overflow: {} USDC", s))
            }
            _ => Err(format!("invalid amount format: {}", s)),
        }
    }
}

impl fmt::Display for UsdcAmount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let whole = self.0 / MICRODOLLARS_PER_USDC;
        let frac = self.0 % MICRODOLLARS_PER_USDC;
        if frac == 0 {
            write!(f, "{} USDC", whole)
        } else {
            let frac_str = format!("{:06}", frac);
            let trimmed = frac_str.trim_end_matches('0');
            write!(f, "{}.{} USDC", whole, trimmed)
        }
    }
}

/// Format a microdollar amount for display.
pub fn format_usdc(microdollars: u64) -> String {
    UsdcAmount(microdollars).to_string()
}

/// Parse a human-readable duration string into milliseconds.
///
/// Accepts: `30s`, `5m`, `1h`, `1d`, `500ms`, or raw milliseconds as a number.
pub fn parse_duration_ms(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if let Ok(ms) = s.parse::<u64>() {
        return Ok(ms);
    }
    if let Some(v) = s.strip_suffix("ms") {
        return v.parse::<u64>().map_err(|_| format!("invalid duration: {}", s));
    }
    if let Some(v) = s.strip_suffix('s') {
        return v
            .parse::<u64>()
            .map(|v| v * 1_000)
            .map_err(|_| format!("invalid duration: {}", s));
    }
    if let Some(v) = s.strip_suffix('m') {
        return v
            .parse::<u64>()
            .map(|v| v * 60_000)
            .map_err(|_| format!("invalid duration: {}", s));
    }
    if let Some(v) = s.strip_suffix('h') {
        return v
            .parse::<u64>()
            .map(|v| v * 3_600_000)
            .map_err(|_| format!("invalid duration: {}", s));
    }
    if let Some(v) = s.strip_suffix('d') {
        return v
            .parse::<u64>()
            .map(|v| v * 86_400_000)
            .map_err(|_| format!("invalid duration: {}", s));
    }
    Err(format!(
        "invalid duration: '{}'. Use: 30s, 5m, 1h, 1d, 500ms, or raw milliseconds",
        s
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_whole_usdc() {
        assert_eq!(UsdcAmount::from_str("1").unwrap().microdollars(), 1_000_000);
        assert_eq!(UsdcAmount::from_str("0").unwrap().microdollars(), 0);
        assert_eq!(UsdcAmount::from_str("100").unwrap().microdollars(), 100_000_000);
    }

    #[test]
    fn test_parse_decimal_usdc() {
        assert_eq!(UsdcAmount::from_str("0.50").unwrap().microdollars(), 500_000);
        assert_eq!(UsdcAmount::from_str("1.50").unwrap().microdollars(), 1_500_000);
        assert_eq!(UsdcAmount::from_str("0.000001").unwrap().microdollars(), 1);
        assert_eq!(UsdcAmount::from_str("1.23").unwrap().microdollars(), 1_230_000);
    }

    #[test]
    fn test_parse_errors() {
        assert!(UsdcAmount::from_str("abc").is_err());
        assert!(UsdcAmount::from_str("1.2.3").is_err());
        assert!(UsdcAmount::from_str("0.0000001").is_err()); // 7 decimals
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", UsdcAmount(1_000_000)), "1 USDC");
        assert_eq!(format!("{}", UsdcAmount(1_500_000)), "1.5 USDC");
        assert_eq!(format!("{}", UsdcAmount(1)), "0.000001 USDC");
        assert_eq!(format!("{}", UsdcAmount(0)), "0 USDC");
    }

    #[test]
    fn test_parse_duration_ms() {
        assert_eq!(parse_duration_ms("30s").unwrap(), 30_000);
        assert_eq!(parse_duration_ms("5m").unwrap(), 300_000);
        assert_eq!(parse_duration_ms("1h").unwrap(), 3_600_000);
        assert_eq!(parse_duration_ms("1d").unwrap(), 86_400_000);
        assert_eq!(parse_duration_ms("500ms").unwrap(), 500);
        assert_eq!(parse_duration_ms("60000").unwrap(), 60_000);
        assert!(parse_duration_ms("abc").is_err());
    }
}
