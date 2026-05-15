// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::openai::chat::Usage;

/// Scan an SSE block (one or more `data: …` lines separated by `\n`) and
/// return the first `usage` object found. `[DONE]` and non-JSON lines are
/// skipped silently.
pub fn extract_usage_from_chunk(chunk: &str) -> Option<Usage> {
    for line in chunk.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("data:") {
            let rest = rest.trim();
            if rest == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(rest) {
                if let Some(u) = v.get("usage") {
                    if !u.is_null() {
                        if let Ok(usage) = serde_json::from_value::<Usage>(u.clone()) {
                            return Some(usage);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Position of the first `\n\n` SSE record terminator in `buf`, or `None`.
pub fn find_double_newline(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}
