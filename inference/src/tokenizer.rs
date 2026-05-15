// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cheap upper-bound token counter for cost estimation. Uses cl100k_base for
//! all models — close enough for the worst-case estimate that only feeds the
//! 10% input safety margin in [`crate::pricing::worst_case_cost_micros`].

use once_cell::sync::Lazy;
use tiktoken_rs::{cl100k_base, CoreBPE};

use crate::openai::chat::{ChatMessage, MessageContent};

static CL100K: Lazy<Option<CoreBPE>> = Lazy::new(|| cl100k_base().ok());

pub fn count_text(text: &str) -> u64 {
    if let Some(bpe) = CL100K.as_ref() {
        bpe.encode_with_special_tokens(text).len() as u64
    } else {
        // Fallback: ~4 chars per token.
        ((text.len() as u64) + 3) / 4
    }
}

pub fn count_messages(messages: &[ChatMessage]) -> u64 {
    let mut total: u64 = 0;
    for m in messages {
        // role overhead
        total += 4;
        match &m.content {
            Some(MessageContent::Text(s)) => total += count_text(s),
            Some(MessageContent::Parts(parts)) => {
                for p in parts {
                    if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                        total += count_text(t);
                    }
                }
            }
            None => {}
        }
        if let Some(tcs) = &m.tool_calls {
            if let Ok(s) = serde_json::to_string(tcs) {
                total += count_text(&s);
            }
        }
    }
    total + 2
}
