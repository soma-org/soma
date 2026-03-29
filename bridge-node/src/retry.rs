//! Retry utilities with exponential backoff.
//!
//! Adapted from Sui's `retry_with_max_elapsed_time!` pattern. Uses the `backoff`
//! crate for exponential backoff with configurable max elapsed time.

use std::time::Duration;

use backoff::ExponentialBackoff;
use tracing::warn;

use crate::error::{BridgeError, BridgeResult};

/// Default initial backoff interval (400ms).
const DEFAULT_INITIAL_INTERVAL_MS: u64 = 400;

/// Default maximum backoff interval (120s).
const DEFAULT_MAX_INTERVAL_SECS: u64 = 120;

/// Default multiplier (2x).
const DEFAULT_MULTIPLIER: f64 = 2.0;

/// Create an exponential backoff configuration.
pub fn exponential_backoff(max_elapsed_time: Duration) -> ExponentialBackoff {
    ExponentialBackoff {
        initial_interval: Duration::from_millis(DEFAULT_INITIAL_INTERVAL_MS),
        max_interval: Duration::from_secs(DEFAULT_MAX_INTERVAL_SECS),
        multiplier: DEFAULT_MULTIPLIER,
        max_elapsed_time: Some(max_elapsed_time),
        ..ExponentialBackoff::default()
    }
}

/// Retry an async operation with exponential backoff.
///
/// Only retries on `BridgeError::TransientProviderError` and
/// `BridgeError::ProviderError`. All other errors are returned immediately.
///
/// Returns the successful result or the last error if max elapsed time is exceeded.
pub async fn retry_with_backoff<F, Fut, T>(
    operation_name: &str,
    max_elapsed: Duration,
    mut f: F,
) -> BridgeResult<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = BridgeResult<T>>,
{
    let backoff = exponential_backoff(max_elapsed);
    let mut current_interval = backoff.initial_interval;
    let start = std::time::Instant::now();
    let mut attempt = 0u32;

    loop {
        attempt += 1;
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if is_retryable(&e) => {
                let elapsed = start.elapsed();
                if elapsed >= max_elapsed {
                    warn!(
                        operation = operation_name,
                        attempt,
                        elapsed_ms = elapsed.as_millis() as u64,
                        "Max retry time exceeded"
                    );
                    return Err(e);
                }

                warn!(
                    operation = operation_name,
                    attempt,
                    backoff_ms = current_interval.as_millis() as u64,
                    error = %e,
                    "Retrying after transient error"
                );

                tokio::time::sleep(current_interval).await;
                // Exponential increase, capped at max_interval
                current_interval = std::cmp::min(
                    Duration::from_secs_f64(
                        current_interval.as_secs_f64() * backoff.multiplier,
                    ),
                    backoff.max_interval,
                );
            }
            Err(e) => {
                // Non-retryable error — return immediately
                return Err(e);
            }
        }
    }
}

/// Determine if an error is retryable.
fn is_retryable(error: &BridgeError) -> bool {
    matches!(
        error,
        BridgeError::TransientProviderError(_) | BridgeError::ProviderError(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_succeeds_after_transient_failures() {
        let attempt_count = Arc::new(AtomicU32::new(0));
        let count = attempt_count.clone();

        let result = retry_with_backoff(
            "test_op",
            Duration::from_secs(10),
            || {
                let count = count.clone();
                async move {
                    let attempt = count.fetch_add(1, Ordering::Relaxed);
                    if attempt < 2 {
                        Err(BridgeError::TransientProviderError(
                            "too many results".into(),
                        ))
                    } else {
                        Ok(42u32)
                    }
                }
            },
        )
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt_count.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_retry_non_retryable_error_returns_immediately() {
        let attempt_count = Arc::new(AtomicU32::new(0));
        let count = attempt_count.clone();

        let result: BridgeResult<()> = retry_with_backoff(
            "test_op",
            Duration::from_secs(10),
            || {
                let count = count.clone();
                async move {
                    count.fetch_add(1, Ordering::Relaxed);
                    Err(BridgeError::ConfigError("bad config".into()))
                }
            },
        )
        .await;

        assert!(result.is_err());
        // Should NOT retry on non-retryable errors
        assert_eq!(attempt_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_retry_timeout_exceeded() {
        let attempt_count = Arc::new(AtomicU32::new(0));
        let count = attempt_count.clone();

        let result: BridgeResult<()> = retry_with_backoff(
            "test_op",
            Duration::from_millis(500), // very short timeout
            || {
                let count = count.clone();
                async move {
                    count.fetch_add(1, Ordering::Relaxed);
                    Err(BridgeError::ProviderError("always fails".into()))
                }
            },
        )
        .await;

        assert!(result.is_err());
        // Should have retried at least once before timing out
        assert!(attempt_count.load(Ordering::Relaxed) >= 2);
    }

    #[tokio::test]
    async fn test_retry_succeeds_on_first_try() {
        let result = retry_with_backoff(
            "test_op",
            Duration::from_secs(10),
            || async { Ok::<_, BridgeError>(99u32) },
        )
        .await;

        assert_eq!(result.unwrap(), 99);
    }
}
