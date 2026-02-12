// Failpoint infrastructure for fault injection in tests.
//
// Ported from Sui's `sui-macros` crate. All macros compile to no-ops unless
// `cfg(msim)` or `cfg(fail_points)` is set — zero cost in production.
//
// ## Usage in production code:
//
// ```rust
// utils::fail_point!("before-open-new-epoch-store");
// ```
//
// ## Usage in tests:
//
// ```rust
// use utils::fp::register_fail_point;
// register_fail_point("before-open-new-epoch-store", || {
//     panic!("injected crash");
// });
// ```

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;

type FpCallback = dyn Fn() -> Box<dyn std::any::Any + Send + 'static> + Send + Sync;
type FpMap = HashMap<&'static str, Arc<FpCallback>>;

#[cfg(msim)]
fn with_fp_map<T>(func: impl FnOnce(&mut FpMap) -> T) -> T {
    thread_local! {
        static MAP: std::cell::RefCell<FpMap> = std::cell::RefCell::new(HashMap::new());
    }

    MAP.with(|val| func(&mut val.borrow_mut()))
}

#[cfg(not(msim))]
fn with_fp_map<T>(func: impl FnOnce(&mut FpMap) -> T) -> T {
    use std::sync::{LazyLock, Mutex};

    static MAP: LazyLock<Mutex<FpMap>> = LazyLock::new(|| Mutex::new(HashMap::new()));
    let mut map = MAP.lock().unwrap();
    func(&mut map)
}

fn get_callback(identifier: &'static str) -> Option<Arc<FpCallback>> {
    with_fp_map(|map| map.get(identifier).cloned())
}

fn register_fail_point_impl(identifier: &'static str, callback: Arc<FpCallback>) {
    with_fp_map(move |map| {
        assert!(
            map.insert(identifier, callback).is_none(),
            "duplicate fail point registration: {identifier}"
        );
    })
}

fn clear_fail_point_impl(identifier: &'static str) {
    with_fp_map(move |map| {
        assert!(map.remove(identifier).is_some(), "fail point {identifier:?} does not exist",);
    })
}

// --- Public handle functions (called by macros) ---

pub fn handle_fail_point(identifier: &'static str) {
    if let Some(callback) = get_callback(identifier) {
        let result = callback();
        if result.downcast::<()>().is_err() {
            panic!("sync failpoint must return ()");
        }
        tracing::trace!("hit failpoint {}", identifier);
    }
}

pub async fn handle_fail_point_async(identifier: &'static str) {
    if let Some(callback) = get_callback(identifier) {
        tracing::trace!("hit async failpoint {}", identifier);
        let result = callback();
        match result.downcast::<BoxFuture<'static, ()>>() {
            Ok(fut) => (*fut).await,
            Err(_) => panic!("async failpoint must return BoxFuture<'static, ()>"),
        }
    }
}

pub fn handle_fail_point_if(identifier: &'static str) -> bool {
    if let Some(callback) = get_callback(identifier) {
        tracing::trace!("hit failpoint_if {}", identifier);
        match callback().downcast::<bool>() {
            Ok(b) => *b,
            Err(_) => panic!("failpoint-if must return bool"),
        }
    } else {
        false
    }
}

pub fn handle_fail_point_arg<T: Send + 'static>(identifier: &'static str) -> Option<T> {
    if let Some(callback) = get_callback(identifier) {
        tracing::trace!("hit failpoint_arg {}", identifier);
        match callback().downcast::<Option<T>>() {
            Ok(opt) => *opt,
            Err(_) => panic!("failpoint-arg must return Option<T>"),
        }
    } else {
        None
    }
}

// --- Public registration API (called from tests) ---

/// Register a synchronous fail point callback.
pub fn register_fail_point(
    identifier: &'static str,
    callback: impl Fn() + Sync + Send + 'static,
) {
    register_fail_point_impl(
        identifier,
        Arc::new(move || {
            callback();
            Box::new(())
        }),
    );
}

/// Register an async fail point. The callback returns a future that can yield
/// (e.g., by sleeping), allowing delay injection.
pub fn register_fail_point_async<F>(
    identifier: &'static str,
    callback: impl Fn() -> F + Sync + Send + 'static,
) where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    register_fail_point_impl(
        identifier,
        Arc::new(move || {
            let result: BoxFuture<'static, ()> = Box::pin(callback());
            Box::new(result)
        }),
    );
}

/// Register a conditional fail point. The callback returns `bool`; if `true`,
/// the inline closure at the fail point site is executed.
pub fn register_fail_point_if(
    identifier: &'static str,
    callback: impl Fn() -> bool + Sync + Send + 'static,
) {
    register_fail_point_impl(identifier, Arc::new(move || Box::new(callback())));
}

/// Register a parameterized fail point. The callback returns `Option<T>`; if
/// `Some(v)`, then `v` is passed to the inline closure at the fail point site.
pub fn register_fail_point_arg<T: Send + 'static>(
    identifier: &'static str,
    callback: impl Fn() -> Option<T> + Sync + Send + 'static,
) {
    register_fail_point_impl(identifier, Arc::new(move || Box::new(callback())));
}

/// Register the same callback for multiple fail points.
pub fn register_fail_points(
    identifiers: &[&'static str],
    callback: impl Fn() + Sync + Send + 'static,
) {
    let cb: Arc<FpCallback> = Arc::new(move || {
        callback();
        Box::new(())
    });
    for id in identifiers {
        register_fail_point_impl(id, cb.clone());
    }
}

/// Remove a registered fail point.
pub fn clear_fail_point(identifier: &'static str) {
    clear_fail_point_impl(identifier);
}

// --- Macros ---

// Macros are defined in utils/src/lib.rs to avoid module/macro name collisions.

#[cfg(test)]
#[cfg(any(msim, fail_points))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn test_sync_fail_point() {
        static HIT: AtomicBool = AtomicBool::new(false);

        register_fail_point("test-sync", || {
            HIT.store(true, Ordering::SeqCst);
        });

        assert!(!HIT.load(Ordering::SeqCst));
        crate::fail_point!("test-sync");
        assert!(HIT.load(Ordering::SeqCst));

        clear_fail_point("test-sync");
    }

    #[test]
    fn test_fail_point_if() {
        register_fail_point_if("test-if", || true);

        let mut triggered = false;
        crate::fail_point_if!("test-if", || {
            triggered = true;
        });
        assert!(triggered);

        clear_fail_point("test-if");
    }

    #[test]
    fn test_fail_point_arg() {
        register_fail_point_arg("test-arg", || Some(42u64));

        let mut value = 0u64;
        crate::fail_point_arg!("test-arg", |v: u64| {
            value = v;
        });
        assert_eq!(value, 42);

        clear_fail_point("test-arg");
    }

    #[test]
    fn test_unregistered_fail_point_is_noop() {
        // Should not panic or do anything
        crate::fail_point!("nonexistent");

        let mut triggered = false;
        crate::fail_point_if!("nonexistent-if", || {
            triggered = true;
        });
        assert!(!triggered);
    }

    #[tokio::test]
    async fn test_async_fail_point() {
        static HIT: AtomicBool = AtomicBool::new(false);

        register_fail_point_async("test-async", || async {
            HIT.store(true, Ordering::SeqCst);
        });

        assert!(!HIT.load(Ordering::SeqCst));
        crate::fail_point_async!("test-async");
        assert!(HIT.load(Ordering::SeqCst));

        clear_fail_point("test-async");
    }
}
