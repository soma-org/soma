// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

pub mod agg;
pub mod codec;
pub mod fp;
pub mod logging;
pub mod notify_once;
pub mod notify_read;

// --- Failpoint macros ---
//
// Defined here (crate root) rather than in `fp.rs` to avoid a module/macro
// name collision when callers write `utils::fail_point!(...)`.

#[cfg(any(msim, fail_points))]
#[macro_export]
macro_rules! fail_point {
    ($tag:expr) => {
        $crate::fp::handle_fail_point($tag)
    };
}

#[cfg(not(any(msim, fail_points)))]
#[macro_export]
macro_rules! fail_point {
    ($tag:expr) => {};
}

#[cfg(any(msim, fail_points))]
#[macro_export]
macro_rules! fail_point_async {
    ($tag:expr) => {
        $crate::fp::handle_fail_point_async($tag).await
    };
}

#[cfg(not(any(msim, fail_points)))]
#[macro_export]
macro_rules! fail_point_async {
    ($tag:expr) => {};
}

#[cfg(any(msim, fail_points))]
#[macro_export]
macro_rules! fail_point_if {
    ($tag:expr, $body:expr) => {
        if $crate::fp::handle_fail_point_if($tag) {
            $body();
        }
    };
}

#[cfg(not(any(msim, fail_points)))]
#[macro_export]
macro_rules! fail_point_if {
    ($tag:expr, $body:expr) => {};
}

#[cfg(any(msim, fail_points))]
#[macro_export]
macro_rules! fail_point_arg {
    ($tag:expr, |$v:ident : $t:ty| $body:expr) => {
        if let Some($v) = $crate::fp::handle_fail_point_arg::<$t>($tag) {
            $body;
        }
    };
}

#[cfg(not(any(msim, fail_points)))]
#[macro_export]
macro_rules! fail_point_arg {
    ($tag:expr, |$v:ident : $t:ty| $body:expr) => {};
}
