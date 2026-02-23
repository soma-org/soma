// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use super::SomaNode;
use authority::authority::AuthorityState;
use std::sync::Arc;
use std::{future::Future, time::Duration};
use types::transaction::Transaction;
use types::{
    base::ConciseableName,
    committee::{self, Committee, CommitteeTrait},
};

/// Wrap SomaNode to allow correct access to SomaNode in simulator tests.
pub struct SomaNodeHandle {
    node: Option<Arc<SomaNode>>,
    shutdown_on_drop: bool,
}

impl SomaNodeHandle {
    pub fn new(node: Arc<SomaNode>) -> Self {
        Self { node: Some(node), shutdown_on_drop: false }
    }

    pub fn inner(&self) -> &Arc<SomaNode> {
        self.node.as_ref().unwrap()
    }

    pub fn with<T>(&self, cb: impl FnOnce(&SomaNode) -> T) -> T {
        let _guard = self.guard();
        cb(self.inner())
    }

    pub fn state(&self) -> Arc<AuthorityState> {
        self.with(|soma_node| soma_node.state())
    }

    pub fn shutdown_on_drop(&mut self) {
        self.shutdown_on_drop = true;
    }
}

impl Clone for SomaNodeHandle {
    fn clone(&self) -> Self {
        Self { node: self.node.clone(), shutdown_on_drop: false }
    }
}

#[cfg(not(msim))]
impl SomaNodeHandle {
    // Must return something to silence lints above at `let _guard = ...`
    fn guard(&self) -> u32 {
        0
    }

    pub async fn with_async<'a, F, R, T>(&'a self, cb: F) -> T
    where
        F: FnOnce(&'a SomaNode) -> R,
        R: Future<Output = T>,
    {
        cb(self.inner()).await
    }
}

#[cfg(msim)]
impl SomaNodeHandle {
    fn guard(&self) -> msim::runtime::NodeEnterGuard {
        self.inner().sim_state.sim_node.enter_node()
    }

    pub async fn with_async<'a, F, R, T>(&'a self, cb: F) -> T
    where
        F: FnOnce(&'a SomaNode) -> R,
        R: Future<Output = T>,
    {
        let fut = cb(self.node.as_ref().unwrap());
        self.inner().sim_state.sim_node.await_future_in_node(fut).await
    }
}

#[cfg(msim)]
impl Drop for SomaNodeHandle {
    fn drop(&mut self) {
        if self.shutdown_on_drop {
            let node_id = self.inner().sim_state.sim_node.id();
            msim::runtime::Handle::try_current().map(|h| h.delete_node(node_id));
        }
    }
}

impl From<Arc<SomaNode>> for SomaNodeHandle {
    fn from(node: Arc<SomaNode>) -> Self {
        SomaNodeHandle::new(node)
    }
}
