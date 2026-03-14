// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::future::poll_fn;
use std::panic;
use std::pin::pin;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::try_join_all;
use futures::stream::Stream;
use futures::try_join;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

/// Runtime configuration for adaptive concurrency control.
#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    pub initial: usize,
    pub min: usize,
    pub max: usize,
    pub dead_band_low: f64,
    pub dead_band_high: f64,
}

/// Snapshot of concurrency stats passed to the `report` callback.
#[derive(Debug, Clone, Copy)]
pub struct ConcurrencyStats {
    pub limit: usize,
    pub inflight: usize,
}

/// Wrapper type for errors to allow signaling early return (`Break`) or error propagation.
#[derive(Debug)]
pub enum Break<E> {
    Break,
    Err(E),
}

/// Extension trait introducing `try_for_each_spawned` to all streams.
pub trait TrySpawnStreamExt: Stream {
    fn try_for_each_spawned<Fut, F, E>(
        self,
        limit: impl Into<Option<usize>>,
        f: F,
    ) -> impl Future<Output = Result<(), E>>
    where
        Fut: Future<Output = Result<(), E>> + Send + 'static,
        F: FnMut(Self::Item) -> Fut,
        E: Send + 'static;

    fn try_for_each_send_spawned<Fut, F, T, E, R>(
        self,
        config: ConcurrencyConfig,
        f: F,
        tx: mpsc::Sender<T>,
        report: R,
    ) -> impl Future<Output = Result<(), Break<E>>>
    where
        Fut: Future<Output = Result<T, Break<E>>> + Send + 'static,
        F: FnMut(Self::Item) -> Fut,
        T: Send + 'static,
        E: Send + 'static,
        R: Fn(ConcurrencyStats);

    fn try_for_each_broadcast_spawned<Fut, F, T, E, R>(
        self,
        config: ConcurrencyConfig,
        f: F,
        txs: Vec<mpsc::Sender<T>>,
        report: R,
    ) -> impl Future<Output = Result<(), Break<E>>>
    where
        Fut: Future<Output = Result<T, Break<E>>> + Send + 'static,
        F: FnMut(Self::Item) -> Fut,
        T: Clone + Send + Sync + 'static,
        E: Send + 'static,
        R: Fn(ConcurrencyStats);
}

trait Sender: Clone + Send + Sync + 'static {
    type Value: Send + 'static;
    fn send(&self, value: Self::Value) -> impl Future<Output = Result<(), ()>> + Send;
    fn fill(&self) -> f64;
}

struct SingleSender<T>(mpsc::Sender<T>);
struct BroadcastSender<T>(Arc<Vec<mpsc::Sender<T>>>);

impl ConcurrencyConfig {
    pub fn fixed(n: usize) -> Self {
        Self { initial: n, min: n, max: n, dead_band_low: 0.6, dead_band_high: 0.85 }
    }

    pub fn adaptive(initial: usize, min: usize, max: usize) -> Self {
        Self { initial, min, max, dead_band_low: 0.6, dead_band_high: 0.85 }
    }

    pub fn with_dead_band(mut self, low: f64, high: f64) -> Self {
        self.dead_band_low = low;
        self.dead_band_high = high;
        self
    }
}

impl<E> From<E> for Break<E> {
    fn from(e: E) -> Self {
        Break::Err(e)
    }
}

impl<S: Stream + Sized + 'static> TrySpawnStreamExt for S {
    async fn try_for_each_spawned<Fut, F, E>(
        self,
        limit: impl Into<Option<usize>>,
        mut f: F,
    ) -> Result<(), E>
    where
        Fut: Future<Output = Result<(), E>> + Send + 'static,
        F: FnMut(Self::Item) -> Fut,
        E: Send + 'static,
    {
        let limit = match limit.into() {
            Some(0) | None => usize::MAX,
            Some(n) => n,
        };

        let mut permits = limit;
        let mut join_set = JoinSet::new();
        let mut draining = false;
        let mut error = None;

        let mut self_ = pin!(self);

        loop {
            while !draining && permits > 0 {
                match poll_fn(|cx| self_.as_mut().poll_next(cx)).now_or_never() {
                    Some(Some(item)) => {
                        permits -= 1;
                        join_set.spawn(f(item));
                    }
                    Some(None) => {
                        draining = true;
                    }
                    None => break,
                }
            }

            tokio::select! {
                biased;

                Some(res) = join_set.join_next() => {
                    match res {
                        Ok(Err(e)) if error.is_none() => {
                            error = Some(e);
                            permits += 1;
                            draining = true;
                        }
                        Ok(_) => permits += 1,
                        Err(e) if e.is_panic() => {
                            panic::resume_unwind(e.into_panic())
                        }
                        Err(e) => {
                            assert!(e.is_cancelled());
                            permits += 1;
                            draining = true;
                        }
                    }
                }

                next = poll_fn(|cx| self_.as_mut().poll_next(cx)),
                    if !draining && permits > 0 => {
                    if let Some(item) = next {
                        permits -= 1;
                        join_set.spawn(f(item));
                    } else {
                        draining = true;
                    }
                }

                else => {
                    if permits == limit && draining {
                        break;
                    }
                }
            }
        }

        if let Some(e) = error { Err(e) } else { Ok(()) }
    }

    async fn try_for_each_send_spawned<Fut, F, T, E, R>(
        self,
        config: ConcurrencyConfig,
        f: F,
        tx: mpsc::Sender<T>,
        report: R,
    ) -> Result<(), Break<E>>
    where
        Fut: Future<Output = Result<T, Break<E>>> + Send + 'static,
        F: FnMut(Self::Item) -> Fut,
        T: Send + 'static,
        E: Send + 'static,
        R: Fn(ConcurrencyStats),
    {
        adaptive_spawn_send(self, config, f, SingleSender(tx), report).await
    }

    async fn try_for_each_broadcast_spawned<Fut, F, T, E, R>(
        self,
        config: ConcurrencyConfig,
        f: F,
        txs: Vec<mpsc::Sender<T>>,
        report: R,
    ) -> Result<(), Break<E>>
    where
        Fut: Future<Output = Result<T, Break<E>>> + Send + 'static,
        F: FnMut(Self::Item) -> Fut,
        T: Clone + Send + Sync + 'static,
        E: Send + 'static,
        R: Fn(ConcurrencyStats),
    {
        adaptive_spawn_send(self, config, f, BroadcastSender(Arc::new(txs)), report).await
    }
}

impl<T> Clone for SingleSender<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Send + 'static> Sender for SingleSender<T> {
    type Value = T;

    async fn send(&self, value: T) -> Result<(), ()> {
        self.0.send(value).await.map_err(|_| ())
    }

    fn fill(&self) -> f64 {
        1.0 - (self.0.capacity() as f64 / self.0.max_capacity() as f64)
    }
}

impl<T> Clone for BroadcastSender<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Clone + Send + Sync + 'static> Sender for BroadcastSender<T> {
    type Value = T;

    async fn send(&self, value: T) -> Result<(), ()> {
        let (last, rest) = self.0.split_last().ok_or(())?;
        let rest_fut = try_join_all(rest.iter().map(|tx| {
            let v = value.clone();
            async move { tx.send(v).await.map_err(|_| ()) }
        }));
        let last_fut = last.send(value).map(|r| r.map_err(|_| ()));
        try_join!(rest_fut, last_fut)?;
        Ok(())
    }

    fn fill(&self) -> f64 {
        self.0
            .iter()
            .map(|tx| 1.0 - (tx.capacity() as f64 / tx.max_capacity() as f64))
            .fold(0.0f64, f64::max)
    }
}

async fn adaptive_spawn_send<S, Fut, F, E, Tx, R>(
    stream: S,
    config: ConcurrencyConfig,
    mut f: F,
    sender: Tx,
    report: R,
) -> Result<(), Break<E>>
where
    S: Stream + 'static,
    Fut: Future<Output = Result<Tx::Value, Break<E>>> + Send + 'static,
    F: FnMut(S::Item) -> Fut,
    E: Send + 'static,
    Tx: Sender,
    R: Fn(ConcurrencyStats),
{
    assert!(config.min >= 1, "ConcurrencyConfig::min must be >= 1");
    let mut limit = config.initial;
    let mut epoch: u64 = 0;
    let mut was_saturated = false;
    let mut tasks: JoinSet<Result<u64, Break<E>>> = JoinSet::new();
    let mut stream_done = false;
    let mut error: Option<Break<E>> = None;

    let mut stream = pin!(stream);

    loop {
        if tasks.is_empty() && (stream_done || error.is_some()) {
            break;
        }

        while tasks.len() < limit && !stream_done && error.is_none() {
            match poll_fn(|cx| stream.as_mut().poll_next(cx)).now_or_never() {
                Some(Some(item)) => {
                    let fut = f(item);
                    let tx = sender.clone();
                    let spawn_epoch = epoch;
                    tasks.spawn(async move {
                        let value = fut.await?;
                        tx.send(value).await.map_err(|_| Break::Break)?;
                        Ok(spawn_epoch)
                    });
                    if tasks.len() >= limit {
                        was_saturated = true;
                    }
                }
                Some(None) => stream_done = true,
                None => break,
            }
        }

        let completed = tokio::select! {
            biased;

            Some(r) = tasks.join_next(), if !tasks.is_empty() => Some(r),

            next = poll_fn(|cx| stream.as_mut().poll_next(cx)),
                if tasks.len() < limit && !stream_done && error.is_none() =>
            {
                if let Some(item) = next {
                    let fut = f(item);
                    let tx = sender.clone();
                    let spawn_epoch = epoch;
                    tasks.spawn(async move {
                        let value = fut.await?;
                        tx.send(value).await.map_err(|_| Break::Break)?;
                        Ok(spawn_epoch)
                    });
                    if tasks.len() >= limit {
                        was_saturated = true;
                    }
                } else {
                    stream_done = true;
                }
                None
            }

            else => {
                if tasks.is_empty() && (stream_done || error.is_some()) {
                    break;
                }
                None
            }
        };

        for join_result in completed
            .into_iter()
            .chain(std::iter::from_fn(|| tasks.join_next().now_or_never().flatten()))
        {
            match join_result {
                Ok(Ok(spawn_epoch)) => {
                    let fill = sender.fill();
                    if fill >= config.dead_band_high && spawn_epoch == epoch {
                        let severity =
                            (fill - config.dead_band_high) / (1.0 - config.dead_band_high);
                        let keep = 0.8 - 0.3 * severity;
                        let new_limit = ((limit as f64) * keep).ceil() as usize;
                        limit = new_limit.min(limit.saturating_sub(1)).max(config.min);
                        limit = limit.clamp(config.min, config.max);
                        epoch += 1;
                        was_saturated = false;
                    } else if fill < config.dead_band_low && was_saturated {
                        let increment = ((limit as f64).log10().ceil() as usize).max(1);
                        limit = (limit + increment).min(config.max);
                        was_saturated = false;
                    }
                }
                Ok(Err(e)) if error.is_none() => error = Some(e),
                Ok(Err(_)) => {}
                Err(e) if e.is_panic() => panic::resume_unwind(e.into_panic()),
                Err(e) => {
                    assert!(e.is_cancelled());
                    stream_done = true;
                }
            }
        }

        report(ConcurrencyStats { limit, inflight: tasks.len() });
    }

    if let Some(e) = error { Err(e) } else { Ok(()) }
}
