// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! GraphQL subscription support via Postgres LISTEN/NOTIFY.
//!
//! Architecture:
//! 1. Postgres triggers (installed by migration 000028) fire NOTIFY on key events.
//! 2. `spawn_pg_listener` opens a dedicated Postgres connection that LISTENs.
//! 3. Notifications are parsed and broadcast via `tokio::sync::broadcast` channels.
//! 4. GraphQL `Subscription` resolvers subscribe to these broadcast channels.
//! 5. `async-graphql-axum` handles WebSocket transport.

use std::sync::Arc;

use async_graphql::*;
use tokio::sync::broadcast;
use tokio_postgres::NoTls;
use tracing::{error, info, warn};

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

/// A new transaction was indexed.
pub struct TransactionEvent {
    pub tx_sequence_number: i64,
    pub kind: String,
    pub sender: String,
    pub epoch: i64,
    pub timestamp_ms: i64,
}

#[Object]
impl TransactionEvent {
    async fn sequence_number(&self) -> crate::api::scalars::BigInt {
        crate::api::scalars::BigInt(self.tx_sequence_number)
    }
    async fn kind(&self) -> &str {
        &self.kind
    }
    async fn sender(&self) -> &str {
        &self.sender
    }
    async fn epoch(&self) -> crate::api::scalars::BigInt {
        crate::api::scalars::BigInt(self.epoch)
    }
    async fn timestamp(&self) -> crate::api::scalars::DateTime {
        crate::api::scalars::DateTime(self.timestamp_ms)
    }
}

/// A target was filled (data submitted successfully).
pub struct TargetFilledEvent {
    pub target_id: String,
    pub epoch: i64,
    pub fill_epoch: Option<i64>,
    pub winning_model_id: String,
    pub submitter: String,
    pub reward_pool: i64,
}

#[Object]
impl TargetFilledEvent {
    async fn target_id(&self) -> &str {
        &self.target_id
    }
    async fn epoch(&self) -> crate::api::scalars::BigInt {
        crate::api::scalars::BigInt(self.epoch)
    }
    async fn fill_epoch(&self) -> Option<crate::api::scalars::BigInt> {
        self.fill_epoch.map(crate::api::scalars::BigInt)
    }
    async fn winning_model_id(&self) -> &str {
        &self.winning_model_id
    }
    async fn submitter(&self) -> &str {
        &self.submitter
    }
    async fn reward_pool(&self) -> crate::api::scalars::BigInt {
        crate::api::scalars::BigInt(self.reward_pool)
    }
}

/// A new checkpoint was indexed.
pub struct CheckpointEvent {
    pub cp_sequence_number: i64,
    pub tx_lo: i64,
    pub epoch: i64,
}

#[Object]
impl CheckpointEvent {
    async fn sequence_number(&self) -> crate::api::scalars::BigInt {
        crate::api::scalars::BigInt(self.cp_sequence_number)
    }
    async fn tx_lo(&self) -> crate::api::scalars::BigInt {
        crate::api::scalars::BigInt(self.tx_lo)
    }
    async fn epoch(&self) -> crate::api::scalars::BigInt {
        crate::api::scalars::BigInt(self.epoch)
    }
}

/// A new epoch started.
pub struct EpochEvent {
    pub epoch: i64,
    pub start_timestamp_ms: i64,
    pub protocol_version: i64,
}

#[Object]
impl EpochEvent {
    async fn epoch(&self) -> crate::api::scalars::BigInt {
        crate::api::scalars::BigInt(self.epoch)
    }
    async fn start_timestamp(&self) -> crate::api::scalars::DateTime {
        crate::api::scalars::DateTime(self.start_timestamp_ms)
    }
    async fn protocol_version(&self) -> crate::api::scalars::BigInt {
        crate::api::scalars::BigInt(self.protocol_version)
    }
}

// ---------------------------------------------------------------------------
// Broadcast channels
// ---------------------------------------------------------------------------

/// Broadcast channels for each subscription type.
#[derive(Clone)]
pub struct SubscriptionChannels {
    pub new_transaction: broadcast::Sender<Arc<TransactionEvent>>,
    pub target_filled: broadcast::Sender<Arc<TargetFilledEvent>>,
    pub new_checkpoint: broadcast::Sender<Arc<CheckpointEvent>>,
    pub new_epoch: broadcast::Sender<Arc<EpochEvent>>,
}

impl SubscriptionChannels {
    pub fn new(capacity: usize) -> Self {
        Self {
            new_transaction: broadcast::channel(capacity).0,
            target_filled: broadcast::channel(capacity).0,
            new_checkpoint: broadcast::channel(capacity).0,
            new_epoch: broadcast::channel(capacity).0,
        }
    }
}

// ---------------------------------------------------------------------------
// Postgres LISTEN/NOTIFY listener
// ---------------------------------------------------------------------------

/// Spawn a background task that connects to Postgres with a dedicated connection,
/// LISTENs on notification channels, and broadcasts parsed events.
///
/// Automatically reconnects on connection errors.
pub fn spawn_pg_listener(
    database_url: String,
    channels: SubscriptionChannels,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match run_listener(&database_url, &channels).await {
                Ok(()) => {
                    warn!("Postgres listener exited cleanly, reconnecting...");
                }
                Err(e) => {
                    error!("Postgres listener error: {e}, reconnecting in 5s...");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        }
    })
}

async fn run_listener(
    database_url: &str,
    channels: &SubscriptionChannels,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (client, mut connection) = tokio_postgres::connect(database_url, NoTls).await?;

    // Forward notifications from the connection to a channel so we can
    // process them without blocking the connection driver.
    let (ntf_tx, mut ntf_rx) =
        tokio::sync::mpsc::unbounded_channel::<tokio_postgres::Notification>();

    // Drive the connection and extract notifications.
    tokio::spawn(async move {
        use futures::StreamExt;
        let mut stream = futures::stream::poll_fn(|cx| connection.poll_message(cx));
        while let Some(msg) = stream.next().await {
            match msg {
                Ok(tokio_postgres::AsyncMessage::Notification(n)) => {
                    let _ = ntf_tx.send(n);
                }
                Ok(_) => {} // Notices, parameter status, etc.
                Err(e) => {
                    error!("Postgres connection error: {e}");
                    break;
                }
            }
        }
    });

    // Subscribe to channels
    for ch in &["new_transaction", "target_filled", "new_checkpoint", "new_epoch"] {
        client.execute(&format!("LISTEN {ch}"), &[]).await?;
    }
    info!("Postgres LISTEN active on: new_transaction, target_filled, new_checkpoint, new_epoch");

    // Process incoming notifications
    while let Some(notification) = ntf_rx.recv().await {
        dispatch_notification(&notification, channels);
    }

    Ok(())
}

fn dispatch_notification(
    notification: &tokio_postgres::Notification,
    channels: &SubscriptionChannels,
) {
    let payload = notification.payload();
    let channel = notification.channel();

    // Parse JSON payload from the trigger function
    let json: serde_json::Value = match serde_json::from_str(payload) {
        Ok(v) => v,
        Err(e) => {
            warn!("Failed to parse notification payload on channel {channel}: {e}");
            return;
        }
    };

    match channel {
        "new_transaction" => {
            let event = Arc::new(TransactionEvent {
                tx_sequence_number: json["tx_sequence_number"].as_i64().unwrap_or(0),
                kind: json["kind"].as_str().unwrap_or("").to_string(),
                sender: format!("0x{}", json["sender"].as_str().unwrap_or("")),
                epoch: json["epoch"].as_i64().unwrap_or(0),
                timestamp_ms: json["timestamp_ms"].as_i64().unwrap_or(0),
            });
            let _ = channels.new_transaction.send(event);
        }
        "target_filled" => {
            let event = Arc::new(TargetFilledEvent {
                target_id: format!("0x{}", json["target_id"].as_str().unwrap_or("")),
                epoch: json["epoch"].as_i64().unwrap_or(0),
                fill_epoch: json["fill_epoch"].as_i64(),
                winning_model_id: format!(
                    "0x{}",
                    json["winning_model_id"].as_str().unwrap_or("")
                ),
                submitter: format!("0x{}", json["submitter"].as_str().unwrap_or("")),
                reward_pool: json["reward_pool"].as_i64().unwrap_or(0),
            });
            let _ = channels.target_filled.send(event);
        }
        "new_checkpoint" => {
            let event = Arc::new(CheckpointEvent {
                cp_sequence_number: json["cp_sequence_number"].as_i64().unwrap_or(0),
                tx_lo: json["tx_lo"].as_i64().unwrap_or(0),
                epoch: json["epoch"].as_i64().unwrap_or(0),
            });
            let _ = channels.new_checkpoint.send(event);
        }
        "new_epoch" => {
            let event = Arc::new(EpochEvent {
                epoch: json["epoch"].as_i64().unwrap_or(0),
                start_timestamp_ms: json["start_timestamp_ms"].as_i64().unwrap_or(0),
                protocol_version: json["protocol_version"].as_i64().unwrap_or(0),
            });
            let _ = channels.new_epoch.send(event);
        }
        other => {
            warn!("Unknown notification channel: {other}");
        }
    }
}

// ---------------------------------------------------------------------------
// GraphQL Subscription type
// ---------------------------------------------------------------------------

pub struct Subscription;

#[async_graphql::Subscription]
impl Subscription {
    /// Stream of new transactions as they are indexed.
    async fn new_transaction(
        &self,
        ctx: &Context<'_>,
    ) -> impl futures::Stream<Item = TransactionEvent> {
        let channels = ctx.data_unchecked::<SubscriptionChannels>();
        let mut rx = channels.new_transaction.subscribe();
        async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        yield TransactionEvent {
                            tx_sequence_number: event.tx_sequence_number,
                            kind: event.kind.clone(),
                            sender: event.sender.clone(),
                            epoch: event.epoch,
                            timestamp_ms: event.timestamp_ms,
                        };
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("newTransaction subscription lagged, skipped {n} events");
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }

    /// Stream of targets that have been filled (data submitted successfully).
    async fn target_filled(
        &self,
        ctx: &Context<'_>,
    ) -> impl futures::Stream<Item = TargetFilledEvent> {
        let channels = ctx.data_unchecked::<SubscriptionChannels>();
        let mut rx = channels.target_filled.subscribe();
        async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        yield TargetFilledEvent {
                            target_id: event.target_id.clone(),
                            epoch: event.epoch,
                            fill_epoch: event.fill_epoch,
                            winning_model_id: event.winning_model_id.clone(),
                            submitter: event.submitter.clone(),
                            reward_pool: event.reward_pool,
                        };
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }

    /// Stream of new checkpoints.
    async fn new_checkpoint(
        &self,
        ctx: &Context<'_>,
    ) -> impl futures::Stream<Item = CheckpointEvent> {
        let channels = ctx.data_unchecked::<SubscriptionChannels>();
        let mut rx = channels.new_checkpoint.subscribe();
        async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        yield CheckpointEvent {
                            cp_sequence_number: event.cp_sequence_number,
                            tx_lo: event.tx_lo,
                            epoch: event.epoch,
                        };
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }

    /// Stream of epoch changes.
    async fn new_epoch(
        &self,
        ctx: &Context<'_>,
    ) -> impl futures::Stream<Item = EpochEvent> {
        let channels = ctx.data_unchecked::<SubscriptionChannels>();
        let mut rx = channels.new_epoch.subscribe();
        async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        yield EpochEvent {
                            epoch: event.epoch,
                            start_timestamp_ms: event.start_timestamp_ms,
                            protocol_version: event.protocol_version,
                        };
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }
}
