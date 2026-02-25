// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::Worker;
use crate::reader::{CheckpointReader, ReaderOptions};
use anyhow::Result;
use futures::Future;
use once_cell::sync::Lazy;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tracing::info;
use types::checkpoints::CheckpointSequenceNumber;
use types::full_checkpoint_content::CheckpointData;

pub static MAX_CHECKPOINTS_IN_PROGRESS: Lazy<usize> = Lazy::new(|| {
    std::env::var("MAX_CHECKPOINTS_IN_PROGRESS").ok().and_then(|s| s.parse().ok()).unwrap_or(100)
});

/// Progress tracking for executor
#[derive(Debug, Clone)]
pub struct ExecutorProgress {
    pub last_processed_checkpoint: CheckpointSequenceNumber,
    pub total_processed: u64,
}
/// Worker pool for processing checkpoints
pub struct WorkerPool<W: Worker> {
    worker: Arc<W>,
    task_name: String,
    concurrency: usize,
}

impl<W: Worker + 'static> WorkerPool<W> {
    pub fn new(worker: W, task_name: String, concurrency: usize) -> Self {
        Self { worker: Arc::new(worker), task_name, concurrency }
    }

    pub async fn run(
        self,
        watermark: CheckpointSequenceNumber,
        mut archive_receiver: mpsc::Receiver<Arc<CheckpointData>>,
        executor_progress_sender: mpsc::Sender<(String, CheckpointSequenceNumber)>,
    ) {
        info!(
            "Starting worker pool {} with concurrency {}. Starting at checkpoint {}",
            self.task_name, self.concurrency, watermark
        );

        // Change: Remove Result from the progress channel type
        let (progress_sender, mut progress_receiver) =
            mpsc::channel::<(usize, CheckpointSequenceNumber)>(*MAX_CHECKPOINTS_IN_PROGRESS);
        let mut workers = vec![];
        let mut idle: BTreeSet<_> = (0..self.concurrency).collect();
        let mut pending_archives = std::collections::VecDeque::new();
        let mut current_checkpoint = watermark;

        // Spawn child workers
        for worker_id in 0..self.concurrency {
            let (worker_sender, mut worker_recv) = mpsc::channel::<(
                CheckpointSequenceNumber,
                Arc<CheckpointData>,
            )>(*MAX_CHECKPOINTS_IN_PROGRESS);
            let (term_sender, mut term_receiver) = oneshot::channel::<()>();

            let worker = self.worker.clone();
            let progress_sender = progress_sender.clone();
            let task_name = self.task_name.clone();

            let handle = tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = &mut term_receiver => break,
                        Some((checkpoint_seq_number, archive_data)) = worker_recv.recv() => {
                            info!("Worker {} processing checkpoint {} for {}", worker_id, checkpoint_seq_number, task_name);
                            let start = std::time::Instant::now();

                            // Process with retry
                            let backoff = backoff::ExponentialBackoff::default();
                            let result = backoff::future::retry(backoff, || async {
                                worker
                                    .process_checkpoint(&archive_data)
                                    .await
                                    .map_err(|err| {
                                        info!("Transient error for checkpoint {}: {:?}", checkpoint_seq_number, err);
                                        backoff::Error::transient(err)
                                    })
                            })
                            .await;

                            match result {
                                Ok(_res) => {
                                    info!("Worker {} finished checkpoint {} in {:?}", worker_id, checkpoint_seq_number, start.elapsed());
                                }
                                Err(e) => {
                                    tracing::error!("Worker {} failed to process checkpoint {}: {}", worker_id, checkpoint_seq_number, e);
                                    // Continue anyway - we still mark this checkpoint as processed
                                }
                            }

                            // Always send progress, regardless of success/failure
                            // Change: Just send worker_id and checkpoint_seq_number, no result
                            if progress_sender.send((worker_id, checkpoint_seq_number)).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            });

            workers.push((worker_sender, term_sender, handle));
        }

        // Main worker pool loop
        loop {
            tokio::select! {
                // Change: Remove _result from the pattern match
                Some((worker_id, checkpoint_seq_number)) = progress_receiver.recv() => {
                    idle.insert(worker_id);

                    // Send progress to executor
                    if executor_progress_sender.send((self.task_name.clone(), checkpoint_seq_number)).await.is_err() {
                        break;
                    }

                    // Assign pending work to idle workers
                    while !pending_archives.is_empty() && !idle.is_empty() {
                        let archive = pending_archives.pop_front().unwrap();
                        let worker_id = idle.pop_first().unwrap();
                        if workers[worker_id].0.send((current_checkpoint, archive)).await.is_err() {
                            break;
                        }
                        current_checkpoint += 1;
                    }
                }

                maybe_archive = archive_receiver.recv() => {
                    if maybe_archive.is_none() {
                        break;
                    }
                    let archive = maybe_archive.unwrap();

                    if idle.is_empty() {
                        pending_archives.push_back(archive);
                    } else {
                        let worker_id = idle.pop_first().unwrap();
                        if workers[worker_id].0.send((current_checkpoint, archive)).await.is_err() {
                            break;
                        }
                        current_checkpoint += 1;
                    }
                }
            }
        }

        // Cleanup
        drop(workers);
    }
}

/// Main executor for indexing
pub struct IndexerExecutor {
    pools: Vec<Pin<Box<dyn Future<Output = ()> + Send>>>,
    pool_senders: Vec<mpsc::Sender<Arc<CheckpointData>>>,
    progress_sender: mpsc::Sender<(String, CheckpointSequenceNumber)>,
    progress_receiver: mpsc::Receiver<(String, CheckpointSequenceNumber)>,
    current_checkpoint: CheckpointSequenceNumber,
}

impl IndexerExecutor {
    pub fn new(starting_checkpoint: CheckpointSequenceNumber, num_workers: usize) -> Self {
        let (progress_sender, progress_receiver) =
            mpsc::channel(MAX_CHECKPOINTS_IN_PROGRESS.saturating_mul(num_workers));

        Self {
            pools: vec![],
            pool_senders: vec![],
            progress_sender,
            progress_receiver,
            current_checkpoint: starting_checkpoint,
        }
    }

    /// Register a worker pool
    pub async fn register<W: Worker + 'static>(&mut self, pool: WorkerPool<W>) -> Result<()> {
        let (sender, receiver) = mpsc::channel(*MAX_CHECKPOINTS_IN_PROGRESS);

        self.pools.push(Box::pin(pool.run(
            self.current_checkpoint,
            receiver,
            self.progress_sender.clone(),
        )));
        self.pool_senders.push(sender);
        Ok(())
    }

    /// Main executor loop (same as before)
    pub async fn run(
        mut self,
        path: PathBuf,
        remote_store_url: Option<String>,
        remote_store_options: Vec<(String, String)>,
        reader_options: ReaderOptions,
        mut exit_receiver: oneshot::Receiver<()>,
    ) -> Result<ExecutorProgress> {
        let (reader, mut checkpoint_recv, gc_sender, _exit_sender) = CheckpointReader::initialize(
            path,
            self.current_checkpoint,
            remote_store_url,
            remote_store_options,
            reader_options.clone(),
        );

        tokio::spawn(reader.run());

        for pool in std::mem::take(&mut self.pools) {
            tokio::spawn(pool);
        }

        let mut total_processed = 0u64;
        let mut last_checkpoint = self.current_checkpoint;

        loop {
            tokio::select! {
                _ = &mut exit_receiver => break,

                Some((task_name, checkpoint_sequence_number)) = self.progress_receiver.recv() => {
                    info!("Task {} processed checkpoint {}", task_name, checkpoint_sequence_number);
                    gc_sender.send(checkpoint_sequence_number).await?;
                    last_checkpoint = last_checkpoint.max(checkpoint_sequence_number);
                    total_processed += 1;

                    if let Some(limit) = reader_options.upper_limit {
                        if checkpoint_sequence_number >= limit && self.pool_senders.len() == 1 {
                            break;
                        }
                    }
                }

                Some(archive_data) = checkpoint_recv.recv() => {
                    for sender in &self.pool_senders {
                        sender.send(archive_data.clone()).await?;
                    }
                }
            }
        }

        Ok(ExecutorProgress { last_processed_checkpoint: last_checkpoint, total_processed })
    }
}

/// Setup a single workflow with a worker
pub async fn setup_single_workflow_with_options<W: Worker + 'static>(
    worker: W,
    remote_store_url: String,
    remote_store_options: Vec<(String, String)>,
    initial_checkpoint_number: CheckpointSequenceNumber,
    concurrency: usize,
    reader_options: Option<ReaderOptions>,
) -> Result<(impl Future<Output = Result<ExecutorProgress>>, oneshot::Sender<()>)> {
    let (exit_sender, exit_receiver) = oneshot::channel();
    let mut executor = IndexerExecutor::new(initial_checkpoint_number, 1);
    let worker_pool = WorkerPool::new(worker, "workflow".to_string(), concurrency);
    executor.register(worker_pool).await?;

    Ok((
        executor.run(
            tempfile::tempdir()?.into_path(),
            Some(remote_store_url),
            remote_store_options,
            reader_options.unwrap_or_default(),
            exit_receiver,
        ),
        exit_sender,
    ))
}
