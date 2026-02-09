use anyhow::Result;
use backoff::backoff::Backoff;
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::path::Path;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tap::pipe::Pipe;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};
use types::checkpoints::CheckpointSequenceNumber;
use types::full_checkpoint_content::CheckpointData;

use crate::util::create_remote_store_client;

pub struct CheckpointReader {
    /// Local directory path for reading archive files
    path: PathBuf,
    /// Optional remote store URL for fetching from object storage
    remote_store_url: Option<String>,
    remote_store_options: Vec<(String, String)>,
    /// Current checkpoint sequence number to read next
    current_checkpoint_seq: CheckpointSequenceNumber,
    /// Last checkpoint sequence number that was pruned/processed
    last_pruned_watermark: CheckpointSequenceNumber,
    /// Sender for checkpoint archive data
    checkpoint_sender: mpsc::Sender<Arc<CheckpointData>>,
    /// Receiver for processed checkpoint notifications
    processed_receiver: mpsc::Receiver<CheckpointSequenceNumber>,
    /// Remote fetcher task receiver
    remote_fetcher_receiver: Option<mpsc::Receiver<Result<Arc<CheckpointData>>>>,
    /// Exit signal receiver
    exit_receiver: oneshot::Receiver<()>,
    /// Reader options
    options: ReaderOptions,
}

#[derive(Clone)]
pub struct ReaderOptions {
    /// Interval in milliseconds to check for new files
    pub tick_interval_ms: u64,
    /// Timeout in seconds for remote fetch operations
    pub timeout_secs: u64,
    /// Number of concurrent remote fetches
    pub batch_size: usize,
    /// Upper limit of checkpoint sequence number to process
    pub upper_limit: Option<CheckpointSequenceNumber>,
    /// Whether to delete processed archive files
    pub gc_archive_files: bool,
}

impl Default for ReaderOptions {
    fn default() -> Self {
        Self {
            tick_interval_ms: 100,
            timeout_secs: 5,
            batch_size: 10,
            upper_limit: None,
            gc_archive_files: true,
        }
    }
}

impl CheckpointReader {
    /// Initialize the checkpoint reader
    pub fn initialize(
        path: PathBuf,
        starting_checkpoint: CheckpointSequenceNumber,
        remote_store_url: Option<String>,
        remote_store_options: Vec<(String, String)>,
        options: ReaderOptions,
    ) -> (
        Self,
        mpsc::Receiver<Arc<CheckpointData>>,
        mpsc::Sender<CheckpointSequenceNumber>,
        oneshot::Sender<()>,
    ) {
        let (checkpoint_sender, checkpoint_recv) = mpsc::channel(100);
        let (processed_sender, processed_receiver) = mpsc::channel(100);
        let (exit_sender, exit_receiver) = oneshot::channel();

        let reader = Self {
            path,
            remote_store_url,
            remote_store_options,
            current_checkpoint_seq: starting_checkpoint,
            last_pruned_watermark: starting_checkpoint,
            checkpoint_sender,
            processed_receiver,
            remote_fetcher_receiver: None,
            exit_receiver,
            options,
        };

        (reader, checkpoint_recv, processed_sender, exit_sender)
    }

    /// Read local archive files
    async fn read_local_files(&self) -> Result<Vec<Arc<CheckpointData>>> {
        let mut archives = vec![];
        for offset in 0..100 {
            // Max 100 checkpoints at a time
            let checkpoint_seq = self.current_checkpoint_seq + offset as u64;
            if self.exceeds_limit(checkpoint_seq) {
                break;
            }

            let file_path = self.path.join(format!("{}.chk", checkpoint_seq));
            match fs::read(file_path) {
                Ok(bytes) => {
                    let archive_data: CheckpointData = bcs::from_bytes(&bytes)?;
                    archives.push(Arc::new(archive_data));
                }
                Err(err) => match err.kind() {
                    std::io::ErrorKind::NotFound => break,
                    _ => return Err(err.into()),
                },
            }
        }
        Ok(archives)
    }

    fn exceeds_limit(&self, checkpoint_seq: CheckpointSequenceNumber) -> bool {
        if let Some(upper_limit) = self.options.upper_limit {
            checkpoint_seq > upper_limit
        } else {
            false
        }
    }

    /// Fetch from object store
    async fn fetch_from_object_store(
        store: &dyn ObjectStore,
        checkpoint_seq: CheckpointSequenceNumber,
    ) -> Result<Arc<CheckpointData>> {
        let path = Path::from(format!("{}.chk", checkpoint_seq));
        let response = store.get(&path).await?;
        let bytes = response.bytes().await?;
        let archive_data: CheckpointData = bcs::from_bytes(&bytes)?;
        Ok(Arc::new(archive_data))
    }

    /// Start remote fetcher task
    fn start_remote_fetcher(&mut self) -> mpsc::Receiver<Result<Arc<CheckpointData>>> {
        let batch_size = self.options.batch_size;
        let start_checkpoint = self.current_checkpoint_seq;
        let (sender, receiver) = mpsc::channel(batch_size);

        let url = self.remote_store_url.clone().expect("remote store url must be set");
        let store = create_remote_store_client(
            url,
            self.remote_store_options.clone(),
            self.options.timeout_secs,
        )
        .expect("failed to create remote store client");

        tokio::spawn(async move {
            let mut checkpoint_stream = (start_checkpoint..u64::MAX)
                .map(|checkpoint_seq| Self::fetch_from_object_store(&*store, checkpoint_seq))
                .pipe(futures::stream::iter)
                .buffered(batch_size);

            while let Some(archive) = checkpoint_stream.next().await {
                if sender.send(archive).await.is_err() {
                    info!("Remote reader dropped");
                    break;
                }
            }
        });

        receiver
    }

    /// Fetch from remote store
    fn remote_fetch(&mut self) -> Vec<Arc<CheckpointData>> {
        let mut archives = vec![];
        if self.remote_fetcher_receiver.is_none() {
            self.remote_fetcher_receiver = Some(self.start_remote_fetcher());
        }

        while !self.exceeds_limit(self.current_checkpoint_seq + archives.len() as u64) {
            match self.remote_fetcher_receiver.as_mut().unwrap().try_recv() {
                Ok(Ok(archive)) => archives.push(archive),
                Ok(Err(err)) => {
                    error!("Remote reader error: {:?}", err);
                    self.remote_fetcher_receiver = None;
                    break;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    error!("Remote reader channel disconnected");
                    self.remote_fetcher_receiver = None;
                    break;
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
            }
        }
        archives
    }

    /// Sync archives
    async fn sync(&mut self) -> Result<()> {
        let mut archives = self.read_local_files().await?;

        let mut read_source = "local";
        if self.remote_store_url.is_some() && archives.is_empty() {
            archives = self.remote_fetch();
            read_source = "remote";
        } else {
            // Cancel remote fetcher if local has data
            self.remote_fetcher_receiver = None;
        }

        info!(
            "Read from {}. Current checkpoint: {}, new archives: {}",
            read_source,
            self.current_checkpoint_seq,
            archives.len()
        );

        for archive in archives {
            self.checkpoint_sender.send(archive).await?;
            self.current_checkpoint_seq += 1;
        }

        Ok(())
    }

    /// Clean up processed files
    fn gc_processed_files(&mut self, watermark: CheckpointSequenceNumber) -> Result<()> {
        self.last_pruned_watermark = watermark;
        if !self.options.gc_archive_files {
            return Ok(());
        }

        info!("Cleaning processed files, watermark: {}", watermark);
        for entry in fs::read_dir(&self.path)? {
            let entry = entry?;
            let file_name = entry.file_name();
            if let Some(checkpoint_seq) = Self::parse_checkpoint_seq(&file_name) {
                if checkpoint_seq < watermark {
                    fs::remove_file(entry.path())?;
                }
            }
        }
        Ok(())
    }

    fn parse_checkpoint_seq(file_name: &std::ffi::OsStr) -> Option<CheckpointSequenceNumber> {
        file_name.to_str().and_then(|s| s.strip_suffix(".chk")).and_then(|s| s.parse().ok())
    }

    /// Main run loop
    pub async fn run(mut self) -> Result<()> {
        std::fs::create_dir_all(&self.path)?;

        self.gc_processed_files(self.last_pruned_watermark)?;

        loop {
            tokio::select! {
                _ = &mut self.exit_receiver => break,
                Some(checkpoint_seq) = self.processed_receiver.recv() => {
                    self.gc_processed_files(checkpoint_seq)?;
                }
                _ = tokio::time::sleep(Duration::from_millis(self.options.tick_interval_ms)) => {
                    self.sync().await?;
                }
            }
        }
        Ok(())
    }
}
