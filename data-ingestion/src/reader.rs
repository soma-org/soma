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
use types::checkpoint::CommitArchiveData;
use types::consensus::commit::CommitIndex;

use crate::util::create_remote_store_client;

pub struct CommitReader {
    /// Local directory path for reading archive files
    path: PathBuf,
    /// Optional remote store URL for fetching from object storage
    remote_store_url: Option<String>,
    remote_store_options: Vec<(String, String)>,
    /// Current commit index to read next
    current_commit_index: CommitIndex,
    /// Last commit index that was pruned/processed
    last_pruned_watermark: CommitIndex,
    /// Sender for commit archive data
    commit_sender: mpsc::Sender<Arc<CommitArchiveData>>,
    /// Receiver for processed commit notifications
    processed_receiver: mpsc::Receiver<CommitIndex>,
    /// Remote fetcher task receiver
    remote_fetcher_receiver: Option<mpsc::Receiver<Result<Arc<CommitArchiveData>>>>,
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
    /// Upper limit of commit index to process
    pub upper_limit: Option<CommitIndex>,
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

impl CommitReader {
    /// Initialize the commit reader
    pub fn initialize(
        path: PathBuf,
        starting_commit_index: CommitIndex,
        remote_store_url: Option<String>,
        remote_store_options: Vec<(String, String)>,
        options: ReaderOptions,
    ) -> (
        Self,
        mpsc::Receiver<Arc<CommitArchiveData>>,
        mpsc::Sender<CommitIndex>,
        oneshot::Sender<()>,
    ) {
        let (commit_sender, commit_recv) = mpsc::channel(100);
        let (processed_sender, processed_receiver) = mpsc::channel(100);
        let (exit_sender, exit_receiver) = oneshot::channel();

        let reader = Self {
            path,
            remote_store_url,
            remote_store_options,
            current_commit_index: starting_commit_index,
            last_pruned_watermark: starting_commit_index,
            commit_sender,
            processed_receiver,
            remote_fetcher_receiver: None,
            exit_receiver,
            options,
        };

        (reader, commit_recv, processed_sender, exit_sender)
    }

    /// Read local archive files
    async fn read_local_files(&self) -> Result<Vec<Arc<CommitArchiveData>>> {
        let mut archives = vec![];
        for offset in 0..100 {
            // Max 100 commits at a time
            let commit_index = self.current_commit_index + offset as u32;
            if self.exceeds_limit(commit_index) {
                break;
            }

            let file_path = self.path.join(format!("{}.dat", commit_index));
            match fs::read(file_path) {
                Ok(bytes) => {
                    let archive_data: CommitArchiveData = bcs::from_bytes(&bytes)?;
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

    fn exceeds_limit(&self, commit_index: CommitIndex) -> bool {
        if let Some(upper_limit) = self.options.upper_limit {
            commit_index > upper_limit
        } else {
            false
        }
    }

    /// Fetch from object store
    async fn fetch_from_object_store(
        store: &dyn ObjectStore,
        commit_index: CommitIndex,
    ) -> Result<Arc<CommitArchiveData>> {
        let path = Path::from(format!("{}.dat", commit_index));
        let response = store.get(&path).await?;
        let bytes = response.bytes().await?;
        let archive_data: CommitArchiveData = bcs::from_bytes(&bytes)?;
        Ok(Arc::new(archive_data))
    }

    /// Start remote fetcher task
    fn start_remote_fetcher(&mut self) -> mpsc::Receiver<Result<Arc<CommitArchiveData>>> {
        let batch_size = self.options.batch_size;
        let start_commit = self.current_commit_index;
        let (sender, receiver) = mpsc::channel(batch_size);

        let url = self
            .remote_store_url
            .clone()
            .expect("remote store url must be set");
        let store = create_remote_store_client(
            url,
            self.remote_store_options.clone(),
            self.options.timeout_secs,
        )
        .expect("failed to create remote store client");

        tokio::spawn(async move {
            let mut commit_stream = (start_commit..u32::MAX)
                .map(|commit_index| Self::fetch_from_object_store(&*store, commit_index))
                .pipe(futures::stream::iter)
                .buffered(batch_size);

            while let Some(archive) = commit_stream.next().await {
                if sender.send(archive).await.is_err() {
                    info!("Remote reader dropped");
                    break;
                }
            }
        });

        receiver
    }

    /// Fetch from remote store
    fn remote_fetch(&mut self) -> Vec<Arc<CommitArchiveData>> {
        let mut archives = vec![];
        if self.remote_fetcher_receiver.is_none() {
            self.remote_fetcher_receiver = Some(self.start_remote_fetcher());
        }

        while !self.exceeds_limit(self.current_commit_index + archives.len() as u32) {
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
            "Read from {}. Current commit: {}, new archives: {}",
            read_source,
            self.current_commit_index,
            archives.len()
        );

        for archive in archives {
            self.commit_sender.send(archive).await?;
            self.current_commit_index += 1;
        }

        Ok(())
    }

    /// Clean up processed files
    fn gc_processed_files(&mut self, watermark: CommitIndex) -> Result<()> {
        self.last_pruned_watermark = watermark;
        if !self.options.gc_archive_files {
            return Ok(());
        }

        info!("Cleaning processed files, watermark: {}", watermark);
        for entry in fs::read_dir(&self.path)? {
            let entry = entry?;
            let file_name = entry.file_name();
            if let Some(commit_index) = Self::parse_commit_index(&file_name) {
                if commit_index < watermark {
                    fs::remove_file(entry.path())?;
                }
            }
        }
        Ok(())
    }

    fn parse_commit_index(file_name: &std::ffi::OsStr) -> Option<CommitIndex> {
        file_name
            .to_str()
            .and_then(|s| s.strip_suffix(".dat"))
            .and_then(|s| s.parse().ok())
    }

    /// Main run loop
    pub async fn run(mut self) -> Result<()> {
        std::fs::create_dir_all(&self.path)?;

        self.gc_processed_files(self.last_pruned_watermark)?;

        loop {
            tokio::select! {
                _ = &mut self.exit_receiver => break,
                Some(commit_index) = self.processed_receiver.recv() => {
                    self.gc_processed_files(commit_index)?;
                }
                _ = tokio::time::sleep(Duration::from_millis(self.options.tick_interval_ms)) => {
                    self.sync().await?;
                }
            }
        }
        Ok(())
    }
}
