// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::RwLock;
use std::task::Context;
use std::task::Poll;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use bytes::Bytes;
use gcp_auth::Token;
use gcp_auth::TokenProvider;
use http::HeaderValue;
use http::Request;
use http::Response;
use prometheus::Registry;
use tonic::body::Body;
use tonic::codegen::Service;
use tonic::transport::Certificate;
use tonic::transport::Channel;
use tonic::transport::ClientTlsConfig;

use crate::Watermark;
use crate::bigtable::metrics::KvMetrics;
use crate::bigtable::proto::bigtable::v2::MutateRowsRequest;
use crate::bigtable::proto::bigtable::v2::ReadRowsRequest;
use crate::bigtable::proto::bigtable::v2::RequestStats;
use crate::bigtable::proto::bigtable::v2::RowFilter;
use crate::bigtable::proto::bigtable::v2::RowRange;
use crate::bigtable::proto::bigtable::v2::RowSet;
use crate::bigtable::proto::bigtable::v2::bigtable_client::BigtableClient as BigtableInternalClient;
use crate::bigtable::proto::bigtable::v2::mutate_rows_request::Entry;
use crate::bigtable::proto::bigtable::v2::read_rows_response::cell_chunk::RowStatus;
use crate::bigtable::proto::bigtable::v2::request_stats::StatsView;
use crate::bigtable::proto::bigtable::v2::row_filter::Chain;
use crate::bigtable::proto::bigtable::v2::row_filter::Filter;
use crate::bigtable::proto::bigtable::v2::row_range::EndKey;
use crate::bigtable::proto::bigtable::v2::row_range::StartKey;
use crate::tables;

const DEFAULT_MAX_DECODING_MESSAGE_SIZE: usize = 32 * 1024 * 1024;
const DEFAULT_CHANNEL_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_CHANNEL_POOL_SIZE: usize = 10;

/// Error returned when a batch write has per-entry failures.
#[derive(Debug)]
pub struct PartialWriteError {
    pub failed_keys: Vec<MutationError>,
}

#[derive(Debug)]
pub struct MutationError {
    pub key: Bytes,
    pub code: i32,
    pub message: String,
}

#[derive(Clone)]
pub struct BigTableClient {
    table_prefix: String,
    client: BigtableInternalClient<AuthChannel>,
    client_name: String,
    metrics: Option<Arc<KvMetrics>>,
    app_profile_id: Option<String>,
}

#[derive(Clone)]
struct AuthChannel {
    channel: Channel,
    policy: String,
    token_provider: Option<Arc<dyn TokenProvider>>,
    token: Arc<RwLock<Option<Arc<Token>>>>,
}

impl BigTableClient {
    pub async fn new_local(host: String, instance_id: String) -> Result<Self> {
        Self::new_for_host(host, instance_id, "local")
    }

    pub(crate) fn new_for_host(
        host: String,
        instance_id: String,
        client_name: &str,
    ) -> Result<Self> {
        let auth_channel = AuthChannel {
            channel: Channel::from_shared(format!("http://{host}"))?.connect_lazy(),
            policy: "https://www.googleapis.com/auth/bigtable.data".to_string(),
            token_provider: None,
            token: Arc::new(RwLock::new(None)),
        };
        Ok(Self {
            table_prefix: format!("projects/emulator/instances/{}/tables/", instance_id),
            client: BigtableInternalClient::new(auth_channel),
            client_name: client_name.to_string(),
            metrics: None,
            app_profile_id: None,
        })
    }

    pub async fn new_remote(
        instance_id: String,
        project_id: Option<String>,
        is_read_only: bool,
        timeout: Option<Duration>,
        max_decoding_message_size: Option<usize>,
        client_name: String,
        registry: Option<&Registry>,
        app_profile_id: Option<String>,
        channel_pool_size: Option<usize>,
    ) -> Result<Self> {
        Self::new_remote_with_credentials(
            instance_id,
            project_id,
            is_read_only,
            timeout,
            max_decoding_message_size,
            client_name,
            registry,
            app_profile_id,
            channel_pool_size,
            None,
        )
        .await
    }

    pub async fn new_remote_with_credentials(
        instance_id: String,
        project_id: Option<String>,
        is_read_only: bool,
        timeout: Option<Duration>,
        max_decoding_message_size: Option<usize>,
        client_name: String,
        registry: Option<&Registry>,
        app_profile_id: Option<String>,
        channel_pool_size: Option<usize>,
        credentials_path: Option<String>,
    ) -> Result<Self> {
        let pool_size = channel_pool_size
            .unwrap_or(DEFAULT_CHANNEL_POOL_SIZE)
            .max(1);
        let policy = if is_read_only {
            "https://www.googleapis.com/auth/bigtable.data.readonly"
        } else {
            "https://www.googleapis.com/auth/bigtable.data"
        };
        let token_provider: Arc<dyn TokenProvider> = match credentials_path {
            Some(path) => Arc::new(gcp_auth::CustomServiceAccount::from_file(&path)?),
            None => gcp_auth::provider().await?,
        };
        let tls_config = ClientTlsConfig::new()
            .ca_certificate(Certificate::from_pem(include_bytes!("./proto/google.pem")))
            .domain_name("bigtable.googleapis.com");
        let mut endpoint = Channel::from_static("https://bigtable.googleapis.com")
            .http2_keep_alive_interval(Duration::from_secs(60))
            .keep_alive_while_idle(true)
            .tls_config(tls_config)?;
        endpoint = endpoint.timeout(timeout.unwrap_or(DEFAULT_CHANNEL_TIMEOUT));
        let project_id = match project_id {
            Some(p) => p,
            None => token_provider.project_id().await?.to_string(),
        };
        let table_prefix = format!("projects/{}/instances/{}/tables/", project_id, instance_id);
        let channel = if pool_size > 1 {
            let (channel, tx) = Channel::balance_channel::<usize>(64);
            for i in 0..pool_size {
                tx.try_send(tonic::transport::channel::Change::Insert(
                    i,
                    endpoint.clone(),
                ))
                .expect("channel balancer dropped");
            }
            channel
        } else {
            endpoint.connect_lazy()
        };
        let auth_channel = AuthChannel {
            channel,
            policy: policy.to_string(),
            token_provider: Some(token_provider),
            token: Arc::new(RwLock::new(None)),
        };
        let client = BigtableInternalClient::new(auth_channel).max_decoding_message_size(
            max_decoding_message_size.unwrap_or(DEFAULT_MAX_DECODING_MESSAGE_SIZE),
        );
        Ok(Self {
            table_prefix,
            client,
            client_name,
            metrics: registry.map(KvMetrics::new),
            app_profile_id,
        })
    }

    /// Get the pipeline watermark from the watermarks table.
    pub async fn get_pipeline_watermark(&mut self, pipeline: &str) -> Result<Option<Watermark>> {
        let pipeline_key = tables::watermarks::encode_key(pipeline);

        let rows = self
            .multi_get(
                tables::watermark_alt_legacy::NAME,
                vec![pipeline_key.clone()],
                None,
            )
            .await?;

        for (key, row) in rows {
            if key.as_ref() == pipeline_key.as_slice() {
                return Ok(Some(tables::watermarks::decode(&row)?));
            }
        }

        Ok(None)
    }

    /// Set the pipeline watermark in the watermarks table.
    pub async fn set_pipeline_watermark(
        &mut self,
        pipeline: &str,
        watermark: &Watermark,
    ) -> Result<()> {
        let entry = tables::make_entry(
            tables::watermarks::encode_key(pipeline),
            tables::watermarks::encode(watermark)?,
            Some(watermark.timestamp_ms_hi_inclusive),
        );
        self.write_entries(tables::watermarks::NAME, [entry]).await
    }

    /// Write pre-built entries to BigTable.
    pub async fn write_entries(
        &mut self,
        table: &str,
        entries: impl IntoIterator<Item = Entry>,
    ) -> Result<()> {
        let entries: Vec<Entry> = entries.into_iter().collect();
        if entries.is_empty() {
            return Ok(());
        }

        let row_keys: Vec<Bytes> = entries.iter().map(|e| e.row_key.clone()).collect();

        let mut request = MutateRowsRequest {
            table_name: format!("{}{}", self.table_prefix, table),
            entries,
            ..MutateRowsRequest::default()
        };
        if let Some(ref app_profile_id) = self.app_profile_id {
            request.app_profile_id = app_profile_id.clone();
        }
        let mut response = self.client.clone().mutate_rows(request).await?.into_inner();
        let mut failed_keys: Vec<MutationError> = Vec::new();

        while let Some(part) = response.message().await? {
            for entry in part.entries {
                if let Some(ref status) = entry.status {
                    if status.code != 0 {
                        if let Some(key) = row_keys.get(entry.index as usize) {
                            failed_keys.push(MutationError {
                                key: key.clone(),
                                code: status.code,
                                message: status.message.clone(),
                            });
                        }
                    }
                }
            }
        }

        if !failed_keys.is_empty() {
            return Err(PartialWriteError { failed_keys }.into());
        }

        Ok(())
    }

    pub async fn read_rows(
        &mut self,
        mut request: ReadRowsRequest,
        table_name: &str,
    ) -> Result<Vec<(Bytes, Vec<(Bytes, Bytes)>)>> {
        #[derive(Default)]
        enum CellValue {
            #[default]
            Empty,
            Single(Bytes),
            Multi(Vec<u8>),
        }

        impl CellValue {
            fn extend(&mut self, data: Bytes) {
                *self = match std::mem::take(self) {
                    CellValue::Empty => CellValue::Single(data),
                    CellValue::Single(existing) => {
                        let mut vec = existing.to_vec();
                        vec.extend_from_slice(&data);
                        CellValue::Multi(vec)
                    }
                    CellValue::Multi(mut vec) => {
                        vec.extend_from_slice(&data);
                        CellValue::Multi(vec)
                    }
                };
            }

            fn replace(&mut self, data: Bytes) {
                *self = CellValue::Single(data);
            }

            fn into_bytes(self) -> Bytes {
                match self {
                    CellValue::Empty => Bytes::new(),
                    CellValue::Single(b) => b,
                    CellValue::Multi(v) => Bytes::from(v),
                }
            }
        }

        if let Some(ref app_profile_id) = self.app_profile_id {
            request.app_profile_id = app_profile_id.clone();
        }
        let mut result = vec![];
        let mut response = self.client.clone().read_rows(request).await?.into_inner();

        let mut row_key: Option<Bytes> = None;
        let mut row = vec![];
        let mut cell_value = CellValue::Empty;
        let mut cell_name: Option<Bytes> = None;
        let mut timestamp = 0;

        while let Some(message) = response.message().await? {
            self.report_bt_stats(&message.request_stats, table_name);
            for chunk in message.chunks.into_iter() {
                if !chunk.row_key.is_empty() {
                    row_key = Some(chunk.row_key);
                }
                match chunk.qualifier {
                    Some(qualifier) => {
                        if let Some(name) = cell_name.take() {
                            row.push((name, cell_value.into_bytes()));
                            cell_value = CellValue::Empty;
                        }
                        cell_name = Some(Bytes::from(qualifier));
                        timestamp = chunk.timestamp_micros;
                        cell_value.extend(chunk.value);
                    }
                    None => {
                        if chunk.timestamp_micros == 0 {
                            cell_value.extend(chunk.value);
                        } else if chunk.timestamp_micros >= timestamp {
                            timestamp = chunk.timestamp_micros;
                            cell_value.replace(chunk.value);
                        }
                    }
                }
                if chunk.row_status.is_some() {
                    if let Some(RowStatus::CommitRow(_)) = chunk.row_status {
                        if let Some(name) = cell_name.take() {
                            row.push((name, cell_value.into_bytes()));
                        }
                        if let Some(key) = row_key.take() {
                            result.push((key, row));
                        }
                    }
                    row_key = None;
                    row = vec![];
                    cell_value = CellValue::Empty;
                    cell_name = None;
                }
            }
        }
        Ok(result)
    }

    pub async fn multi_get(
        &mut self,
        table_name: &str,
        keys: Vec<Vec<u8>>,
        filter: Option<RowFilter>,
    ) -> Result<Vec<(Bytes, Vec<(Bytes, Bytes)>)>> {
        let start_time = Instant::now();
        let num_keys_requested = keys.len();
        let result = self.multi_get_internal(table_name, keys, filter).await;
        let elapsed_ms = start_time.elapsed().as_millis() as f64;

        let Some(metrics) = &self.metrics else {
            return result;
        };

        let labels = [&self.client_name, table_name];
        let Ok(rows) = &result else {
            metrics.kv_get_errors.with_label_values(&labels).inc();
            return result;
        };

        metrics
            .kv_get_batch_size
            .with_label_values(&labels)
            .observe(num_keys_requested as f64);

        if num_keys_requested > rows.len() {
            metrics
                .kv_get_not_found
                .with_label_values(&labels)
                .inc_by((num_keys_requested - rows.len()) as u64);
        }

        metrics
            .kv_get_success
            .with_label_values(&labels)
            .inc_by(rows.len() as u64);

        metrics
            .kv_get_latency_ms
            .with_label_values(&labels)
            .observe(elapsed_ms);

        if num_keys_requested > 0 {
            metrics
                .kv_get_latency_ms_per_key
                .with_label_values(&labels)
                .observe(elapsed_ms / num_keys_requested as f64);
        }

        result
    }

    fn report_bt_stats(&self, request_stats: &Option<RequestStats>, table_name: &str) {
        let Some(metrics) = &self.metrics else {
            return;
        };
        let labels = [&self.client_name, table_name];
        if let Some(StatsView::FullReadStatsView(view)) =
            request_stats.as_ref().and_then(|r| r.stats_view.as_ref())
        {
            if let Some(latency) = view
                .request_latency_stats
                .as_ref()
                .and_then(|s| s.frontend_server_latency)
            {
                if latency.seconds < 0 || latency.nanos < 0 {
                    return;
                }
                let duration = Duration::new(latency.seconds as u64, latency.nanos as u32);
                metrics
                    .kv_bt_chunk_latency_ms
                    .with_label_values(&labels)
                    .observe(duration.as_millis() as f64);
            }
            if let Some(iteration_stats) = &view.read_iteration_stats {
                metrics
                    .kv_bt_chunk_rows_returned_count
                    .with_label_values(&labels)
                    .inc_by(iteration_stats.rows_returned_count as u64);
                metrics
                    .kv_bt_chunk_rows_seen_count
                    .with_label_values(&labels)
                    .inc_by(iteration_stats.rows_seen_count as u64);
            }
        }
    }

    async fn multi_get_internal(
        &mut self,
        table_name: &str,
        keys: Vec<Vec<u8>>,
        filter: Option<RowFilter>,
    ) -> Result<Vec<(Bytes, Vec<(Bytes, Bytes)>)>> {
        let version_filter = RowFilter {
            filter: Some(Filter::CellsPerColumnLimitFilter(1)),
        };
        let filter = Some(match filter {
            Some(filter) => RowFilter {
                filter: Some(Filter::Chain(Chain {
                    filters: vec![filter, version_filter],
                })),
            },
            None => version_filter,
        });
        let request = ReadRowsRequest {
            table_name: format!("{}{}", self.table_prefix, table_name),
            rows_limit: keys.len() as i64,
            rows: Some(RowSet {
                row_keys: keys.into_iter().map(Bytes::from).collect(),
                row_ranges: vec![],
            }),
            filter,
            request_stats_view: 2,
            ..ReadRowsRequest::default()
        };
        self.read_rows(request, table_name).await
    }

    /// Scan a range of rows.
    pub(crate) async fn range_scan(
        &mut self,
        table_name: &str,
        start_key: Option<Bytes>,
        end_key: Option<Bytes>,
        limit: i64,
        reversed: bool,
    ) -> Result<Vec<(Bytes, Vec<(Bytes, Bytes)>)>> {
        let start_time = Instant::now();
        let result = self
            .range_scan_internal(table_name, start_key, end_key, limit, reversed)
            .await;
        let elapsed_ms = start_time.elapsed().as_millis() as f64;
        let labels = [&self.client_name, table_name];
        match &self.metrics {
            Some(metrics) => match result {
                Ok(result) => {
                    metrics.kv_scan_success.with_label_values(&labels).inc();
                    if result.is_empty() {
                        metrics.kv_scan_not_found.with_label_values(&labels).inc();
                    }
                    metrics
                        .kv_scan_latency_ms
                        .with_label_values(&labels)
                        .observe(elapsed_ms);
                    Ok(result)
                }
                Err(e) => {
                    metrics.kv_scan_error.with_label_values(&labels).inc();
                    Err(e)
                }
            },
            None => result,
        }
    }

    async fn range_scan_internal(
        &mut self,
        table_name: &str,
        start_key: Option<Bytes>,
        end_key: Option<Bytes>,
        limit: i64,
        reversed: bool,
    ) -> Result<Vec<(Bytes, Vec<(Bytes, Bytes)>)>> {
        let range = RowRange {
            start_key: start_key.map(StartKey::StartKeyClosed),
            end_key: end_key.map(EndKey::EndKeyClosed),
        };
        let filter = Some(RowFilter {
            filter: Some(Filter::CellsPerColumnLimitFilter(1)),
        });
        let request = ReadRowsRequest {
            table_name: format!("{}{}", self.table_prefix, table_name),
            rows_limit: limit,
            rows: Some(RowSet {
                row_keys: vec![],
                row_ranges: vec![range],
            }),
            filter,
            reversed,
            request_stats_view: 2,
            ..ReadRowsRequest::default()
        };
        self.read_rows(request, table_name).await
    }
}

impl std::fmt::Display for PartialWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "partial write: {} entries failed",
            self.failed_keys.len()
        )?;
        for failed in &self.failed_keys {
            write!(f, "\n  code {}: {}", failed.code, failed.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for PartialWriteError {}

impl Service<Request<Body>> for AuthChannel {
    type Response = Response<Body>;
    type Error = Box<dyn std::error::Error + Send + Sync>;
    #[allow(clippy::type_complexity)]
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.channel.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, mut request: Request<Body>) -> Self::Future {
        let cloned_channel = self.channel.clone();
        let cloned_token = self.token.clone();
        let mut inner = std::mem::replace(&mut self.channel, cloned_channel);
        let policy = self.policy.clone();
        let token_provider = self.token_provider.clone();

        let mut auth_token = None;
        if token_provider.is_some() {
            let guard = self.token.read().expect("failed to acquire a read lock");
            if let Some(token) = &*guard {
                if !token.has_expired() {
                    auth_token = Some(token.clone());
                }
            }
        }

        Box::pin(async move {
            if let Some(ref provider) = token_provider {
                let token = match auth_token {
                    None => {
                        let new_token = provider.token(&[policy.as_ref()]).await?;
                        let mut guard = cloned_token.write().unwrap();
                        *guard = Some(new_token.clone());
                        new_token
                    }
                    Some(token) => token,
                };
                let token_string = token.as_str().parse::<String>()?;
                let header =
                    HeaderValue::from_str(format!("Bearer {}", token_string.as_str()).as_str())?;
                request.headers_mut().insert("authorization", header);
            }
            // enable reverse scan
            let header = HeaderValue::from_static("CAE=");
            request.headers_mut().insert("bigtable-features", header);
            Ok(inner.call(request).await?)
        })
    }
}
