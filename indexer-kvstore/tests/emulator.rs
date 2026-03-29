// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! BigTable emulator helper for integration tests.
//! Requires `gcloud` and `cbt` on PATH, and the BigTable emulator component installed.

use std::io::BufRead;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::process::Stdio;

use anyhow::{Context, Result, bail};
use futures::future::try_join_all;
use indexer_kvstore::BigTableClient;
use tokio::process::Command as TokioCommand;

pub const INSTANCE_ID: &str = "bigtable_test_instance";

const TABLES: &[&str] = &[
    indexer_kvstore::tables::checkpoints::NAME,
    indexer_kvstore::tables::checkpoints_by_digest::NAME,
    indexer_kvstore::tables::transactions::NAME,
    indexer_kvstore::tables::objects::NAME,
    indexer_kvstore::tables::epochs::NAME,
    indexer_kvstore::tables::watermark_alt_legacy::NAME,
];

/// Resolve the path to the `cbtemulator` binary shipped with the gcloud SDK.
fn cbtemulator_path() -> Result<PathBuf> {
    let output = Command::new("gcloud")
        .args(["info", "--format=value(installation.sdk_root)"])
        .output()
        .context("failed to run `gcloud info`")?;

    if !output.status.success() {
        bail!("`gcloud info` failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    let sdk_root =
        String::from_utf8(output.stdout).context("non-UTF-8 gcloud output")?.trim().to_string();

    let path = PathBuf::from(sdk_root).join("platform/bigtable-emulator/cbtemulator");
    assert!(
        path.exists(),
        "cbtemulator not found at {}; install with `gcloud components install bigtable`",
        path.display()
    );
    Ok(path)
}

/// Verify that the BigTable emulator tooling is available.
/// Call this at the start of any test that needs the emulator.
#[allow(dead_code)]
pub fn require_bigtable_emulator() -> Result<PathBuf> {
    let path = cbtemulator_path()?;

    // Also verify `cbt` is on PATH.
    let status = Command::new("cbt")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("`cbt` not found on PATH")?;

    if !status.success() {
        bail!("`cbt -version` returned non-zero exit code");
    }

    Ok(path)
}

pub struct BigTableEmulator {
    child: Child,
    host: String,
    _stdout_drain: std::thread::JoinHandle<()>,
}

impl BigTableEmulator {
    /// Start the cbtemulator on an ephemeral port.
    ///
    /// Reads stdout until the emulator prints its listen address, then drains
    /// remaining stdout in a background thread to prevent SIGPIPE.
    pub fn start() -> Result<Self> {
        let emulator_path = cbtemulator_path()?;

        let mut child = Command::new(&emulator_path)
            .arg("-port=0")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to spawn {}", emulator_path.display()))?;

        let stdout = child.stdout.take().expect("stdout was piped");
        let mut reader = std::io::BufReader::new(stdout);

        let host;
        loop {
            let mut line = String::new();
            let n = reader.read_line(&mut line).context("reading cbtemulator stdout")?;
            if n == 0 {
                bail!("cbtemulator exited before printing listen address");
            }
            // The emulator prints: "Cloud Bigtable emulator running on 127.0.0.1:PORT"
            if let Some(idx) = line.find("Cloud Bigtable emulator running on") {
                let addr_start = idx + "Cloud Bigtable emulator running on".len();
                host = line[addr_start..].trim().to_string();
                break;
            }
        }

        // Drain remaining stdout in a background thread to prevent SIGPIPE.
        let drain = std::thread::spawn(move || {
            let mut buf = String::new();
            loop {
                buf.clear();
                match reader.read_line(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
            }
        });

        Ok(Self { child, host, _stdout_drain: drain })
    }

    pub fn host(&self) -> &str {
        &self.host
    }
}

impl Drop for BigTableEmulator {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Create all required BigTable tables on the emulator.
///
/// Runs in two phases: first create all tables, then add column families.
/// (`createfamily` requires the table to already exist.)
pub async fn create_tables(host: &str, instance_id: &str) -> Result<()> {
    // Phase 1: create tables
    let create_futs: Vec<_> = TABLES
        .iter()
        .map(|table| {
            TokioCommand::new("cbt")
                .args(["-instance", instance_id, "-project", "emulator", "createtable", table])
                .env("BIGTABLE_EMULATOR_HOST", host)
                .output()
        })
        .collect();

    let results = try_join_all(create_futs).await.context("cbt createtable commands failed")?;
    for result in &results {
        if !result.status.success() {
            bail!("cbt createtable failed: {}", String::from_utf8_lossy(&result.stderr));
        }
    }

    // Phase 2: create column families (tables must exist first)
    let family_futs: Vec<_> = TABLES
        .iter()
        .map(|table| {
            TokioCommand::new("cbt")
                .args([
                    "-instance",
                    instance_id,
                    "-project",
                    "emulator",
                    "createfamily",
                    table,
                    "soma",
                ])
                .env("BIGTABLE_EMULATOR_HOST", host)
                .output()
        })
        .collect();

    let results = try_join_all(family_futs).await.context("cbt createfamily commands failed")?;
    for result in &results {
        if !result.status.success() {
            bail!("cbt createfamily failed: {}", String::from_utf8_lossy(&result.stderr));
        }
    }

    Ok(())
}

/// Create a `BigTableClient` connected to the emulator.
pub async fn client(host: &str) -> Result<BigTableClient> {
    BigTableClient::new_local(host.to_string(), INSTANCE_ID.to_string()).await
}
