// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Ephemeral Postgres database for testing.
//!
//! Spins up a fresh Postgres instance per test using `initdb` + `pg_ctl`.
//! Requires `postgres`, `initdb`, and `pg_isready` on PATH.
//! Install via `brew install postgresql` on macOS.

use std::io::Read as _;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use tempfile::TempDir;
use url::Url;

/// An ephemeral Postgres instance backed by a temporary directory.
/// Automatically stops the server and cleans up on drop.
pub struct TempDb {
    dir: TempDir,
    port: u16,
    url: Url,
}

impl TempDb {
    /// Spin up a new ephemeral Postgres database.
    ///
    /// # Panics
    /// Panics if `initdb`, `pg_ctl`, or `pg_isready` are not on PATH,
    /// or if the server fails to start within 30 seconds.
    pub fn new() -> Self {
        let dir = TempDir::new().expect("failed to create temp dir");
        let data_dir = dir.path().join("pgdata");
        let port = pick_unused_port();

        // initdb
        let output = Command::new("initdb")
            .args([
                "-D",
                data_dir.to_str().unwrap(),
                "--no-locale",
                "--encoding=UTF8",
                "-A",
                "trust",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .expect("failed to run initdb — is postgresql installed?");
        assert!(
            output.status.success(),
            "initdb failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // pg_ctl start
        let log_file = dir.path().join("postgres.log");
        let output = Command::new("pg_ctl")
            .args([
                "-D",
                data_dir.to_str().unwrap(),
                "-l",
                log_file.to_str().unwrap(),
                "-o",
                &format!(
                    "-p {} -k {} -h ''",
                    port,
                    dir.path().to_str().unwrap()
                ),
                "start",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .expect("failed to run pg_ctl");
        assert!(
            output.status.success(),
            "pg_ctl start failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Wait for ready
        let start = Instant::now();
        let socket_dir = dir.path().to_str().unwrap().to_string();
        loop {
            let ok = Command::new("pg_isready")
                .args(["-h", &socket_dir, "-p", &port.to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);

            if ok {
                break;
            }

            if start.elapsed() > Duration::from_secs(30) {
                // Read the log file for diagnostics
                let mut log = String::new();
                if let Ok(mut f) = std::fs::File::open(&log_file) {
                    let _ = f.read_to_string(&mut log);
                }
                panic!(
                    "Postgres did not become ready within 30s.\nLog:\n{}",
                    log
                );
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        // Create the test database
        let output = Command::new("createdb")
            .args([
                "-h",
                &socket_dir,
                "-p",
                &port.to_string(),
                "indexer_test",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .expect("failed to run createdb");
        assert!(
            output.status.success(),
            "createdb failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let url = Url::parse(&format!(
            "postgres://localhost:{}/indexer_test?host={}",
            port,
            dir.path().to_str().unwrap()
        ))
        .expect("invalid URL");

        Self { dir, port, url }
    }

    /// The connection URL for this database.
    pub fn url(&self) -> &Url {
        &self.url
    }

    /// The port the database is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// The data directory path.
    pub fn data_dir(&self) -> PathBuf {
        self.dir.path().join("pgdata")
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let data_dir = self.data_dir();
        // pg_ctl stop -m immediate
        let _ = Command::new("pg_ctl")
            .args([
                "-D",
                data_dir.to_str().unwrap(),
                "-m",
                "immediate",
                "stop",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

/// Pick an unused TCP port by briefly binding to port 0.
fn pick_unused_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("failed to bind ephemeral port")
        .local_addr()
        .unwrap()
        .port()
}
