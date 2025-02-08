use std::{sync::Arc, time::SystemTime};

use super::{block::BlockTimestampMs, committee::local_committee_and_keys};
use crate::committee::{AuthorityIndex, Committee};
use crate::crypto::{AuthorityKeyPair, NetworkKeyPair, ProtocolKeyPair};
use crate::parameters::Parameters;
use parking_lot::RwLock;
// #[cfg(test)]
use tempfile::TempDir;
use tokio::time::Instant;

#[derive(Clone)]
pub struct Context {
    /// Index of this authority in the committee.
    pub own_index: Arc<RwLock<Option<AuthorityIndex>>>,
    /// Committee of the current epoch.
    pub committee: Committee,
    // /// Parameters of this authority.
    pub parameters: Parameters,
    // /// Protocol configuration of current epoch.
    // pub protocol_config: ProtocolConfig,
    // /// Metrics of this authority.
    // pub metrics: Arc<Metrics>,
    /// Access to local clock
    pub clock: Arc<Clock>,
}

impl Context {
    pub fn new(
        own_index: Option<AuthorityIndex>,
        committee: Committee,
        parameters: Parameters,
        // protocol_config: ProtocolConfig,
        // metrics: Arc<Metrics>,
        clock: Arc<Clock>,
    ) -> Self {
        Self {
            own_index: Arc::new(RwLock::new(own_index)),
            committee,
            parameters,
            // protocol_config,
            // metrics,
            clock,
        }
    }

    /// Create a test context with a committee of given size and even stake
    // #[cfg(test)]
    pub fn new_for_test(
        committee_size: usize,
    ) -> (
        Self,
        Vec<(NetworkKeyPair, ProtocolKeyPair)>,
        Vec<AuthorityKeyPair>,
    ) {
        let (committee, keypairs, authority_keypairs) =
            local_committee_and_keys(0, vec![1; committee_size]);
        let clock = Arc::new(Clock::new());
        let temp_dir = TempDir::new().unwrap();
        let context = Context::new(
            Some(AuthorityIndex::new_for_test(0)),
            committee,
            Parameters {
                db_path: temp_dir.into_path(),
                ..Default::default()
            },
            // ProtocolConfig::get_for_max_version_UNSAFE(),
            // metrics,
            clock,
        );
        (context, keypairs, authority_keypairs)
    }

    // #[cfg(test)]
    pub fn with_authority_index(mut self, authority: AuthorityIndex) -> Self {
        self.own_index = Arc::new(RwLock::new(Some(authority)));
        self
    }

    // #[cfg(test)]
    pub fn with_committee(mut self, committee: Committee) -> Self {
        self.committee = committee;
        self
    }

    // #[cfg(test)]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn set_own_index(&self, index: AuthorityIndex) {
        *self.own_index.write() = Some(index);
    }

    pub fn own_index(&self) -> Option<AuthorityIndex> {
        *self.own_index.read()
    }
}

/// A clock that allows to derive the current UNIX system timestamp while guaranteeing that
/// timestamp will be monotonically incremented having tolerance to ntp and system clock changes and corrections.
/// Explicitly avoid to make `[Clock]` cloneable to ensure that a single instance is shared behind an `[Arc]`
/// wherever is needed in order to make sure that consecutive calls to receive the system timestamp
/// will remain monotonically increasing.
pub struct Clock {
    unix_epoch_instant: Instant,
}

impl Clock {
    pub fn new() -> Self {
        let now = Instant::now();
        let duration_since_unix_epoch =
            match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
                Ok(d) => d,
                Err(e) => panic!("SystemTime before UNIX EPOCH! {e}"),
            };
        let unix_epoch_instant = now.checked_sub(duration_since_unix_epoch).unwrap();

        Self { unix_epoch_instant }
    }

    // Returns the current time expressed as UNIX timestamp in milliseconds.
    // Calculated with Rust Instant to ensure monotonicity.
    pub fn timestamp_utc_ms(&self) -> BlockTimestampMs {
        Instant::now()
            .checked_duration_since(self.unix_epoch_instant)
            .unwrap()
            .as_millis() as BlockTimestampMs
    }
}
