//! # Committee Management
//!
//! ## Overview
//! This module defines the structures and logic for managing validator committees in the
//! Soma blockchain. Committees are responsible for consensus, transaction validation, and
//! network operations, with voting power distributed among validators.
//!
//! ## Responsibilities
//! - Define the validator committee structure and membership
//! - Manage validator voting power and stake distribution
//! - Provide committee-related constants like quorum thresholds
//! - Support authority selection based on stake weight
//! - Handle epoch transitions and committee reconfiguration
//!
//! ## Component Relationships
//! - Used by consensus module to determine leader selection and voting rights
//! - Used by authority module to validate transaction certificates
//! - Used by node module for validator discovery and networking
//! - Provides the foundation for Byzantine Fault Tolerance in the system
//!
//! ## Key Workflows
//! 1. Committee creation at genesis and during epoch transitions
//! 2. Validator selection weighted by stake for leader election
//! 3. Threshold verification for transaction commit certificates
//! 4. Authority identity and metadata management
//!
//! ## Design Patterns
//! - Immutable committee structure for thread safety
//! - Caching of derived data like authority indices for performance
//! - Fixed total voting power with normalized stake distribution
//! - Deterministic stake-weighted authority selection

use crate::base::{AuthorityName, ConciseableName};
use crate::consensus::committee::get_available_local_address;
use crate::crypto::{
    get_key_pair_from_rng, random_committee_key_pairs_of_size, AuthorityKeyPair,
    AuthorityPublicKey, NetworkKeyPair, NetworkPublicKey, ProtocolKeyPair, ProtocolPublicKey,
    DIGEST_LENGTH,
};
use crate::crypto::{AuthoritySignature, DefaultHash as DefaultHashFunction};
use crate::digests::TransactionDigest;
use crate::error::{ConsensusError, ConsensusResult, SomaError, SomaResult};
use crate::intent::{Intent, IntentMessage, IntentScope};
use crate::multiaddr::Multiaddr;
use fastcrypto::ed25519::{Ed25519KeyPair, Ed25519PublicKey};
use fastcrypto::hash::HashFunction;
use fastcrypto::traits::{KeyPair, Signer, VerifyingKey};
use rand::rngs::{OsRng, StdRng, ThreadRng};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use shared::authority_committee::AuthorityCommittee;
use shared::crypto::keys::EncoderPublicKey;
use shared::encoder_committee::Encoder;
use shared::probe::ProbeMetadata;
use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{Display, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};
use tracing::info;

/// Identifier for a specific epoch in the blockchain's history.
///
/// Epochs represent discrete time periods during which a specific committee
/// of validators is active. The EpochId increments with each reconfiguration.
pub type EpochId = u64;

/// Represents the voting power of an authority in the committee.
///
/// The sum of all voting powers in a committee is fixed at TOTAL_VOTING_POWER.
/// Voting power determines the influence of an authority in consensus decisions.
pub type VotingPower = u64;

/// Hash digest type for committee state.
///
/// Used for efficiently representing and comparing committee configurations.
pub type CommitteeDigest = [u8; 32];

/// Total voting power across all validators in a committee, fixed at 10,000.
///
/// Individual voting powers can be interpreted as basis points
/// (e.g., voting_power: 100 = 1%, voting_power: 1 = 0.01%).
/// Fixing the total voting power allows clients to hardcode the quorum threshold
/// and total voting power rather than recomputing these.
pub const TOTAL_VOTING_POWER: VotingPower = 10_000;

/// Quorum threshold for BFT consensus (2f+1).
///
/// Any message signed by this much voting power can be trusted
/// up to BFT assumptions. This represents approximately 66.67% of
/// the total voting power.
pub const QUORUM_THRESHOLD: VotingPower = 6_667;

/// Validity threshold defined by f+1 (approximately 33.34%).
///
/// Used for determining when enough validators have seen or acknowledged
/// a message. This is the minimum threshold needed to ensure at least
/// one honest validator has processed a message.
pub const VALIDITY_THRESHOLD: VotingPower = 3_334;

// Cap voting power of an individual validator at 10%.
// TODO: determine what this should be
pub const MAX_VOTING_POWER: u64 = 1_000;

/// Minimum amount of voting power required to become a validator
/// .12% of voting power
// TODO: consider making this .06 or .03
pub const VALIDATOR_CONSENSUS_MIN_POWER: u64 = 12;

/// Low voting power threshold for validators
/// Validators below this threshold fall into the "at risk" group.
/// .08% of voting power
// TODO: consider making this .04 or .02
pub const VALIDATOR_CONSENSUS_LOW_POWER: u64 = 8;

/// Very low voting power threshold for validators
/// Validators below this threshold will be removed immediately at epoch change.
/// .04% of voting power
// TODO: consider making this .02 or .01
pub const VALIDATOR_CONSENSUS_VERY_LOW_POWER: u64 = 4;

/// Minimum amount of voting power required to become a networking validator
/// This is the minimum stake required to communicate with validators and encoders
/// 0.01% of voting power
pub const VALIDATOR_NETWORKING_MIN_POWER: u64 = 1;

/// A validator can have stake below `validator_low_stake_threshold`
/// for this many epochs before being kicked out.
pub const VALIDATOR_LOW_STAKE_GRACE_PERIOD: u64 = 7;

/// Minimum amount of voting power required to become a encoder
/// .12% of voting power
// TODO: consider making this .06 or .03
pub const ENCODER_MIN_POWER: u64 = 12;

/// Low voting power threshold for encoders
/// Encoders below this threshold fall into the "at risk" group.
/// .08% of voting power
// TODO: consider making this .04 or .02
pub const ENCODER_LOW_POWER: u64 = 8;

/// Very low voting power threshold for encoders
/// Encoders below this threshold will be removed immediately at epoch change.
/// .04% of voting power
// TODO: consider making this .02 or .01
pub const ENCODER_VERY_LOW_POWER: u64 = 4;

/// A encoder can have stake below `eoder_low_stake_threshold`
/// for this many epochs before being kicked out.
pub const ENCODER_LOW_STAKE_GRACE_PERIOD: u64 = 7;

/// Represents a committee of validators for a specific epoch.
///
/// The Committee structure tracks validator membership, voting power distribution,
/// authority metadata, and provides operations for validator selection and threshold
/// verification.
///
/// ## Thread Safety
/// This structure is immutable after creation and can be safely shared across threads
/// using Arc<Committee>.
///
/// ## Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # type EpochId = u64;
/// # type AuthorityName = [u8; 32];
/// # type VotingPower = u64;
/// # struct Authority {}
/// # struct Committee { epoch: EpochId, voting_rights: Vec<(AuthorityName, VotingPower)> }
/// # impl Committee {
/// #     fn new(_: EpochId, _: BTreeMap<AuthorityName, VotingPower>, _: BTreeMap<AuthorityName, Authority>) -> Self {
/// #         Committee { epoch: 0, voting_rights: vec![] }
/// #     }
/// #     fn quorum_threshold(&self) -> VotingPower { 6667 }
/// #     fn stake(&self, _: &AuthorityName) -> VotingPower { 1000 }
/// # }
/// # let mut voting_rights = BTreeMap::new();
/// # let mut authorities = BTreeMap::new();
/// # voting_rights.insert([0; 32], 5000);
/// # voting_rights.insert([1; 32], 5000);
/// // Create a committee for epoch 1
/// let committee = Committee::new(1, voting_rights, authorities);
///
/// // Check if a certificate has reached quorum
/// # let certificate_stake = 7000;
/// if certificate_stake >= committee.quorum_threshold() {
///     // Certificate has quorum and can be trusted
/// }
///
/// // Get an authority's voting power
/// # let authority_name = [0; 32];
/// let stake = committee.stake(&authority_name);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, Eq)]
pub struct Committee {
    /// The epoch this committee is active for
    pub epoch: EpochId,

    /// Ordered list of authorities and their voting power
    pub voting_rights: Vec<(AuthorityName, VotingPower)>,

    /// Cached mapping from authority names to their public keys
    expanded_keys: HashMap<AuthorityName, AuthorityPublicKey>,

    /// Cached mapping from authority names to their index in voting_rights
    index_map: HashMap<AuthorityName, usize>,

    /// Detailed information about each authority
    pub authorities: HashMap<AuthorityName, Authority>,
}

impl Committee {
    /// Creates a new committee with the specified epoch, voting rights, and authorities.
    ///
    /// # Arguments
    /// * `epoch` - The epoch ID for this committee
    /// * `voting_rights` - Mapping of authority names to their voting power
    /// * `authorities` - Mapping of authority names to their detailed information
    ///
    /// # Returns
    /// A new Committee instance
    ///
    /// # Panics
    /// This function will panic if:
    /// - The voting rights collection is empty
    /// - All authorities have zero voting power
    /// - The total voting power doesn't equal TOTAL_VOTING_POWER (10,000)
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::BTreeMap;
    /// # type AuthorityName = [u8; 32];
    /// # type VotingPower = u64;
    /// # struct Authority {}
    /// # struct Committee { epoch: u64 }
    /// # impl Committee {
    /// #     fn new(epoch: u64, _: BTreeMap<AuthorityName, VotingPower>, _: BTreeMap<AuthorityName, Authority>) -> Self {
    /// #         Committee { epoch }
    /// #     }
    /// # }
    /// # let mut voting_rights = BTreeMap::new();
    /// # let mut authorities = BTreeMap::new();
    /// # voting_rights.insert([0; 32], 5000);
    /// # voting_rights.insert([1; 32], 5000);
    /// let committee = Committee::new(1, voting_rights, authorities);
    /// ```
    pub fn new(
        epoch: EpochId,
        voting_rights: BTreeMap<AuthorityName, VotingPower>,
        authorities: BTreeMap<AuthorityName, Authority>,
    ) -> Self {
        let mut voting_rights_vec: Vec<(AuthorityName, VotingPower)> =
            voting_rights.iter().map(|(a, s)| (*a, *s)).collect();

        // Existing validation
        assert!(!voting_rights_vec.is_empty());
        assert!(voting_rights_vec.iter().any(|(_, s)| *s != 0));

        voting_rights_vec.sort_by_key(|(a, _)| *a);
        let total_votes: VotingPower = voting_rights_vec.iter().map(|(_, votes)| *votes).sum();
        // assert_eq!(total_votes, TOTAL_VOTING_POWER);

        let (expanded_keys, index_map) = Self::load_inner(&voting_rights_vec);

        Committee {
            epoch,
            voting_rights: voting_rights_vec,
            expanded_keys,
            index_map,
            authorities: authorities.into_iter().collect(),
        }
    }

    pub fn new_simple_test_committee_of_size(size: usize) -> (Self, Vec<AuthorityKeyPair>) {
        let mut rng = StdRng::from_seed([0; 32]); // Or keep random_committee_key_pairs_of_size
        let mut authorities = BTreeMap::new();
        let mut voting_weights = BTreeMap::new();
        let mut key_pairs = Vec::new();

        for i in 0..size {
            let authority_keypair = AuthorityKeyPair::generate(&mut rng);
            let protocol_keypair = ProtocolKeyPair::generate(&mut rng);
            let network_keypair = NetworkKeyPair::generate(&mut rng);

            let name = AuthorityName::from(authority_keypair.public());

            authorities.insert(
                name,
                Authority {
                    stake: 1, // Will be normalized
                    address: get_available_local_address(),
                    hostname: format!("test_host_{i}"),
                    authority_key: authority_keypair.public().clone(),
                    protocol_key: protocol_keypair.public(),
                    network_key: network_keypair.public(),
                },
            );

            voting_weights.insert(name, 1);
            key_pairs.push(authority_keypair);
        }

        let committee =
            Self::new_for_testing_with_normalized_voting_power(0, voting_weights, authorities);

        (committee, key_pairs)
    }

    /// Normalize the given weights to TOTAL_VOTING_POWER and create the committee.
    /// Used for testing only
    pub fn new_for_testing_with_normalized_voting_power(
        epoch: EpochId,
        mut voting_weights: BTreeMap<AuthorityName, VotingPower>,
        authorities: BTreeMap<AuthorityName, Authority>,
    ) -> Self {
        let num_nodes = voting_weights.len();
        let total_votes: VotingPower = voting_weights.values().cloned().sum();
        let normalization_coef = TOTAL_VOTING_POWER as f64 / total_votes as f64;
        let mut total_sum = 0;

        // Normalize voting weights first
        for (idx, (_auth, weight)) in voting_weights.iter_mut().enumerate() {
            if idx < num_nodes - 1 {
                *weight = (*weight as f64 * normalization_coef).floor() as u64;
                total_sum += *weight;
            } else {
                *weight = TOTAL_VOTING_POWER - total_sum;
            }
        }

        Self::new(epoch, voting_weights, authorities)
    }

    pub fn convert_to_authority_committee(committee: &Committee) -> AuthorityCommittee {
        // Extract authorities from the Committee
        let authorities = committee
            .authorities()
            .map(|(_, auth)| shared::authority_committee::Authority {
                stake: auth.stake,
                authority_key: shared::crypto::keys::AuthorityPublicKey::new(
                    auth.authority_key.clone(),
                ),
                protocol_key: shared::crypto::keys::ProtocolPublicKey::new(
                    auth.protocol_key.inner(),
                ),
            })
            .collect();

        // Create new AuthorityCommittee with proper constructor
        AuthorityCommittee::new(committee.epoch(), authorities)
    }

    // We call this if these have not yet been computed
    pub fn load_inner(
        voting_rights: &[(AuthorityName, VotingPower)],
    ) -> (
        HashMap<AuthorityName, AuthorityPublicKey>,
        HashMap<AuthorityName, usize>,
    ) {
        let expanded_keys: HashMap<AuthorityName, AuthorityPublicKey> = voting_rights
            .iter()
            .map(|(addr, _)| {
                (
                    *addr,
                    (*addr)
                        .try_into()
                        .expect("Validator pubkey is always verified on-chain"),
                )
            })
            .collect();

        let index_map: HashMap<AuthorityName, usize> = voting_rights
            .iter()
            .enumerate()
            .map(|(index, (addr, _))| (*addr, index))
            .collect();
        (expanded_keys, index_map)
    }

    pub fn authority_index(&self, author: &AuthorityName) -> Option<u32> {
        self.index_map.get(author).map(|i| *i as u32)
    }

    pub fn authority_by_index(&self, index: u32) -> Option<&AuthorityName> {
        self.voting_rights.get(index as usize).map(|(name, _)| name)
    }

    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    pub fn public_key(&self, authority: &AuthorityName) -> SomaResult<&AuthorityPublicKey> {
        debug_assert_eq!(self.expanded_keys.len(), self.voting_rights.len());
        match self.expanded_keys.get(authority) {
            Some(v) => Ok(v),
            None => Err(SomaError::InvalidCommittee(format!(
                "Authority #{} not found, committee size {}",
                authority,
                self.expanded_keys.len()
            ))),
        }
    }

    /// Samples authorities by weight
    pub fn sample(&self) -> &AuthorityName {
        // unwrap safe unless committee is empty
        Self::choose_multiple_weighted(&self.voting_rights[..], 1, &mut ThreadRng::default())
            .next()
            .unwrap()
    }

    fn choose_multiple_weighted<'a>(
        slice: &'a [(AuthorityName, VotingPower)],
        count: usize,
        rng: &mut impl Rng,
    ) -> impl Iterator<Item = &'a AuthorityName> {
        // unwrap is safe because we validate the committee composition in `new` above.
        // See https://docs.rs/rand/latest/rand/distributions/weighted/enum.WeightedError.html
        // for possible errors.
        slice
            .choose_multiple_weighted(rng, count, |(_, weight)| *weight as f64)
            .unwrap()
            .map(|(a, _)| a)
    }

    pub fn choose_multiple_weighted_iter(
        &self,
        count: usize,
    ) -> impl Iterator<Item = &AuthorityName> {
        self.voting_rights
            .choose_multiple_weighted(&mut ThreadRng::default(), count, |(_, weight)| {
                *weight as f64
            })
            .unwrap()
            .map(|(a, _)| a)
    }

    pub fn total_votes(&self) -> VotingPower {
        TOTAL_VOTING_POWER
    }

    pub fn quorum_threshold(&self) -> VotingPower {
        QUORUM_THRESHOLD
    }

    pub fn validity_threshold(&self) -> VotingPower {
        VALIDITY_THRESHOLD
    }

    pub fn threshold<const STRENGTH: bool>(&self) -> VotingPower {
        if STRENGTH {
            QUORUM_THRESHOLD
        } else {
            VALIDITY_THRESHOLD
        }
    }

    pub fn num_members(&self) -> usize {
        self.voting_rights.len()
    }

    pub fn members(&self) -> impl Iterator<Item = &(AuthorityName, VotingPower)> {
        self.voting_rights.iter()
    }

    pub fn names(&self) -> impl Iterator<Item = &AuthorityName> {
        self.voting_rights.iter().map(|(name, _)| name)
    }

    pub fn stakes(&self) -> impl Iterator<Item = VotingPower> + '_ {
        self.voting_rights.iter().map(|(_, stake)| *stake)
    }

    pub fn authority_exists(&self, name: &AuthorityName) -> bool {
        self.voting_rights
            .binary_search_by_key(name, |(a, _)| *a)
            .is_ok()
    }

    /// Derive a seed deterministically from the transaction digest and shuffle the validators.
    pub fn shuffle_by_stake_from_tx_digest(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Vec<AuthorityName> {
        // the 32 is as requirement of the default StdRng::from_seed choice
        let digest_bytes = tx_digest.into_inner();

        // permute the validators deterministically, based on the digest
        let mut rng = StdRng::from_seed(digest_bytes);
        self.shuffle_by_stake_with_rng(None, None, &mut rng)
    }

    pub fn total_stake(&self) -> VotingPower {
        self.total_votes()
    }

    pub fn stake(&self, authority: &AuthorityName) -> VotingPower {
        self.weight(authority)
    }

    pub fn authority(&self, name: &AuthorityName) -> Option<&Authority> {
        self.authorities.get(name)
    }

    pub fn authorities(&self) -> impl Iterator<Item = (AuthorityIndex, &Authority)> {
        self.voting_rights
            .iter()
            .enumerate()
            .map(|(idx, (name, _))| {
                (
                    AuthorityIndex(idx as u32),
                    self.authorities.get(name).expect("Authority must exist"),
                )
            })
    }

    pub fn reached_quorum(&self, stake: VotingPower) -> bool {
        stake >= self.quorum_threshold()
    }

    pub fn reached_validity(&self, stake: VotingPower) -> bool {
        stake >= self.validity_threshold()
    }

    pub fn is_valid_index(&self, index: AuthorityIndex) -> bool {
        (index.value() as usize) < self.voting_rights.len()
    }

    pub fn size(&self) -> usize {
        self.voting_rights.len()
    }

    pub fn stake_by_index(&self, index: AuthorityIndex) -> VotingPower {
        self.voting_rights
            .get(index.value())
            .map(|(_, stake)| *stake)
            .unwrap_or(0)
    }

    pub fn authority_by_authority_index(&self, index: AuthorityIndex) -> Option<&Authority> {
        self.voting_rights
            .get(index.value())
            .map(|(name, _)| self.authorities.get(name))
            .flatten()
    }

    pub fn to_authority_index(&self, index: usize) -> Option<AuthorityIndex> {
        if index < self.voting_rights.len() {
            Some(AuthorityIndex(index as u32))
        } else {
            None
        }
    }
}

impl CommitteeTrait<AuthorityName> for Committee {
    fn shuffle_by_stake_with_rng(
        &self,
        // try these authorities first
        preferences: Option<&BTreeSet<AuthorityName>>,
        // only attempt from these authorities.
        restrict_to: Option<&BTreeSet<AuthorityName>>,
        rng: &mut impl Rng,
    ) -> Vec<AuthorityName> {
        let restricted = self
            .voting_rights
            .iter()
            .filter(|(name, _)| {
                if let Some(restrict_to) = restrict_to {
                    restrict_to.contains(name)
                } else {
                    true
                }
            })
            .cloned();

        let (preferred, rest): (Vec<_>, Vec<_>) = if let Some(preferences) = preferences {
            restricted.partition(|(name, _)| preferences.contains(name))
        } else {
            (Vec::new(), restricted.collect())
        };

        Self::choose_multiple_weighted(&preferred, preferred.len(), rng)
            .chain(Self::choose_multiple_weighted(&rest, rest.len(), rng))
            .cloned()
            .collect()
    }

    fn weight(&self, author: &AuthorityName) -> VotingPower {
        match self.voting_rights.binary_search_by_key(author, |(a, _)| *a) {
            Err(_) => 0,
            Ok(idx) => self.voting_rights[idx].1,
        }
    }
}

impl PartialEq for Committee {
    fn eq(&self, other: &Self) -> bool {
        self.epoch == other.epoch && self.voting_rights == other.voting_rights
    }
}

impl Hash for Committee {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.epoch.hash(state);
        self.voting_rights.hash(state);
    }
}

impl Display for Committee {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut voting_rights = String::new();
        for (name, vote) in &self.voting_rights {
            write!(voting_rights, "{}: {}, ", name.concise(), vote)?;
        }
        write!(
            f,
            "Committee (epoch={:?}, voting_rights=[{}])",
            self.epoch, voting_rights
        )
    }
}

pub trait CommitteeTrait<K: Ord> {
    fn shuffle_by_stake_with_rng(
        &self,
        // try these authorities first
        preferences: Option<&BTreeSet<K>>,
        // only attempt from these authorities.
        restrict_to: Option<&BTreeSet<K>>,
        rng: &mut impl Rng,
    ) -> Vec<K>;

    fn shuffle_by_stake(
        &self,
        // try these authorities first
        preferences: Option<&BTreeSet<K>>,
        // only attempt from these authorities.
        restrict_to: Option<&BTreeSet<K>>,
    ) -> Vec<K> {
        self.shuffle_by_stake_with_rng(preferences, restrict_to, &mut ThreadRng::default())
    }

    fn weight(&self, author: &K) -> VotingPower;
}
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct NetworkMetadata {
    // Existing network fields
    pub consensus_address: Multiaddr,
    pub network_address: Multiaddr,
    pub primary_address: Multiaddr,
    pub encoder_validator_address: Multiaddr,

    // Added fields from ValidatorMetadata
    pub protocol_key: ProtocolPublicKey,
    pub network_key: NetworkPublicKey,
    pub authority_key: AuthorityPublicKey,
    pub hostname: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitteeWithNetworkMetadata {
    epoch_id: EpochId,
    validators: BTreeMap<AuthorityName, (VotingPower, NetworkMetadata)>,

    #[serde(skip)]
    committee: OnceCell<Committee>,
}
impl CommitteeWithNetworkMetadata {
    pub fn new(
        epoch_id: EpochId,
        validators: BTreeMap<AuthorityName, (VotingPower, NetworkMetadata)>,
    ) -> Self {
        Self {
            epoch_id,
            validators,
            committee: OnceCell::new(),
        }
    }
    pub fn epoch(&self) -> EpochId {
        self.epoch_id
    }

    pub fn validators(&self) -> &BTreeMap<AuthorityName, (VotingPower, NetworkMetadata)> {
        &self.validators
    }

    pub fn committee(&self) -> &Committee {
        self.committee.get_or_init(|| {
            let voting_rights: BTreeMap<_, _> = self
                .validators
                .iter()
                .map(|(name, (stake, _))| (*name, *stake))
                .collect();

            let authorities = self
                .validators
                .iter()
                .map(|(name, (stake, meta))| {
                    (
                        *name,
                        Authority {
                            stake: *stake,
                            address: meta.consensus_address.clone(),
                            hostname: meta.hostname.clone(),
                            protocol_key: meta.protocol_key.clone(),
                            network_key: meta.network_key.clone(),
                            authority_key: meta.authority_key.clone(),
                        },
                    )
                })
                .collect();

            Committee::new(self.epoch_id, voting_rights, authorities)
        })
    }
}

impl Display for CommitteeWithNetworkMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CommitteeWithNetworkMetadata (epoch={}, validators={:?})",
            self.epoch_id, self.validators
        )
    }
}

pub type Epoch = EpochId;

/// Voting power of an authority, roughly proportional to the actual amount staked
/// by the authority.
/// Total stake / voting power of all authorities should sum to 10,000.
pub type Stake = VotingPower;

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct Authority {
    /// Voting power of the authority in the committee.
    pub stake: Stake,

    /// Network address for communicating with the authority.
    pub address: Multiaddr,

    /// The authority's hostname, for metrics and logging.
    pub hostname: String,

    /// The authority's public key for verifying blocks.
    pub protocol_key: ProtocolPublicKey,

    /// The authority's public key for TLS and as network identity.
    pub network_key: NetworkPublicKey,

    pub authority_key: AuthorityPublicKey,
}

#[derive(
    Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Debug, Default, Hash, Serialize, Deserialize,
)]
pub struct AuthorityIndex(pub u32);

impl AuthorityIndex {
    pub const ZERO: Self = Self(0);
    pub const MIN: Self = Self::ZERO;
    pub const MAX: Self = Self(u32::MAX);

    pub fn value(&self) -> usize {
        self.0 as usize
    }

    pub fn new_for_test(index: u32) -> Self {
        Self(index)
    }
}

impl Display for AuthorityIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.value() < 26 {
            let c = (b'A' + self.value() as u8) as char;
            f.write_str(c.to_string().as_str())
        } else {
            write!(f, "[{:02}]", self.value())
        }
    }
}

impl<T> Index<AuthorityIndex> for Vec<T> {
    type Output = T;

    fn index(&self, index: AuthorityIndex) -> &Self::Output {
        self.get(index.value()).unwrap()
    }
}

impl<T> IndexMut<AuthorityIndex> for Vec<T> {
    fn index_mut(&mut self, index: AuthorityIndex) -> &mut Self::Output {
        self.get_mut(index.value()).unwrap()
    }
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct EncoderCommittee {
    /// The epoch number
    pub epoch: EpochId,

    /// The active encoders with their voting power
    pub members: BTreeMap<EncoderPublicKey, u64>,

    /// Network metadata for encoders
    pub network_metadata: BTreeMap<EncoderPublicKey, EncoderNetworkMetadata>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct EncoderNetworkMetadata {
    pub network_address: Multiaddr,
    pub object_server_address: Multiaddr,
    pub network_key: NetworkPublicKey,
    pub hostname: String,
}

/// Digest of encoder committee, used for signing
#[derive(Serialize, Deserialize)]
pub struct EncoderCommitteeDigest([u8; DIGEST_LENGTH]);

impl EncoderCommittee {
    pub fn compute_digest(&self) -> ConsensusResult<EncoderCommitteeDigest> {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(bcs::to_bytes(self).map_err(ConsensusError::SerializationFailure)?);
        Ok(EncoderCommitteeDigest(hasher.finalize().into()))
    }

    pub fn sign(&self, keypair: &AuthorityKeyPair) -> ConsensusResult<AuthoritySignature> {
        let digest = self.compute_digest()?;
        let message = bcs::to_bytes(&to_encoder_committee_intent(digest))
            .map_err(ConsensusError::SerializationFailure)?;
        Ok(keypair.sign(&message))
    }

    pub fn verify_signature(
        &self,
        signature: &AuthoritySignature,
        public_key: &AuthorityPublicKey,
    ) -> ConsensusResult<()> {
        let digest = self.compute_digest()?;
        let message = bcs::to_bytes(&to_encoder_committee_intent(digest))
            .map_err(ConsensusError::SerializationFailure)?;
        public_key
            .verify(&message, signature)
            .map_err(ConsensusError::SignatureVerificationFailure)
    }

    pub fn convert_encoder_committee(
        committee: &EncoderCommittee,
        epoch: EpochId,
    ) -> shared::encoder_committee::EncoderCommittee {
        // Extract encoders with their voting powers
        let encoders = committee
            .members
            .iter()
            .map(|(key, voting_power)| {
                // Calculate voting power as u16
                let voting_power = (*voting_power as u16).min(10_000);

                // TODO: Create a test probe metadata (will be replaced with real data in production)
                let mut seed = [0u8; 32];
                seed[0..8].copy_from_slice(&key.to_bytes()[0..8]); // Use part of the public key as seed
                let probe = ProbeMetadata::new_for_test(&seed);

                Encoder {
                    voting_power,
                    encoder_key: key.clone(),
                    probe,
                }
            })
            .collect::<Vec<_>>();

        let shard_size = std::cmp::min(
            encoders.len() as u32,
            std::cmp::max(3, (encoders.len() / 2) as u32),
        );

        // Calculate quorum threshold - typically 2/3 rounded up
        let quorum_threshold = (shard_size * 2 + 2) / 3;

        // Create the encoder service EncoderCommittee
        shared::encoder_committee::EncoderCommittee::new(
            epoch,
            shard_size,
            quorum_threshold,
            encoders,
        )
    }
}

/// Wrap an EncoderCommitteeDigest in the intent message
pub fn to_encoder_committee_intent(
    digest: EncoderCommitteeDigest,
) -> IntentMessage<EncoderCommitteeDigest> {
    IntentMessage::new(Intent::consensus_app(IntentScope::EncoderCommittee), digest)
}

/// Represents a committee of networking validators for a specific epoch.
///
/// NetworkingCommittee includes all validators with sufficient stake to participate
/// in network communications, serving as DDOS protection. This is a superset of
/// consensus validators and includes networking-only validators.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
pub struct NetworkingCommittee {
    /// The epoch this committee is active for
    pub epoch: EpochId,

    /// Mapping of authority names to their network metadata
    /// Note: We don't store voting power here as networking validators
    /// don't participate in consensus decisions
    pub members: BTreeMap<AuthorityName, NetworkMetadata>,
}
/// Digest of networking committee, used for signing
#[derive(Serialize, Deserialize)]
pub struct NetworkingCommitteeDigest([u8; DIGEST_LENGTH]);

impl NetworkingCommittee {
    /// Creates a new networking committee for the specified epoch
    pub fn new(epoch: EpochId, members: BTreeMap<AuthorityName, NetworkMetadata>) -> Self {
        Self { epoch, members }
    }

    /// Get the epoch this committee is active for
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// Get all networking committee members
    pub fn members(&self) -> &BTreeMap<AuthorityName, NetworkMetadata> {
        &self.members
    }

    /// Check if a validator is in the networking committee
    pub fn contains(&self, name: &AuthorityName) -> bool {
        self.members.contains_key(name)
    }

    /// Get network metadata for a specific validator
    pub fn get_metadata(&self, name: &AuthorityName) -> Option<&NetworkMetadata> {
        self.members.get(name)
    }

    /// Get the total number of networking participants
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Compute the digest of the networking committee for signing
    pub fn compute_digest(&self) -> ConsensusResult<NetworkingCommitteeDigest> {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(bcs::to_bytes(self).map_err(ConsensusError::SerializationFailure)?);
        Ok(NetworkingCommitteeDigest(hasher.finalize().into()))
    }

    /// Sign the networking committee with the given keypair
    pub fn sign(&self, keypair: &AuthorityKeyPair) -> ConsensusResult<AuthoritySignature> {
        let digest = self.compute_digest()?;
        let message = bcs::to_bytes(&to_networking_committee_intent(digest))
            .map_err(ConsensusError::SerializationFailure)?;
        Ok(keypair.sign(&message))
    }

    /// Verify a signature on the networking committee
    pub fn verify_signature(
        &self,
        signature: &AuthoritySignature,
        public_key: &AuthorityPublicKey,
    ) -> ConsensusResult<()> {
        let digest = self.compute_digest()?;
        let message = bcs::to_bytes(&to_networking_committee_intent(digest))
            .map_err(ConsensusError::SerializationFailure)?;
        public_key
            .verify(&message, signature)
            .map_err(ConsensusError::SignatureVerificationFailure)
    }
}

/// Wrap a NetworkingCommitteeDigest in the intent message
pub fn to_networking_committee_intent(
    digest: NetworkingCommitteeDigest,
) -> IntentMessage<NetworkingCommitteeDigest> {
    IntentMessage::new(
        Intent::consensus_app(IntentScope::NetworkingCommittee),
        digest,
    )
}
