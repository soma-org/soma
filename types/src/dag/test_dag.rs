use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1, take_while_m_n},
    character::complete::{char, digit1, multispace0, multispace1, space0, space1},
    combinator::{map_res, opt},
    multi::{many0, separated_list0},
    sequence::{delimited, preceded, terminated, tuple},
    IResult,
};
use std::{
    collections::{BTreeMap, HashSet},
    ops::{Bound::Included, RangeInclusive},
    sync::Arc,
};

use super::dag_state::DagState;
use crate::committee::AuthorityIndex;
use crate::consensus::{
    block::{
        genesis_blocks, BlockAPI, BlockDigest, BlockRef, BlockTimestampMs, Round, Slot, TestBlock,
        VerifiedBlock,
    },
    commit::{
        sort_sub_dag_blocks, CommitDigest, CommittedSubDag, TrustedCommit, DEFAULT_WAVE_LENGTH,
    },
    context::Context,
    leader_schedule::LeaderSchedule,
};
use parking_lot::RwLock;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

/// DagBuilder API
///
/// Usage:
///
/// DAG Building
/// ```
/// let context = Arc::new(Context::new_for_test(4).0);
/// let dag_builder = DagBuilder::new(context);
/// dag_builder.layer(1).build(); // Round 1 is fully connected with parents by default.
/// dag_builder.layers(2..10).build(); // Rounds 2 ~ 10 are fully connected with parents by default.
/// dag_builder.layers(11).min_parent_links().build(); // Round 11 is minimally and randomly connected with parents, without weak links.
/// dag_builder.layers(12).no_leader_block(0).build(); // Round 12 misses leader block. Other blocks are fully connected with parents.
/// dag_builder.layers(13).no_leader_link(12, 0).build(); // Round 13 misses votes for leader block. Other blocks are fully connected with parents.
/// dag_builder.layers(14).authorities(vec![3,5]).skip_block().build(); // Round 14 authorities 3 and 5 will not propose any block.
/// dag_builder.layers(15).authorities(vec![3,5]).skip_ancestor_links(vec![1,2]).build(); // Round 15 authorities 3 and 5 will not link to ancestors 1 and 2
/// dag_builder.layers(16).authorities(vec![3,5]).equivocate(3).build(); // Round 16 authorities 3 and 5 will produce 3 equivocating blocks.
/// ```
///
/// Persisting to DagState by Layer
/// ```
/// let dag_state = Arc::new(RwLock::new(DagState::new(
///    dag_builder.context.clone(),
///    Arc::new(MemStore::new()),
/// )));
/// let context = Arc::new(Context::new_for_test(4).0);
/// let dag_builder = DagBuilder::new(context);
/// dag_builder.layer(1).build().persist_layers(dag_state.clone()); // persist the layer
/// ```
///
/// Persisting entire DAG to DagState
/// ```
/// let dag_state = Arc::new(RwLock::new(DagState::new(
///    dag_builder.context.clone(),
///    Arc::new(MemStore::new()),
/// )));
/// let context = Arc::new(Context::new_for_test(4).0);
/// let dag_builder = DagBuilder::new(context);
/// dag_builder.layer(1).build();
/// dag_builder.layers(2..10).build();
/// dag_builder.persist_all_blocks(dag_state.clone()); // persist entire DAG
/// ```
///
/// Printing DAG
/// ```
/// let context = Arc::new(Context::new_for_test(4).0);
/// let dag_builder = DagBuilder::new(context);
/// dag_builder.layer(1).build();
/// dag_builder.print(); // pretty print the entire DAG
/// ```
#[allow(unused)]
pub struct DagBuilder {
    pub(crate) context: Arc<Context>,
    pub(crate) leader_schedule: LeaderSchedule,
    // The genesis blocks
    pub(crate) genesis: BTreeMap<BlockRef, VerifiedBlock>,
    // The current set of ancestors that any new layer will attempt to connect to.
    pub(crate) last_ancestors: Vec<BlockRef>,
    // All blocks created by dag builder. Will be used to pretty print or to be
    // retrieved for testing/persiting to dag state.
    pub blocks: BTreeMap<BlockRef, VerifiedBlock>,

    wave_length: Round,
    number_of_leaders: u32,
    pipeline: bool,
}

#[allow(unused)]
impl DagBuilder {
    pub fn new(context: Arc<Context>) -> Self {
        let leader_schedule = LeaderSchedule::new(context.clone());
        let genesis_blocks = genesis_blocks(context.clone());
        let genesis: BTreeMap<BlockRef, VerifiedBlock> = genesis_blocks
            .into_iter()
            .map(|block| (block.reference(), block))
            .collect();
        let last_ancestors = genesis.keys().cloned().collect();
        Self {
            context,
            leader_schedule,
            wave_length: DEFAULT_WAVE_LENGTH,
            number_of_leaders: 1,
            pipeline: false,
            genesis,
            last_ancestors,
            blocks: BTreeMap::new(),
        }
    }

    pub(crate) fn blocks(&self, rounds: RangeInclusive<Round>) -> Vec<VerifiedBlock> {
        assert!(
            !self.blocks.is_empty(),
            "No blocks have been created, please make sure that you have called build method"
        );
        self.blocks
            .iter()
            .filter_map(|(block_ref, block)| rounds.contains(&block_ref.round).then_some(block))
            .cloned()
            .collect::<Vec<VerifiedBlock>>()
    }

    // TODO: reuse logic from Linearizer.
    pub(crate) fn get_sub_dag_and_commit(
        &self,
        leader_block: VerifiedBlock,
        last_committed_rounds: Vec<Round>,
        commit_index: u32,
    ) -> (CommittedSubDag, TrustedCommit) {
        let mut to_commit = Vec::new();
        let mut committed = HashSet::new();

        let timestamp_ms = leader_block.timestamp_ms();
        let leader_block_ref = leader_block.reference();
        let mut buffer = vec![leader_block];
        assert!(committed.insert(leader_block_ref));
        while let Some(x) = buffer.pop() {
            to_commit.push(x.clone());

            let ancestors = self.get_blocks(
                &x.ancestors()
                    .iter()
                    .copied()
                    .filter(|ancestor| {
                        // We skip the block if we already committed it or we reached a
                        // round that we already committed.
                        !committed.contains(ancestor)
                            && last_committed_rounds[ancestor.author] < ancestor.round
                    })
                    .collect::<Vec<_>>(),
            );

            for ancestor in ancestors {
                buffer.push(ancestor.clone());
                assert!(committed.insert(ancestor.reference()));
            }
        }

        sort_sub_dag_blocks(&mut to_commit);

        let commit = TrustedCommit::new_for_test(
            commit_index,
            CommitDigest::MIN,
            timestamp_ms,
            leader_block_ref,
            to_commit
                .iter()
                .map(|block| block.reference())
                .collect::<Vec<_>>(),
        );

        let sub_dag = CommittedSubDag::new(
            leader_block_ref,
            to_commit,
            timestamp_ms,
            commit.reference(),
        );

        (sub_dag, commit)
    }

    pub fn leader_blocks(&self, rounds: RangeInclusive<Round>) -> Vec<Option<VerifiedBlock>> {
        assert!(
            !self.blocks.is_empty(),
            "No blocks have been created, please make sure that you have called build method"
        );
        rounds
            .into_iter()
            .map(|round| self.leader_block(round))
            .collect()
    }

    pub(crate) fn leader_block(&self, round: Round) -> Option<VerifiedBlock> {
        assert!(
            !self.blocks.is_empty(),
            "No blocks have been created, please make sure that you have called build method"
        );
        self.blocks
            .iter()
            .find(|(block_ref, block)| {
                block_ref.round == round
                    && block_ref.author == self.leader_schedule.elect_leader(round, 0)
            })
            .map(|(_block_ref, block)| block.clone())
    }

    pub(crate) fn with_wave_length(mut self, wave_length: Round) -> Self {
        self.wave_length = wave_length;
        self
    }

    pub(crate) fn with_number_of_leaders(mut self, number_of_leaders: u32) -> Self {
        self.number_of_leaders = number_of_leaders;
        self
    }

    pub(crate) fn with_pipeline(mut self, pipeline: bool) -> Self {
        self.pipeline = pipeline;
        self
    }

    pub(crate) fn layer(&mut self, round: Round) -> LayerBuilder {
        LayerBuilder::new(self, round)
    }

    pub fn layers(&mut self, rounds: RangeInclusive<Round>) -> LayerBuilder {
        let mut builder = LayerBuilder::new(self, *rounds.start());
        builder.end_round = Some(*rounds.end());
        builder
    }

    pub(crate) fn persist_all_blocks(&self, dag_state: Arc<RwLock<DagState>>) {
        dag_state
            .write()
            .accept_blocks(self.blocks.values().cloned().collect());
    }

    pub(crate) fn print(&self) {
        let mut dag_str = "DAG {\n".to_string();

        let mut round = 0;
        for block in self.blocks.values() {
            if block.round() > round {
                round = block.round();
                dag_str.push_str(&format!("Round {round} : \n"));
            }
            dag_str.push_str(&format!("    Block {block:#?}\n"));
        }
        dag_str.push_str("}\n");

        tracing::info!("{dag_str}");
    }

    // TODO: merge into layer builder?
    // This method allows the user to specify specific links to ancestors. The
    // layer is written to dag state and the blocks are cached in [`DagBuilder`]
    // state.
    pub(crate) fn layer_with_connections(
        &mut self,
        connections: Vec<(AuthorityIndex, Vec<BlockRef>)>,
        round: Round,
    ) {
        let mut references = Vec::new();
        for (authority, ancestors) in connections {
            let author = authority.value() as u32;
            let base_ts = round as BlockTimestampMs * 1000;
            let block = VerifiedBlock::new_for_test(
                TestBlock::new(round, author)
                    .set_ancestors(ancestors)
                    .set_timestamp_ms(base_ts + author as u64)
                    .build(),
            );
            references.push(block.reference());
            self.blocks.insert(block.reference(), block.clone());
        }
        self.last_ancestors = references;
    }

    /// Gets all uncommitted blocks in a slot.
    pub(crate) fn get_uncommitted_blocks_at_slot(&self, slot: Slot) -> Vec<VerifiedBlock> {
        let mut blocks = vec![];
        for (block_ref, block) in self.blocks.range((
            Included(BlockRef::new(slot.round, slot.authority, BlockDigest::MIN)),
            Included(BlockRef::new(slot.round, slot.authority, BlockDigest::MAX)),
        )) {
            blocks.push(block.clone())
        }
        blocks
    }

    pub(crate) fn get_blocks(&self, block_refs: &[BlockRef]) -> Vec<VerifiedBlock> {
        let mut blocks = vec![None; block_refs.len()];

        for (index, block_ref) in block_refs.iter().enumerate() {
            if block_ref.round == 0 {
                if let Some(block) = self.genesis.get(block_ref) {
                    blocks[index] = Some(block.clone());
                }
                continue;
            }
            if let Some(block) = self.blocks.get(block_ref) {
                blocks[index] = Some(block.clone());
                continue;
            }
        }

        blocks.into_iter().map(|x| x.unwrap()).collect()
    }

    pub(crate) fn genesis_block_refs(&self) -> Vec<BlockRef> {
        self.genesis.keys().cloned().collect()
    }
}

/// Refer to doc comments for [`DagBuilder`] for usage information.
pub struct LayerBuilder<'a> {
    dag_builder: &'a mut DagBuilder,

    start_round: Round,
    end_round: Option<Round>,

    // Configuration options applied to specified authorities
    // TODO: convert configuration options into an enum
    specified_authorities: Option<Vec<AuthorityIndex>>,
    // Number of equivocating blocks per specified authority
    equivocations: usize,
    // Skip block proposal for specified authorities
    skip_block: bool,
    // Skip specified ancestor links for specified authorities
    skip_ancestor_links: Option<Vec<AuthorityIndex>>,
    // Skip leader link for specified authorities
    no_leader_link: bool,

    // Skip leader block proposal
    no_leader_block: bool,
    // Used for leader based configurations
    specified_leader_link_offsets: Option<Vec<u32>>,
    specified_leader_block_offsets: Option<Vec<u32>>,
    leader_round: Option<Round>,

    // All ancestors will be linked to the current layer
    fully_linked_ancestors: bool,
    // Only 2f+1 random ancestors will be linked to the current layer using a
    // seed, if provided
    min_ancestor_links: bool,
    min_ancestor_links_random_seed: Option<u64>,
    // Add random weak links to the current layer using a seed, if provided
    random_weak_links: bool,
    random_weak_links_random_seed: Option<u64>,

    // Ancestors to link to the current layer
    ancestors: Vec<BlockRef>,

    // Accumulated blocks to write to dag state
    blocks: Vec<VerifiedBlock>,
}

#[allow(unused)]
impl<'a> LayerBuilder<'a> {
    fn new(dag_builder: &'a mut DagBuilder, start_round: Round) -> Self {
        assert!(start_round > 0, "genesis round is created by default");
        let ancestors = dag_builder.last_ancestors.clone();
        Self {
            dag_builder,
            start_round,
            end_round: None,
            specified_authorities: None,
            equivocations: 0,
            skip_block: false,
            skip_ancestor_links: None,
            no_leader_link: false,
            no_leader_block: false,
            specified_leader_link_offsets: None,
            specified_leader_block_offsets: None,
            leader_round: None,
            fully_linked_ancestors: true,
            min_ancestor_links: false,
            min_ancestor_links_random_seed: None,
            random_weak_links: false,
            random_weak_links_random_seed: None,
            ancestors,
            blocks: vec![],
        }
    }

    // Configuration methods

    // Only link 2f+1 random ancestors to the current layer round using a seed,
    // if provided. Also provide a flag to guarantee the leader is included.
    // note: configuration is terminal and layer will be built after this call.
    pub fn min_ancestor_links(mut self, include_leader: bool, seed: Option<u64>) -> Self {
        self.min_ancestor_links = true;
        self.min_ancestor_links_random_seed = seed;
        if include_leader {
            self.leader_round = Some(self.ancestors.iter().max_by_key(|b| b.round).unwrap().round);
        }
        self.fully_linked_ancestors = false;
        self.build()
    }

    // No links will be created between the specified ancestors and the specified
    // authorities at the layer round.
    // note: configuration is terminal and layer will be built after this call.
    pub fn skip_ancestor_links(mut self, ancestors_to_skip: Vec<AuthorityIndex>) -> Self {
        // authorities must be specified for this to apply
        assert!(self.specified_authorities.is_some());
        self.skip_ancestor_links = Some(ancestors_to_skip);
        self.fully_linked_ancestors = false;
        self.build()
    }

    // Add random weak links to the current layer round using a seed, if provided
    pub fn random_weak_links(mut self, seed: Option<u64>) -> Self {
        self.random_weak_links = true;
        self.random_weak_links_random_seed = seed;
        self
    }

    // Should be called when building a leader round. Will ensure leader block is missing.
    // A list of specified leader offsets can be provided to skip those leaders.
    // If none are provided all potential leaders for the round will be skipped.
    pub fn no_leader_block(mut self, specified_leader_offsets: Vec<u32>) -> Self {
        self.no_leader_block = true;
        self.specified_leader_block_offsets = Some(specified_leader_offsets);
        self
    }

    // Should be called when building a voting round. Will ensure vote is missing.
    // A list of specified leader offsets can be provided to skip those leader links.
    // If none are provided all potential leaders for the round will be skipped.
    // note: configuration is terminal and layer will be built after this call.
    pub fn no_leader_link(
        mut self,
        leader_round: Round,
        specified_leader_offsets: Vec<u32>,
    ) -> Self {
        self.no_leader_link = true;
        self.specified_leader_link_offsets = Some(specified_leader_offsets);
        self.leader_round = Some(leader_round);
        self.fully_linked_ancestors = false;
        self.build()
    }

    pub fn authorities(mut self, authorities: Vec<AuthorityIndex>) -> Self {
        assert!(
            self.specified_authorities.is_none(),
            "Specified authorities already set"
        );
        self.specified_authorities = Some(authorities);
        self
    }

    // Multiple blocks will be created for the specified authorities at the layer round.
    pub fn equivocate(mut self, equivocations: usize) -> Self {
        // authorities must be specified for this to apply
        assert!(self.specified_authorities.is_some());
        self.equivocations = equivocations;
        self
    }

    // No blocks will be created for the specified authorities at the layer round.
    pub fn skip_block(mut self) -> Self {
        // authorities must be specified for this to apply
        assert!(self.specified_authorities.is_some());
        self.skip_block = true;
        self
    }

    // Apply the configurations & build the dag layer(s).
    pub fn build(mut self) -> Self {
        for round in self.start_round..=self.end_round.unwrap_or(self.start_round) {
            tracing::debug!("BUILDING LAYER ROUND {round}...");

            let authorities = if self.specified_authorities.is_some() {
                self.specified_authorities.clone().unwrap()
            } else {
                self.dag_builder
                    .context
                    .committee
                    .authorities()
                    .map(|x| x.0)
                    .collect()
            };

            // TODO: investigate if these configurations can be called in combination
            // for the same layer
            let mut connections = if self.fully_linked_ancestors {
                self.configure_fully_linked_ancestors()
            } else if self.min_ancestor_links {
                self.configure_min_parent_links()
            } else if self.no_leader_link {
                self.configure_no_leader_links(authorities.clone(), round)
            } else if self.skip_ancestor_links.is_some() {
                self.configure_skipped_ancestor_links(
                    authorities,
                    self.skip_ancestor_links.clone().unwrap(),
                )
            } else {
                vec![]
            };

            if self.random_weak_links {
                connections.append(&mut self.configure_random_weak_links());
            }

            self.create_blocks(round, connections);
        }

        self.dag_builder.last_ancestors = self.ancestors.clone();
        self
    }

    pub fn persist_layers(&self, dag_state: Arc<RwLock<DagState>>) {
        assert!(!self.blocks.is_empty(), "Called to persist layers although no blocks have been created. Make sure you have called build before.");
        dag_state.write().accept_blocks(self.blocks.clone());
    }

    // Layer round is minimally and randomly connected with ancestors.
    pub fn configure_min_parent_links(&mut self) -> Vec<(AuthorityIndex, Vec<BlockRef>)> {
        let quorum_threshold = self.dag_builder.context.committee.quorum_threshold() as usize;
        let mut authorities: Vec<AuthorityIndex> = self
            .dag_builder
            .context
            .committee
            .authorities()
            .map(|authority| authority.0)
            .collect();

        let mut rng = match self.min_ancestor_links_random_seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut authorities_to_shuffle = authorities.clone();

        let mut leaders = vec![];
        if let Some(leader_round) = self.leader_round {
            let leader_offsets = (0..self.dag_builder.number_of_leaders).collect::<Vec<_>>();

            for leader_offset in leader_offsets {
                leaders.push(
                    self.dag_builder
                        .leader_schedule
                        .elect_leader(leader_round, leader_offset),
                );
            }
        }

        authorities
            .iter()
            .map(|authority| {
                authorities_to_shuffle.shuffle(&mut rng);

                // TODO: handle quroum threshold properly with stake
                let min_ancestors: HashSet<AuthorityIndex> = authorities_to_shuffle
                    .iter()
                    .take(quorum_threshold)
                    .cloned()
                    .collect();

                (
                    *authority,
                    self.ancestors
                        .iter()
                        .filter(|a| {
                            leaders.contains(&a.author) || min_ancestors.contains(&a.author)
                        })
                        .cloned()
                        .collect::<Vec<BlockRef>>(),
                )
            })
            .collect()
    }

    // TODO: configure layer round randomly connected with weak links.
    fn configure_random_weak_links(&mut self) -> Vec<(AuthorityIndex, Vec<BlockRef>)> {
        unimplemented!("configure_random_weak_links");
    }

    // Layer round misses link to leader, but other blocks are fully connected with ancestors.
    fn configure_no_leader_links(
        &mut self,
        authorities: Vec<AuthorityIndex>,
        round: Round,
    ) -> Vec<(AuthorityIndex, Vec<BlockRef>)> {
        let mut missing_leaders = Vec::new();
        let mut specified_leader_offsets = self
            .specified_leader_link_offsets
            .clone()
            .expect("specified_leader_offsets should be set");
        let leader_round = self.leader_round.expect("leader round should be set");

        // When no specified leader offsets are available, all leaders are
        // expected to be missing.
        if specified_leader_offsets.is_empty() {
            specified_leader_offsets.extend(0..self.dag_builder.number_of_leaders);
        }

        for leader_offset in specified_leader_offsets {
            missing_leaders.push(
                self.dag_builder
                    .leader_schedule
                    .elect_leader(leader_round, leader_offset),
            );
        }

        self.configure_skipped_ancestor_links(authorities, missing_leaders)
    }

    fn configure_fully_linked_ancestors(&mut self) -> Vec<(AuthorityIndex, Vec<BlockRef>)> {
        self.dag_builder
            .context
            .committee
            .authorities()
            .map(|authority| (authority.0, self.ancestors.clone()))
            .collect::<Vec<_>>()
    }

    fn configure_skipped_ancestor_links(
        &mut self,
        authorities: Vec<AuthorityIndex>,
        ancestors_to_skip: Vec<AuthorityIndex>,
    ) -> Vec<(AuthorityIndex, Vec<BlockRef>)> {
        let filtered_ancestors = self
            .ancestors
            .clone()
            .into_iter()
            .filter(|ancestor| !ancestors_to_skip.contains(&ancestor.author))
            .collect::<Vec<_>>();
        authorities
            .into_iter()
            .map(|authority| (authority, filtered_ancestors.clone()))
            .collect::<Vec<_>>()
    }

    // Creates the blocks for the new layer based on configured connections, also
    // sets the ancestors for future layers to be linked to
    fn create_blocks(&mut self, round: Round, connections: Vec<(AuthorityIndex, Vec<BlockRef>)>) {
        let mut references = Vec::new();
        for (authority, ancestors) in connections {
            if self.should_skip_block(round, authority) {
                continue;
            };
            let num_blocks = self.num_blocks_to_create(authority);

            for num_block in 0..num_blocks {
                let author = authority.value() as u32;
                let base_ts = round as BlockTimestampMs * 1000;
                let block = VerifiedBlock::new_for_test(
                    TestBlock::new(round, author)
                        .set_ancestors(ancestors.clone())
                        .set_timestamp_ms(base_ts + (author + round + num_block) as u64)
                        .build(),
                );
                references.push(block.reference());
                self.dag_builder
                    .blocks
                    .insert(block.reference(), block.clone());
                self.blocks.push(block);
            }
        }
        self.ancestors = references;
    }

    fn num_blocks_to_create(&self, authority: AuthorityIndex) -> u32 {
        if self.specified_authorities.is_some()
            && self
                .specified_authorities
                .clone()
                .unwrap()
                .contains(&authority)
        {
            // Always create 1 block and then the equivocating blocks on top of that.
            1 + self.equivocations as u32
        } else {
            1
        }
    }

    fn should_skip_block(&self, round: Round, authority: AuthorityIndex) -> bool {
        // Safe to unwrap as specified authorites has to be set before skip
        // is specified.
        if self.skip_block
            && self
                .specified_authorities
                .clone()
                .unwrap()
                .contains(&authority)
        {
            return true;
        }
        if self.no_leader_block {
            let mut specified_leader_offsets = self
                .specified_leader_block_offsets
                .clone()
                .expect("specified_leader_block_offsets should be set");

            // When no specified leader offsets are available, all leaders are
            // expected to be skipped.
            if specified_leader_offsets.is_empty() {
                specified_leader_offsets.extend(0..self.dag_builder.number_of_leaders);
            }

            for leader_offset in specified_leader_offsets {
                let leader = self
                    .dag_builder
                    .leader_schedule
                    .elect_leader(round, leader_offset);

                if leader == authority {
                    return true;
                }
            }
        }
        false
    }
}

/// DagParser
///
/// Usage:
///
/// ```
/// let dag_str = "DAG {
///     Round 0 : { 4 },
///     Round 1 : { * },
///     Round 2 : { * },
///     Round 3 : { * },
///     Round 4 : {
///         A -> [-D3],
///         B -> [*],
///         C -> [*],
///         D -> [*],
///     },
///     Round 5 : {
///         A -> [*],
///         B -> [*],
///         C -> [A4],
///         D -> [A4],
///     },
///     Round 6 : { * },
///     Round 7 : { * },
///     Round 8 : { * },
///     }";
///
/// let (_, dag_builder) = parse_dag(dag_str).expect("Invalid dag"); // parse DAG DSL
/// dag_builder.print(); // print the parsed DAG
/// dag_builder.persist_all_blocks(dag_state.clone()); // persist all blocks to DagState
/// ```

pub(crate) fn parse_dag(dag_string: &str) -> IResult<&str, DagBuilder> {
    let (input, _) = tuple((tag("DAG"), multispace0, char('{')))(dag_string)?;

    let (mut input, num_authors) = parse_genesis(input)?;

    let context = Arc::new(Context::new_for_test(num_authors as usize).0);
    let mut dag_builder = DagBuilder::new(context);

    // Parse subsequent rounds
    loop {
        match parse_round(input, &dag_builder) {
            Ok((new_input, (round, connections))) => {
                dag_builder.layer_with_connections(connections, round);
                input = new_input
            }
            Err(nom::Err::Error(_)) | Err(nom::Err::Failure(_)) => break,
            Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
        }
    }
    let (input, _) = tuple((multispace0, char('}')))(input)?;

    Ok((input, dag_builder))
}

fn parse_round<'a>(
    input: &'a str,
    dag_builder: &DagBuilder,
) -> IResult<&'a str, (Round, Vec<(AuthorityIndex, Vec<BlockRef>)>)> {
    let (input, _) = tuple((multispace0, tag("Round"), space1))(input)?;
    let (input, round) = take_while1(|c: char| c.is_ascii_digit())(input)?;

    let (input, connections) = alt((
        |input| parse_fully_connected(input, dag_builder),
        |input| parse_specified_connections(input, dag_builder),
    ))(input)?;

    Ok((input, (round.parse().unwrap(), connections)))
}

fn parse_fully_connected<'a>(
    input: &'a str,
    dag_builder: &DagBuilder,
) -> IResult<&'a str, Vec<(AuthorityIndex, Vec<BlockRef>)>> {
    let (input, _) = tuple((
        space0,
        char(':'),
        space0,
        char('{'),
        space0,
        char('*'),
        space0,
        char('}'),
        opt(char(',')),
    ))(input)?;

    let ancestors = dag_builder.last_ancestors.clone();
    let connections = dag_builder
        .context
        .committee
        .authorities()
        .map(|authority| (authority.0, ancestors.clone()))
        .collect::<Vec<_>>();

    Ok((input, connections))
}

fn parse_specified_connections<'a>(
    input: &'a str,
    dag_builder: &DagBuilder,
) -> IResult<&'a str, Vec<(AuthorityIndex, Vec<BlockRef>)>> {
    let (input, _) = tuple((space0, char(':'), space0, char('{'), multispace0))(input)?;

    // parse specified connections
    // case 1: all authorities; [*]
    // case 2: specific included authorities; [A0, B0, C0]
    // case 3: specific excluded authorities;  [-A0]
    // case 4: mixed all authorities + specific included/excluded authorities; [*, A0]
    // TODO: case 5: byzantine case of multiple blocks per slot; [*]; timestamp=1
    let (input, authors_and_connections) = many0(parse_author_and_connections)(input)?;

    let mut output = Vec::new();
    for (author, connections) in authors_and_connections {
        let mut block_refs = HashSet::new();
        for connection in connections {
            if connection == "*" {
                block_refs.extend(dag_builder.last_ancestors.clone());
            } else if connection.starts_with('-') {
                let (input, _) = char('-')(connection)?;
                let (_, slot) = parse_slot(input)?;
                let stored_block_refs = get_blocks(slot, dag_builder);
                block_refs.extend(dag_builder.last_ancestors.clone());

                block_refs.retain(|ancestor| !stored_block_refs.contains(ancestor));
            } else {
                let input = connection;
                let (_, slot) = parse_slot(input)?;
                let stored_block_refs = get_blocks(slot, dag_builder);

                block_refs.extend(stored_block_refs);
            }
        }
        output.push((author, block_refs.into_iter().collect()));
    }

    let (input, _) = tuple((multispace0, char('}'), opt(char(','))))(input)?;

    Ok((input, output))
}

fn get_blocks(slot: Slot, dag_builder: &DagBuilder) -> Vec<BlockRef> {
    // note: special case for genesis blocks as they are cached separately
    let block_refs = if slot.round == 0 {
        dag_builder
            .genesis_block_refs()
            .into_iter()
            .filter(|block| Slot::from(*block) == slot)
            .collect::<Vec<_>>()
    } else {
        dag_builder
            .get_uncommitted_blocks_at_slot(slot)
            .iter()
            .map(|block| block.reference())
            .collect::<Vec<_>>()
    };
    block_refs
}

fn parse_author_and_connections(input: &str) -> IResult<&str, (AuthorityIndex, Vec<&str>)> {
    // parse author
    let (input, author) = preceded(
        multispace0,
        terminated(
            take_while1(|c: char| c.is_alphabetic()),
            preceded(opt(space0), tag("->")),
        ),
    )(input)?;

    // parse connections
    let (input, connections) = delimited(
        preceded(opt(space0), char('[')),
        separated_list0(tag(", "), parse_block),
        terminated(char(']'), opt(char(','))),
    )(input)?;
    let (input, _) = opt(multispace1)(input)?;
    Ok((
        input,
        (
            str_to_authority_index(author).expect("Invalid authority index"),
            connections,
        ),
    ))
}

fn parse_block(input: &str) -> IResult<&str, &str> {
    alt((
        map_res(tag("*"), |s: &str| Ok::<_, nom::error::ErrorKind>(s)),
        map_res(
            take_while1(|c: char| c.is_alphanumeric() || c == '-'),
            |s: &str| Ok::<_, nom::error::ErrorKind>(s),
        ),
    ))(input)
}

fn parse_genesis(input: &str) -> IResult<&str, u32> {
    let (input, num_authorities) = preceded(
        tuple((
            multispace0,
            tag("Round"),
            space1,
            char('0'),
            space0,
            char(':'),
            space0,
            char('{'),
            space0,
        )),
        |i| parse_authority_count(i),
    )(input)?;
    let (input, _) = tuple((space0, char('}'), opt(char(','))))(input)?;

    Ok((input, num_authorities))
}

fn parse_authority_count(input: &str) -> IResult<&str, u32> {
    let (input, num_str) = digit1(input)?;
    Ok((input, num_str.parse().unwrap()))
}

fn parse_slot(input: &str) -> IResult<&str, Slot> {
    let parse_authority = map_res(
        take_while_m_n(1, 1, |c: char| c.is_alphabetic() && c.is_uppercase()),
        |letter: &str| {
            Ok::<_, nom::error::ErrorKind>(
                str_to_authority_index(letter).expect("Invalid authority index"),
            )
        },
    );

    let parse_round = map_res(digit1, |digits: &str| digits.parse::<Round>());

    let mut parser = tuple((parse_authority, parse_round));

    let (input, (authority, round)) = parser(input)?;
    Ok((input, Slot::new(round, authority)))
}

// Helper function to convert a string representation (e.g., 'A' or '[26]') to an AuthorityIndex
fn str_to_authority_index(input: &str) -> Option<AuthorityIndex> {
    if input.starts_with('[') && input.ends_with(']') && input.len() > 2 {
        input[1..input.len() - 1]
            .parse::<u32>()
            .ok()
            .map(AuthorityIndex::new_for_test)
    } else if input.len() == 1 && input.chars().next()?.is_ascii_uppercase() {
        // Handle single uppercase ASCII alphabetic character
        let alpha_char = input.chars().next().unwrap();
        let index = alpha_char as u32 - 'A' as u32;
        Some(AuthorityIndex::new_for_test(index))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::consensus::block::BlockAPI;

    #[tokio::test]
    async fn test_dag_parsing() {
        let _ = tracing_subscriber::fmt::init();
        let dag_str = "DAG { 
            Round 0 : { 4 },
            Round 1 : { * },
            Round 2 : { * },
            Round 3 : {
                A -> [*],
                B -> [*],
                C -> [*],
                D -> [*],
            },
            Round 4 : {
                A -> [A3, B3, C3],
                B -> [A3, B3, C3],
                C -> [A3, B3, C3],
                D -> [*],
            },
            Round 5 : {
                A -> [*],
                B -> [-A4],
                C -> [-A4],
                D -> [-A4],
            },
            Round 6 : {
                A -> [A3, B3, C3, A1, B1],
                B -> [*, A0],
                C -> [-A5],
            }
         }";
        let result = parse_dag(dag_str);
        assert!(result.is_ok());

        let (_, dag_builder) = result.unwrap();
        assert_eq!(dag_builder.genesis.len(), 4);
        assert_eq!(dag_builder.blocks.len(), 23);

        // Check the blocks were correctly parsed in Round 6
        let blocks_a6 = dag_builder
            .get_uncommitted_blocks_at_slot(Slot::new(6, AuthorityIndex::new_for_test(0)));
        assert_eq!(blocks_a6.len(), 1);
        let block_a6 = blocks_a6.first().unwrap();
        assert_eq!(block_a6.round(), 6);
        assert_eq!(block_a6.author(), AuthorityIndex::new_for_test(0));
        assert_eq!(block_a6.ancestors().len(), 5);
        let expected_block_a6_ancestor_slots = [
            Slot::new(3, AuthorityIndex::new_for_test(0)),
            Slot::new(3, AuthorityIndex::new_for_test(1)),
            Slot::new(3, AuthorityIndex::new_for_test(2)),
            Slot::new(1, AuthorityIndex::new_for_test(0)),
            Slot::new(1, AuthorityIndex::new_for_test(1)),
        ];
        for ancestor in block_a6.ancestors() {
            assert!(expected_block_a6_ancestor_slots.contains(&Slot::from(*ancestor)));
        }

        let blocks_b6 = dag_builder
            .get_uncommitted_blocks_at_slot(Slot::new(6, AuthorityIndex::new_for_test(1)));
        assert_eq!(blocks_b6.len(), 1);
        let block_b6 = blocks_b6.first().unwrap();
        assert_eq!(block_b6.round(), 6);
        assert_eq!(block_b6.author(), AuthorityIndex::new_for_test(1));
        assert_eq!(block_b6.ancestors().len(), 5);
        let expected_block_b6_ancestor_slots = [
            Slot::new(5, AuthorityIndex::new_for_test(0)),
            Slot::new(5, AuthorityIndex::new_for_test(1)),
            Slot::new(5, AuthorityIndex::new_for_test(2)),
            Slot::new(5, AuthorityIndex::new_for_test(3)),
            Slot::new(0, AuthorityIndex::new_for_test(0)),
        ];
        for ancestor in block_b6.ancestors() {
            assert!(expected_block_b6_ancestor_slots.contains(&Slot::from(*ancestor)));
        }

        let blocks_c6 = dag_builder
            .get_uncommitted_blocks_at_slot(Slot::new(6, AuthorityIndex::new_for_test(2)));
        assert_eq!(blocks_c6.len(), 1);
        let block_c6 = blocks_c6.first().unwrap();
        assert_eq!(block_c6.round(), 6);
        assert_eq!(block_c6.author(), AuthorityIndex::new_for_test(2));
        assert_eq!(block_c6.ancestors().len(), 3);
        let expected_block_c6_ancestor_slots = [
            Slot::new(5, AuthorityIndex::new_for_test(1)),
            Slot::new(5, AuthorityIndex::new_for_test(2)),
            Slot::new(5, AuthorityIndex::new_for_test(3)),
        ];
        for ancestor in block_c6.ancestors() {
            assert!(expected_block_c6_ancestor_slots.contains(&Slot::from(*ancestor)));
        }
    }

    #[tokio::test]
    async fn test_genesis_round_parsing() {
        let dag_str = "Round 0 : { 4 }";
        let result = parse_genesis(dag_str);
        assert!(result.is_ok());
        let (_, num_authorities) = result.unwrap();

        assert_eq!(num_authorities, 4);
    }

    #[tokio::test]
    async fn test_slot_parsing() {
        let dag_str = "A0";
        let result = parse_slot(dag_str);
        assert!(result.is_ok());
        let (_, slot) = result.unwrap();

        assert_eq!(slot.authority, str_to_authority_index("A").unwrap());
        assert_eq!(slot.round, 0);
    }

    #[tokio::test]
    async fn test_all_round_parsing() {
        let dag_str = "Round 1 : { * }";
        let context = Arc::new(Context::new_for_test(4).0);
        let dag_builder = DagBuilder::new(context);
        let result = parse_round(dag_str, &dag_builder);
        assert!(result.is_ok());
        let (_, (round, connections)) = result.unwrap();

        assert_eq!(round, 1);
        for (i, (authority, references)) in connections.into_iter().enumerate() {
            assert_eq!(authority, AuthorityIndex::new_for_test(i as u32));
            assert_eq!(references, dag_builder.last_ancestors);
        }
    }

    #[tokio::test]
    async fn test_specific_round_parsing() {
        let dag_str = "Round 1 : {
            A -> [A0, B0, C0, D0],
            B -> [*, A0],
            C -> [-A0],
        }";
        let context = Arc::new(Context::new_for_test(4).0);
        let dag_builder = DagBuilder::new(context);
        let result = parse_round(dag_str, &dag_builder);
        assert!(result.is_ok());
        let (_, (round, connections)) = result.unwrap();

        let skipped_slot = Slot::new_for_test(0, 0); // A0
        let mut expected_references = vec![
            dag_builder.last_ancestors.clone(),
            dag_builder.last_ancestors.clone(),
            dag_builder
                .last_ancestors
                .into_iter()
                .filter(|ancestor| Slot::from(*ancestor) != skipped_slot)
                .collect(),
        ];

        assert_eq!(round, 1);
        for (i, (authority, mut references)) in connections.into_iter().enumerate() {
            assert_eq!(authority, AuthorityIndex::new_for_test(i as u32));
            references.sort();
            expected_references[i].sort();
            assert_eq!(references, expected_references[i]);
        }
    }

    #[tokio::test]
    async fn test_parse_author_and_connections() {
        let expected_authority = str_to_authority_index("A").unwrap();

        // case 1: all authorities
        let dag_str = "A -> [*]";
        let result = parse_author_and_connections(dag_str);
        assert!(result.is_ok());
        let (_, (actual_author, actual_connections)) = result.unwrap();
        assert_eq!(actual_author, expected_authority);
        assert_eq!(actual_connections, ["*"]);

        // case 2: specific included authorities
        let dag_str = "A -> [A0, B0, C0]";
        let result = parse_author_and_connections(dag_str);
        assert!(result.is_ok());
        let (_, (actual_author, actual_connections)) = result.unwrap();
        assert_eq!(actual_author, expected_authority);
        assert_eq!(actual_connections, ["A0", "B0", "C0"]);

        // case 3: specific excluded authorities
        let dag_str = "A -> [-A0, -B0]";
        let result = parse_author_and_connections(dag_str);
        assert!(result.is_ok());
        let (_, (actual_author, actual_connections)) = result.unwrap();
        assert_eq!(actual_author, expected_authority);
        assert_eq!(actual_connections, ["-A0", "-B0"]);

        // case 4: mixed all authorities + specific included/excluded authorities
        let dag_str = "A -> [*, A0, -B0]";
        let result = parse_author_and_connections(dag_str);
        assert!(result.is_ok());
        let (_, (actual_author, actual_connections)) = result.unwrap();
        assert_eq!(actual_author, expected_authority);
        assert_eq!(actual_connections, ["*", "A0", "-B0"]);

        // TODO: case 5: byzantine case of multiple blocks per slot; [*]; timestamp=1
    }

    #[tokio::test]
    async fn test_str_to_authority_index() {
        assert_eq!(
            str_to_authority_index("A"),
            Some(AuthorityIndex::new_for_test(0))
        );
        assert_eq!(
            str_to_authority_index("Z"),
            Some(AuthorityIndex::new_for_test(25))
        );
        assert_eq!(
            str_to_authority_index("[26]"),
            Some(AuthorityIndex::new_for_test(26))
        );
        assert_eq!(
            str_to_authority_index("[100]"),
            Some(AuthorityIndex::new_for_test(100))
        );
        assert_eq!(str_to_authority_index("a"), None);
        assert_eq!(str_to_authority_index("0"), None);
        assert_eq!(str_to_authority_index(" "), None);
        assert_eq!(str_to_authority_index("!"), None);
    }
}

/// Build a fully interconnected dag up to the specified round. This function
/// starts building the dag from the specified [`start`] parameter or from
/// genesis if none are specified up to and including the specified round [`stop`]
/// parameter.
pub(crate) fn build_dag(
    context: Arc<Context>,
    dag_state: Arc<RwLock<DagState>>,
    start: Option<Vec<BlockRef>>,
    stop: Round,
) -> Vec<BlockRef> {
    let mut ancestors = match start {
        Some(start) => {
            assert!(!start.is_empty());
            assert_eq!(
                start.iter().map(|x| x.round).max(),
                start.iter().map(|x| x.round).min()
            );
            start
        }
        None => genesis_blocks(context.clone())
            .iter()
            .map(|x| x.reference())
            .collect::<Vec<_>>(),
    };

    let num_authorities = context.committee.size();
    let starting_round = ancestors.first().unwrap().round + 1;
    for round in starting_round..=stop {
        let (references, blocks): (Vec<_>, Vec<_>) = context
            .committee
            .authorities()
            .map(|authority| {
                let author_idx = authority.0.value() as u32;
                // Test the case where a block from round R+1 has smaller timestamp than a block from round R.
                let ts = round as BlockTimestampMs / 2 * num_authorities as BlockTimestampMs
                    + author_idx as BlockTimestampMs;
                let block = VerifiedBlock::new_for_test(
                    TestBlock::new(round, author_idx)
                        .set_timestamp_ms(ts)
                        .set_ancestors(ancestors.clone())
                        .build(),
                );

                (block.reference(), block)
            })
            .unzip();
        dag_state.write().accept_blocks(blocks);
        ancestors = references;
    }

    ancestors
}

// TODO: Add layer_round as input parameter so ancestors can be from any round.
pub(crate) fn build_dag_layer(
    // A list of (authority, parents) pairs. For each authority, we add a block
    // linking to the specified parents.
    connections: Vec<(AuthorityIndex, Vec<BlockRef>)>,
    dag_state: Arc<RwLock<DagState>>,
) -> Vec<BlockRef> {
    let mut references = Vec::new();
    for (authority, ancestors) in connections {
        let round = ancestors.first().unwrap().round + 1;
        let author = authority.value() as u32;
        let block = VerifiedBlock::new_for_test(
            TestBlock::new(round, author)
                .set_ancestors(ancestors)
                .build(),
        );
        references.push(block.reference());
        dag_state.write().accept_block(block);
    }
    references
}

pub(crate) fn create_random_dag(
    seed: u64,
    include_leader_percentage: u64,
    num_rounds: Round,
    context: Arc<Context>,
) -> DagBuilder {
    assert!(
        (0..=100).contains(&include_leader_percentage),
        "include_leader_percentage must be in the range 0..100"
    );

    let mut rng = StdRng::seed_from_u64(seed);
    let mut dag_builder = DagBuilder::new(context);

    for r in 1..=num_rounds {
        let random_num = rng.gen_range(0..100);
        let include_leader = random_num <= include_leader_percentage;
        dag_builder
            .layer(r)
            .min_ancestor_links(include_leader, Some(random_num));
    }

    dag_builder
}
