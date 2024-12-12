use std::sync::Arc;

use super::context::Context;
use crate::committee::AuthorityIndex;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
/// The `LeaderSchedule` is responsible for producing the leader schedule across
/// an epoch.
#[derive(Clone)]
pub struct LeaderSchedule {
    context: Arc<Context>,
}

impl LeaderSchedule {
    pub fn new(context: Arc<Context>) -> Self {
        Self { context }
    }

    pub(crate) fn elect_leader(&self, round: u32, leader_offset: u32) -> AuthorityIndex {
        cfg_if::cfg_if! {
            if #[cfg(test)] {
                let leader = AuthorityIndex::new_for_test((round + leader_offset) % self.context.committee.size() as u32);
                leader
            } else {
                let leader = self.elect_leader_stake_based(round, leader_offset);
                leader
            }
        }
    }

    pub(crate) fn elect_leader_stake_based(&self, round: u32, offset: u32) -> AuthorityIndex {
        assert!((offset as usize) < self.context.committee.size());

        // To ensure that we elect different leaders for the same round (using
        // different offset) we are using the round number as seed to shuffle in
        // a weighted way the results, but skip based on the offset.
        let mut seed_bytes = [0u8; 32];
        seed_bytes[32 - 4..].copy_from_slice(&(round).to_le_bytes());
        let mut rng = StdRng::from_seed(seed_bytes);

        let choices = self
            .context
            .committee
            .authorities()
            .map(|(index, authority)| (index, authority.stake as f32))
            .collect::<Vec<_>>();

        let leader_index = *choices
            .choose_multiple_weighted(&mut rng, self.context.committee.size(), |item| item.1)
            .expect("Weighted choice error: stake values incorrect!")
            .skip(offset as usize)
            .map(|(index, _)| index)
            .next()
            .unwrap();

        leader_index
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[tokio::test]
    async fn test_elect_leader() {
        let context = Arc::new(Context::new_for_test(4).0);
        let leader_schedule = LeaderSchedule::new(context);

        assert_eq!(
            leader_schedule.elect_leader(0, 0),
            AuthorityIndex::new_for_test(0)
        );
        assert_eq!(
            leader_schedule.elect_leader(1, 0),
            AuthorityIndex::new_for_test(1)
        );
        assert_eq!(
            leader_schedule.elect_leader(5, 0),
            AuthorityIndex::new_for_test(1)
        );
        // ensure we elect different leaders for the same round for the multi-leader case
        assert_ne!(
            leader_schedule.elect_leader_stake_based(1, 1),
            leader_schedule.elect_leader_stake_based(1, 2)
        );
    }

    #[tokio::test]
    async fn test_elect_leader_stake_based() {
        let context = Arc::new(Context::new_for_test(4).0);
        let leader_schedule = LeaderSchedule::new(context);

        assert_eq!(
            leader_schedule.elect_leader_stake_based(0, 0),
            AuthorityIndex::new_for_test(1)
        );
        assert_eq!(
            leader_schedule.elect_leader_stake_based(1, 0),
            AuthorityIndex::new_for_test(1)
        );
        assert_eq!(
            leader_schedule.elect_leader_stake_based(5, 0),
            AuthorityIndex::new_for_test(3)
        );
        // ensure we elect different leaders for the same round for the multi-leader case
        assert_ne!(
            leader_schedule.elect_leader_stake_based(1, 1),
            leader_schedule.elect_leader_stake_based(1, 2)
        );
    }
}
