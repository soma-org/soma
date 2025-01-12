use std::sync::Arc;

use super::context::Context;
use crate::{
    committee::{AuthorityIndex, Committee, EpochId},
    storage::read_store::ReadStore,
};
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
/// The `LeaderSchedule` is responsible for producing the leader schedule across
/// an epoch.
#[derive(Clone)]
pub struct LeaderSchedule {
    context: Arc<Context>,
    committee_store: Option<Arc<dyn ReadStore>>,
}

impl LeaderSchedule {
    pub fn new(context: Arc<Context>, committee_store: Option<Arc<dyn ReadStore>>) -> Self {
        Self {
            context,
            committee_store,
        }
    }

    pub fn get_committee(&self, epoch: EpochId) -> Committee {
        if let Some(committee_store) = &self.committee_store {
            if let Ok(Some(committee)) = committee_store.get_committee(epoch) {
                (*committee).clone()
            } else {
                self.context.committee.clone()
            }
        } else {
            self.context.committee.clone()
        }
    }

    pub(crate) fn elect_leader(
        &self,
        round: u32,
        leader_offset: u32,
        epoch: Option<EpochId>,
    ) -> AuthorityIndex {
        cfg_if::cfg_if! {
            if #[cfg(test)] {
                let committee = epoch.map_or_else(
                    || self.context.committee.clone(),
                    |epoch| self.get_committee(epoch),
                );
                let leader = AuthorityIndex::new_for_test((round + leader_offset) % committee.size() as u32);
                leader
            } else {
                let leader = self.elect_leader_stake_based(round, leader_offset, epoch);
                leader
            }
        }
    }

    pub(crate) fn elect_leader_stake_based(
        &self,
        round: u32,
        offset: u32,
        epoch: Option<EpochId>,
    ) -> AuthorityIndex {
        let committee = epoch.map_or_else(
            || self.context.committee.clone(),
            |epoch| self.get_committee(epoch),
        );
        assert!((offset as usize) < committee.size());

        // To ensure that we elect different leaders for the same round (using
        // different offset) we are using the round number as seed to shuffle in
        // a weighted way the results, but skip based on the offset.
        let mut seed_bytes = [0u8; 32];
        seed_bytes[32 - 4..].copy_from_slice(&(round).to_le_bytes());
        let mut rng = StdRng::from_seed(seed_bytes);

        let choices = committee
            .authorities()
            .map(|(index, authority)| (index, authority.stake as f32))
            .collect::<Vec<_>>();

        let leader_index = *choices
            .choose_multiple_weighted(&mut rng, committee.size(), |item| item.1)
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
        let leader_schedule = LeaderSchedule::new(context, None);

        assert_eq!(
            leader_schedule.elect_leader(0, 0, None),
            AuthorityIndex::new_for_test(0)
        );
        assert_eq!(
            leader_schedule.elect_leader(1, 0, None),
            AuthorityIndex::new_for_test(1)
        );
        assert_eq!(
            leader_schedule.elect_leader(5, 0, None),
            AuthorityIndex::new_for_test(1)
        );
        // ensure we elect different leaders for the same round for the multi-leader case
        assert_ne!(
            leader_schedule.elect_leader_stake_based(1, 1, None),
            leader_schedule.elect_leader_stake_based(1, 2, None)
        );
    }

    #[tokio::test]
    async fn test_elect_leader_stake_based() {
        let context = Arc::new(Context::new_for_test(4).0);
        let leader_schedule = LeaderSchedule::new(context, None);

        assert_eq!(
            leader_schedule.elect_leader_stake_based(0, 0, None),
            AuthorityIndex::new_for_test(1)
        );
        assert_eq!(
            leader_schedule.elect_leader_stake_based(1, 0, None),
            AuthorityIndex::new_for_test(1)
        );
        assert_eq!(
            leader_schedule.elect_leader_stake_based(5, 0, None),
            AuthorityIndex::new_for_test(3)
        );
        // ensure we elect different leaders for the same round for the multi-leader case
        assert_ne!(
            leader_schedule.elect_leader_stake_based(1, 1, None),
            leader_schedule.elect_leader_stake_based(1, 2, None)
        );
    }
}
