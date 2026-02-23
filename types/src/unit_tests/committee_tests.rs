use crate::committee::{
    Committee, CommitteeTrait, QUORUM_THRESHOLD, TOTAL_VOTING_POWER, VALIDITY_THRESHOLD,
};
use fastcrypto::traits::KeyPair;

#[test]
fn test_committee_creation() {
    // new_simple_test_committee_of_size internally validates that
    // total voting power == 10000 and the committee is non-empty.
    let (committee, key_pairs) = Committee::new_simple_test_committee_of_size(4);

    assert_eq!(committee.num_members(), 4);
    assert_eq!(key_pairs.len(), 4);

    // Total voting power across all members must equal TOTAL_VOTING_POWER
    let total: u64 = committee.stakes().sum();
    assert_eq!(total, TOTAL_VOTING_POWER);
}

#[test]
fn test_committee_simple_test_creation() {
    for size in [1, 4, 7] {
        let (committee, key_pairs) = Committee::new_simple_test_committee_of_size(size);
        assert_eq!(committee.num_members(), size);
        assert_eq!(key_pairs.len(), size);

        // Verify total voting power is always TOTAL_VOTING_POWER
        let total: u64 = committee.stakes().sum();
        assert_eq!(
            total, TOTAL_VOTING_POWER,
            "Total voting power should be {} for committee of size {}",
            TOTAL_VOTING_POWER, size
        );

        // Verify every member has non-zero weight
        for (name, _) in committee.members() {
            assert!(
                committee.weight(name) > 0,
                "Every committee member should have positive weight"
            );
        }
    }
}

#[test]
fn test_committee_quorum_threshold() {
    let (committee, _) = Committee::new_simple_test_committee_of_size(4);
    assert_eq!(
        committee.quorum_threshold(),
        QUORUM_THRESHOLD,
        "quorum_threshold() should return QUORUM_THRESHOLD ({})",
        QUORUM_THRESHOLD
    );
    assert_eq!(committee.quorum_threshold(), 6_667);
}

#[test]
fn test_committee_validity_threshold() {
    let (committee, _) = Committee::new_simple_test_committee_of_size(4);
    assert_eq!(
        committee.validity_threshold(),
        VALIDITY_THRESHOLD,
        "validity_threshold() should return VALIDITY_THRESHOLD ({})",
        VALIDITY_THRESHOLD
    );
    assert_eq!(committee.validity_threshold(), 3_334);
}

#[test]
fn test_committee_weight_lookup() {
    let (committee, key_pairs) = Committee::new_simple_test_committee_of_size(4);

    // Each key pair's public key corresponds to an authority name
    for kp in &key_pairs {
        let name = crate::base::AuthorityName::from(kp.public());
        let weight = committee.weight(&name);
        assert!(weight > 0, "Known authority should have positive weight");
    }

    // Verify total matches
    let total: u64 = key_pairs
        .iter()
        .map(|kp| {
            let name = crate::base::AuthorityName::from(kp.public());
            committee.weight(&name)
        })
        .sum();
    assert_eq!(total, TOTAL_VOTING_POWER);

    // Unknown authority should have weight 0
    let unknown = crate::base::AuthorityName::default();
    // The default may or may not be in the committee; check by membership
    if !committee.authority_exists(&unknown) {
        assert_eq!(committee.weight(&unknown), 0, "Unknown authority should have zero weight");
    }
}

#[test]
fn test_committee_epoch() {
    // new_simple_test_committee_of_size creates committee with epoch 0
    let (committee, _) = Committee::new_simple_test_committee_of_size(4);
    assert_eq!(committee.epoch(), 0, "Default test committee should have epoch 0");

    // Verify with a specific epoch via the normalized constructor
    use crate::committee::Authority;
    use crate::committee::get_available_local_address;
    use crate::crypto::{AuthorityKeyPair, NetworkKeyPair, ProtocolKeyPair};
    use fastcrypto::traits::KeyPair;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::collections::BTreeMap;

    let mut rng = StdRng::from_seed([1; 32]);
    let kp = AuthorityKeyPair::generate(&mut rng);
    let protocol_kp = ProtocolKeyPair::generate(&mut rng);
    let network_kp = NetworkKeyPair::generate(&mut rng);
    let name = crate::base::AuthorityName::from(kp.public());

    let mut voting = BTreeMap::new();
    voting.insert(name, TOTAL_VOTING_POWER);

    let mut authorities = BTreeMap::new();
    authorities.insert(
        name,
        Authority {
            stake: TOTAL_VOTING_POWER,
            address: get_available_local_address(),
            hostname: "test_host".to_string(),
            authority_key: kp.public().clone(),
            protocol_key: protocol_kp.public(),
            network_key: network_kp.public(),
        },
    );

    let target_epoch = 42u64;
    let committee = Committee::new(target_epoch, voting, authorities);
    assert_eq!(
        committee.epoch(),
        target_epoch,
        "Committee should return the epoch it was created with"
    );
}
