//! Tests for Proof of Possession (PoP) and duplicate validator metadata checks.
//!
//! Covers:
//! - PoP verification in genesis (GenesisValidatorInfo::validate)
//! - PoP verification in request_add_validator
//! - PoP requirement in stage_next_epoch_metadata (protocol key change)
//! - Duplicate metadata detection (is_duplicate, effectuate_staged_metadata)

#[cfg(test)]
#[allow(clippy::module_inception, clippy::unwrap_used, clippy::expect_used)]
mod validator_pop_tests {
    use crate::{
        base::SomaAddress,
        config::genesis_config::SHANNONS_PER_SOMA,
        crypto::{
            AuthorityKeyPair, AuthorityPublicKeyBytes, NetworkKeyPair,
            generate_proof_of_possession, get_key_pair_from_rng,
        },
        effects::ExecutionFailureStatus,
        object::ObjectID,
        system_state::test_utils::{
            create_test_system_state, create_validator_for_testing,
            create_validator_for_testing_with_seed, create_validators_with_stakes,
        },
        transaction::UpdateValidatorMetadataArgs,
        validator_info::{GenesisValidatorInfo, ValidatorInfo},
    };
    use fastcrypto::traits::{KeyPair, ToFromBytes};
    use rand::{SeedableRng, rngs::StdRng};

    // =========================================================================
    // Genesis PoP validation
    // =========================================================================

    #[test]
    fn test_genesis_validator_info_valid_pop() {
        // A validator info with a correctly generated PoP should validate.
        let mut rng = StdRng::from_seed([10; 32]);
        let (address, _): (SomaAddress, fastcrypto::ed25519::Ed25519KeyPair) =
            get_key_pair_from_rng(&mut rng);
        let protocol_kp = AuthorityKeyPair::generate(&mut rng);
        let network_kp = NetworkKeyPair::generate(&mut rng);
        let worker_kp = NetworkKeyPair::generate(&mut rng);

        let pop = generate_proof_of_possession(&protocol_kp, address);

        let info = GenesisValidatorInfo {
            info: ValidatorInfo {
                account_address: address,
                protocol_key: AuthorityPublicKeyBytes::from(protocol_kp.public()),
                network_key: network_kp.public(),
                worker_key: worker_kp.public(),
                proof_of_possession: pop.as_ref().to_vec(),
                network_address: "/ip4/127.0.0.1/tcp/8080".parse().unwrap(),
                p2p_address: "/ip4/127.0.0.1/tcp/8081".parse().unwrap(),
                primary_address: "/ip4/127.0.0.1/tcp/8082".parse().unwrap(),
                proxy_address: "/ip4/127.0.0.1/tcp/8083/http".parse().unwrap(),
                commission_rate: 500,
            },
        };

        assert!(info.validate().is_ok(), "Valid PoP should pass genesis validation");
    }

    #[test]
    fn test_genesis_validator_info_wrong_pop() {
        // A validator info with a PoP generated for a different address should fail.
        let mut rng = StdRng::from_seed([11; 32]);
        let (address, _): (SomaAddress, fastcrypto::ed25519::Ed25519KeyPair) =
            get_key_pair_from_rng(&mut rng);
        let (wrong_address, _): (SomaAddress, fastcrypto::ed25519::Ed25519KeyPair) =
            get_key_pair_from_rng(&mut rng);
        let protocol_kp = AuthorityKeyPair::generate(&mut rng);
        let network_kp = NetworkKeyPair::generate(&mut rng);
        let worker_kp = NetworkKeyPair::generate(&mut rng);

        // Generate PoP for the WRONG address
        let pop = generate_proof_of_possession(&protocol_kp, wrong_address);

        let info = GenesisValidatorInfo {
            info: ValidatorInfo {
                account_address: address,
                protocol_key: AuthorityPublicKeyBytes::from(protocol_kp.public()),
                network_key: network_kp.public(),
                worker_key: worker_kp.public(),
                proof_of_possession: pop.as_ref().to_vec(),
                network_address: "/ip4/127.0.0.1/tcp/8080".parse().unwrap(),
                p2p_address: "/ip4/127.0.0.1/tcp/8081".parse().unwrap(),
                primary_address: "/ip4/127.0.0.1/tcp/8082".parse().unwrap(),
                proxy_address: "/ip4/127.0.0.1/tcp/8083/http".parse().unwrap(),
                commission_rate: 500,
            },
        };

        let result = info.validate();
        assert!(result.is_err(), "PoP for wrong address should fail genesis validation");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Proof of possession verification failed"),
            "Error should mention PoP failure, got: {}",
            err
        );
    }

    #[test]
    fn test_genesis_validator_info_garbage_pop() {
        // Garbage bytes in proof_of_possession should fail validation.
        let mut rng = StdRng::from_seed([12; 32]);
        let (address, _): (SomaAddress, fastcrypto::ed25519::Ed25519KeyPair) =
            get_key_pair_from_rng(&mut rng);
        let protocol_kp = AuthorityKeyPair::generate(&mut rng);
        let network_kp = NetworkKeyPair::generate(&mut rng);
        let worker_kp = NetworkKeyPair::generate(&mut rng);

        let info = GenesisValidatorInfo {
            info: ValidatorInfo {
                account_address: address,
                protocol_key: AuthorityPublicKeyBytes::from(protocol_kp.public()),
                network_key: network_kp.public(),
                worker_key: worker_kp.public(),
                proof_of_possession: vec![0xDE, 0xAD, 0xBE, 0xEF],
                network_address: "/ip4/127.0.0.1/tcp/8080".parse().unwrap(),
                p2p_address: "/ip4/127.0.0.1/tcp/8081".parse().unwrap(),
                primary_address: "/ip4/127.0.0.1/tcp/8082".parse().unwrap(),
                proxy_address: "/ip4/127.0.0.1/tcp/8083/http".parse().unwrap(),
                commission_rate: 500,
            },
        };

        let result = info.validate();
        assert!(result.is_err(), "Garbage PoP bytes should fail genesis validation");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Invalid proof of possession bytes"),
            "Error should mention invalid bytes, got: {}",
            err
        );
    }

    // =========================================================================
    // request_add_validator PoP checks
    // =========================================================================

    /// Helper: make a valid add-validator call using real keys and a valid PoP.
    fn make_add_validator_args(
        rng: &mut StdRng,
    ) -> (
        SomaAddress, // signer
        Vec<u8>,     // pubkey_bytes
        Vec<u8>,     // network_pubkey_bytes
        Vec<u8>,     // worker_pubkey_bytes
        Vec<u8>,     // proof_of_possession_bytes
        Vec<u8>,     // net_address (BCS serialized)
        Vec<u8>,     // p2p_address
        Vec<u8>,     // primary_address
        Vec<u8>,     // proxy_address
    ) {
        let (signer, _): (SomaAddress, fastcrypto::ed25519::Ed25519KeyPair) =
            get_key_pair_from_rng(rng);
        let protocol_kp = AuthorityKeyPair::generate(rng);
        let network_kp = NetworkKeyPair::generate(rng);
        let worker_kp = NetworkKeyPair::generate(rng);
        let pop = generate_proof_of_possession(&protocol_kp, signer);

        let bcs_addr = |s: &str| bcs::to_bytes(&s.to_string()).unwrap();

        (
            signer,
            protocol_kp.public().as_bytes().to_vec(),
            network_kp.public().to_bytes().to_vec(),
            worker_kp.public().to_bytes().to_vec(),
            pop.as_ref().to_vec(),
            bcs_addr("/ip4/127.0.0.1/tcp/9000"),
            bcs_addr("/ip4/127.0.0.1/tcp/9001"),
            bcs_addr("/ip4/127.0.0.1/tcp/9002"),
            bcs_addr("/ip4/127.0.0.1/tcp/9003/http"),
        )
    }

    #[test]
    fn test_request_add_validator_valid_pop() {
        let mut system_state =
            create_test_system_state(create_validators_with_stakes(vec![100]), 1_000_000, 0);

        let mut rng = StdRng::from_seed([20; 32]);
        let (signer, pubkey, net_pub, worker_pub, pop, net, p2p, primary, proxy) =
            make_add_validator_args(&mut rng);

        let result = system_state.request_add_validator(
            signer,
            pubkey,
            net_pub,
            worker_pub,
            pop,
            net,
            p2p,
            primary,
            proxy,
            ObjectID::random(),
        );

        assert!(result.is_ok(), "Valid PoP should succeed: {:?}", result.err());
    }

    #[test]
    fn test_request_add_validator_invalid_pop() {
        let mut system_state =
            create_test_system_state(create_validators_with_stakes(vec![100]), 1_000_000, 0);

        let mut rng = StdRng::from_seed([21; 32]);
        let (signer, pubkey, net_pub, worker_pub, _valid_pop, net, p2p, primary, proxy) =
            make_add_validator_args(&mut rng);

        // Generate a PoP for a DIFFERENT address
        let protocol_kp = AuthorityKeyPair::generate(&mut rng);
        let (wrong_addr, _): (SomaAddress, fastcrypto::ed25519::Ed25519KeyPair) =
            get_key_pair_from_rng(&mut rng);
        let wrong_pop = generate_proof_of_possession(&protocol_kp, wrong_addr);

        let result = system_state.request_add_validator(
            signer,
            pubkey,
            net_pub,
            worker_pub,
            wrong_pop.as_ref().to_vec(),
            net,
            p2p,
            primary,
            proxy,
            ObjectID::random(),
        );

        match result {
            Err(ExecutionFailureStatus::InvalidProofOfPossession { reason }) => {
                assert!(
                    reason.contains("PoP verification failed"),
                    "Should mention verification failure, got: {}",
                    reason
                );
            }
            other => panic!("Expected InvalidProofOfPossession, got: {:?}", other),
        }
    }

    #[test]
    fn test_request_add_validator_garbage_pop() {
        let mut system_state =
            create_test_system_state(create_validators_with_stakes(vec![100]), 1_000_000, 0);

        let mut rng = StdRng::from_seed([22; 32]);
        let (signer, pubkey, net_pub, worker_pub, _valid_pop, net, p2p, primary, proxy) =
            make_add_validator_args(&mut rng);

        let result = system_state.request_add_validator(
            signer,
            pubkey,
            net_pub,
            worker_pub,
            vec![0xBA, 0xD0], // garbage bytes
            net,
            p2p,
            primary,
            proxy,
            ObjectID::random(),
        );

        match result {
            Err(ExecutionFailureStatus::InvalidProofOfPossession { reason }) => {
                assert!(
                    reason.contains("Invalid PoP signature bytes"),
                    "Should mention invalid bytes, got: {}",
                    reason
                );
            }
            other => panic!("Expected InvalidProofOfPossession, got: {:?}", other),
        }
    }

    // =========================================================================
    // stage_next_epoch_metadata PoP checks
    // =========================================================================

    #[test]
    fn test_stage_metadata_protocol_key_with_valid_pop() {
        // Changing protocol key with a valid PoP should succeed.
        let mut rng = StdRng::from_seed([30; 32]);
        let addr = SomaAddress::random();
        let mut validator = create_validator_for_testing(addr, 100 * SHANNONS_PER_SOMA);

        // Generate a new protocol key and valid PoP
        let new_protocol_kp = AuthorityKeyPair::generate(&mut rng);
        let pop = generate_proof_of_possession(&new_protocol_kp, addr);

        let args = UpdateValidatorMetadataArgs {
            next_epoch_protocol_pubkey: Some(new_protocol_kp.public().as_bytes().to_vec()),
            next_epoch_proof_of_possession: Some(pop.as_ref().to_vec()),
            ..Default::default()
        };

        let result = validator.stage_next_epoch_metadata(&args, &[]);
        assert!(result.is_ok(), "Valid PoP with key change should succeed: {:?}", result.err());
        assert!(
            validator.metadata.next_epoch_protocol_pubkey.is_some(),
            "Protocol key should be staged"
        );
        assert!(
            validator.metadata.next_epoch_proof_of_possession.is_some(),
            "PoP should be staged"
        );
    }

    #[test]
    fn test_stage_metadata_protocol_key_missing_pop() {
        // Changing protocol key without providing PoP should fail.
        let mut rng = StdRng::from_seed([31; 32]);
        let addr = SomaAddress::random();
        let mut validator = create_validator_for_testing(addr, 100 * SHANNONS_PER_SOMA);

        let new_protocol_kp = AuthorityKeyPair::generate(&mut rng);

        let args = UpdateValidatorMetadataArgs {
            next_epoch_protocol_pubkey: Some(new_protocol_kp.public().as_bytes().to_vec()),
            next_epoch_proof_of_possession: None, // Missing!
            ..Default::default()
        };

        let result = validator.stage_next_epoch_metadata(&args, &[]);
        match result {
            Err(ExecutionFailureStatus::MissingProofOfPossession) => {} // expected
            other => panic!("Expected MissingProofOfPossession, got: {:?}", other),
        }
    }

    #[test]
    fn test_stage_metadata_protocol_key_wrong_pop() {
        // Changing protocol key with a PoP for a different address should fail.
        let mut rng = StdRng::from_seed([32; 32]);
        let addr = SomaAddress::random();
        let mut validator = create_validator_for_testing(addr, 100 * SHANNONS_PER_SOMA);

        let new_protocol_kp = AuthorityKeyPair::generate(&mut rng);
        let wrong_addr = SomaAddress::random();
        let wrong_pop = generate_proof_of_possession(&new_protocol_kp, wrong_addr);

        let args = UpdateValidatorMetadataArgs {
            next_epoch_protocol_pubkey: Some(new_protocol_kp.public().as_bytes().to_vec()),
            next_epoch_proof_of_possession: Some(wrong_pop.as_ref().to_vec()),
            ..Default::default()
        };

        let result = validator.stage_next_epoch_metadata(&args, &[]);
        match result {
            Err(ExecutionFailureStatus::InvalidProofOfPossession { .. }) => {} // expected
            other => panic!("Expected InvalidProofOfPossession, got: {:?}", other),
        }
    }

    // =========================================================================
    // Duplicate metadata checks
    // =========================================================================

    #[test]
    fn test_is_duplicate_detects_same_protocol_key() {
        let addr1 = SomaAddress::random();
        let addr2 = SomaAddress::random();
        // Use the same seed -> same keys
        let v1 = create_validator_for_testing_with_seed(addr1, 100, [50; 32], 20000);
        let v2 = create_validator_for_testing_with_seed(addr2, 100, [50; 32], 20100);

        let dup = v1.is_duplicate(&v2);
        assert!(dup.is_some(), "Same protocol key should be detected as duplicate");
        assert!(
            dup.unwrap().contains("protocol_pubkey"),
            "Duplicate field should be protocol_pubkey"
        );
    }

    #[test]
    fn test_is_duplicate_different_keys_not_duplicate() {
        let addr1 = SomaAddress::random();
        let addr2 = SomaAddress::random();
        // Different seeds -> different keys
        let v1 = create_validator_for_testing_with_seed(addr1, 100, [60; 32], 21000);
        let v2 = create_validator_for_testing_with_seed(addr2, 100, [61; 32], 21100);

        assert!(
            v1.is_duplicate(&v2).is_none(),
            "Validators with different keys should not be duplicates"
        );
    }

    #[test]
    fn test_request_add_validator_duplicate_key_rejected() {
        // Create a system state with one validator, then try to add another with the same key.
        let existing = create_validator_for_testing_with_seed(
            SomaAddress::random(),
            100 * SHANNONS_PER_SOMA,
            [70; 32],
            22000,
        );
        let mut system_state = create_test_system_state(vec![existing.clone()], 1_000_000, 0);

        // Generate a new validator with the SAME protocol key seed
        let mut rng = StdRng::from_seed([70; 32]);
        let new_addr = SomaAddress::random();
        let protocol_kp = AuthorityKeyPair::generate(&mut rng);

        // Different network/worker keys
        let mut rng2 = StdRng::from_seed([71; 32]);
        let network_kp = NetworkKeyPair::generate(&mut rng2);
        let worker_kp = NetworkKeyPair::generate(&mut rng2);

        let pop = generate_proof_of_possession(&protocol_kp, new_addr);

        let bcs_addr = |s: &str| bcs::to_bytes(&s.to_string()).unwrap();

        let result = system_state.request_add_validator(
            new_addr,
            protocol_kp.public().as_bytes().to_vec(),
            network_kp.public().to_bytes().to_vec(),
            worker_kp.public().to_bytes().to_vec(),
            pop.as_ref().to_vec(),
            bcs_addr("/ip4/127.0.0.1/tcp/23000"),
            bcs_addr("/ip4/127.0.0.1/tcp/23001"),
            bcs_addr("/ip4/127.0.0.1/tcp/23002"),
            bcs_addr("/ip4/127.0.0.1/tcp/23003/http"),
            ObjectID::random(),
        );

        match result {
            Err(ExecutionFailureStatus::DuplicateValidatorMetadata { field }) => {
                assert!(
                    field.contains("protocol_pubkey"),
                    "Should identify protocol_pubkey as duplicate, got: {}",
                    field
                );
            }
            other => panic!("Expected DuplicateValidatorMetadata, got: {:?}", other),
        }
    }

    // =========================================================================
    // effectuate_staged_metadata applies PoP
    // =========================================================================

    #[test]
    fn test_effectuate_applies_staged_pop() {
        // When protocol key is staged with PoP, effectuate should apply both.
        let mut rng = StdRng::from_seed([40; 32]);
        let addr = SomaAddress::random();
        let mut validator = create_validator_for_testing(addr, 100 * SHANNONS_PER_SOMA);

        let old_pubkey = validator.metadata.protocol_pubkey.clone();

        // Stage a new protocol key with valid PoP
        let new_protocol_kp = AuthorityKeyPair::generate(&mut rng);
        let pop = generate_proof_of_possession(&new_protocol_kp, addr);
        validator.metadata.next_epoch_protocol_pubkey = Some(new_protocol_kp.public().to_owned());
        validator.metadata.next_epoch_proof_of_possession = Some(pop.clone());

        // Effectuate
        validator.effectuate_staged_metadata();

        assert_ne!(
            validator.metadata.protocol_pubkey, old_pubkey,
            "Protocol key should have changed"
        );
        assert_eq!(
            validator.metadata.protocol_pubkey,
            new_protocol_kp.public().to_owned(),
            "Protocol key should match the staged key"
        );
        assert_eq!(
            validator.metadata.proof_of_possession, pop,
            "PoP should be updated to the staged PoP"
        );
        // Staged fields should be cleared
        assert!(validator.metadata.next_epoch_protocol_pubkey.is_none());
        assert!(validator.metadata.next_epoch_proof_of_possession.is_none());
    }
}
