// use fastcrypto::hash::{Blake2b256, HashFunction};
// use macros::add_signed_types;
// use rand::rngs::StdRng;
// use rand::SeedableRng;
// use serde::{Deserialize, Serialize};
// use shard::{ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey};
// use shard::{Scope, ScopedMessage};
// pub(crate) type DefaultHashFunction = Blake2b256;
// pub(crate) const DIGEST_LENGTH: usize = DefaultHashFunction::OUTPUT_SIZE;

// #[derive(Serialize, Deserialize, Debug, PartialEq)]
// pub struct ShardInput {
//     field1: u32,
//     field2: String,
// }
// add_signed_types!(ShardInput);

// #[test]
// fn test_signed() {
//     let test_struct = ShardInput {
//         field1: 42,
//         field2: "Hello, cex!".to_string(),
//     };

//     let mut rng = StdRng::from_seed([0; 32]);

//     let kp = ProtocolKeyPair::generate(&mut rng);

//     let signed_shard_input = test_struct.sign(&kp).unwrap();

//     signed_shard_input.verify_signature(&kp.public()).unwrap();
//     // assert_eq!(digest.0.len(), DIGEST_LENGH);

//     println!("{:?}", signed_shard_input);
// }
