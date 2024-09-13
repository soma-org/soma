// use fastcrypto::hash::{Blake2b256, HashFunction};
// use macros::add_digest_types;
// use serde::{Deserialize, Serialize};

// pub(crate) type DefaultHashFunction = Blake2b256;
// pub(crate) const DIGEST_LENGTH: usize = DefaultHashFunction::OUTPUT_SIZE;

// #[derive(Serialize, Deserialize, Debug, PartialEq)]
// struct TestStruct {
//     field1: u32,
//     field2: String,
// }
// add_digest_types!(TestStruct);

// #[test]
// fn test_add_digest_types() {
//     let test_struct = TestStruct {
//         field1: 42,
//         field2: "Hello, cex!".to_string(),
//     };

//     let digest = test_struct.digest();
//     println!("{}", digest);
//     assert_eq!(digest.0.len(), DIGEST_LENGTH);
//     let s = test_struct.serialize();
//     println!("{:?}", s);

//     println!("{:?}", digest);
// }
