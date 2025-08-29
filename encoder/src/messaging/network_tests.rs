#[cfg(test)]
mod tests {
    use crate::{
        messaging::{
            tonic::{
                external::EncoderExternalTonicManager,
                internal::{
                    ConnectionsInfo, EncoderInternalTonicClient, EncoderInternalTonicManager,
                },
            },
            EncoderExternalNetworkManager, EncoderExternalNetworkService,
            EncoderInternalNetworkClient, EncoderInternalNetworkManager,
            EncoderInternalNetworkService,
        },
        types::{
            commit::Commit,
            input::{Input, InputV1},
        },
    };
    use async_trait::async_trait;
    use bytes::Bytes;
    use fastcrypto::traits::KeyPair;
    use parking_lot::RwLock;
    use shared::parameters::Parameters;
    use shared::{
        crypto::keys::{EncoderKeyPair, EncoderPublicKey, PeerKeyPair, PeerPublicKey},
        error::ShardResult,
        metadata::Metadata,
        scope::Scope,
        signed::Signed,
        verified::Verified,
    };
    use soma_network::multiaddr::Multiaddr;
    use soma_tls::AllowPublicKeys;
    use std::{collections::BTreeMap, sync::Arc, time::Duration};
    use std::{
        collections::BTreeSet,
        net::{TcpListener, TcpStream},
    };
    use types::shard::ShardAuthToken;
    use types::shard::{ShardInput, ShardInputV1};
    use types::shard_networking::NetworkingInfo;

    fn get_available_local_address() -> Multiaddr {
        let host = "127.0.0.1";
        let port = get_available_port(host);
        format!("/ip4/{}/udp/{}", host, port).parse().unwrap()
    }
    fn get_available_port(host: &str) -> u16 {
        const MAX_PORT_RETRIES: u32 = 1000;

        for _ in 0..MAX_PORT_RETRIES {
            if let Ok(port) = get_ephemeral_port(host) {
                return port;
            }
        }

        panic!("Error: could not find an available port");
    }

    fn get_ephemeral_port(host: &str) -> std::io::Result<u16> {
        // Request a random available port from the OS
        let listener = TcpListener::bind((host, 0))?;
        let addr = listener.local_addr()?;

        // Create and accept a connection (which we'll promptly drop) in order to force the port
        // into the TIME_WAIT state, ensuring that the port will be reserved from some limited
        // amount of time (roughly 60s on some Linux systems)
        let _sender = TcpStream::connect(addr)?;
        let _incoming = listener.accept()?;

        Ok(addr.port())
    }
    struct MockInternalService {
        pub handle_send_commit: RwLock<Vec<(EncoderPublicKey, Bytes)>>,
        pub handle_send_commit_votes: RwLock<Vec<(EncoderPublicKey, Bytes)>>,
        pub handle_send_reveal: RwLock<Vec<(EncoderPublicKey, Bytes)>>,
        pub handle_send_reveal_votes: RwLock<Vec<(EncoderPublicKey, Bytes)>>,
        pub handle_send_scores: RwLock<Vec<(EncoderPublicKey, Bytes)>>,
    }

    impl MockInternalService {
        fn new() -> Self {
            Self {
                handle_send_commit: RwLock::new(vec![]),
                handle_send_commit_votes: RwLock::new(vec![]),
                handle_send_reveal: RwLock::new(vec![]),
                handle_send_reveal_votes: RwLock::new(vec![]),
                handle_send_scores: RwLock::new(vec![]),
            }
        }
    }

    #[async_trait]
    impl EncoderInternalNetworkService for MockInternalService {
        async fn handle_send_commit(
            &self,
            encoder: &EncoderPublicKey,
            commit_bytes: Bytes,
        ) -> ShardResult<()> {
            self.handle_send_commit
                .write()
                .push((encoder.to_owned(), commit_bytes));
            Ok(())
        }
        async fn handle_send_commit_votes(
            &self,
            encoder: &EncoderPublicKey,
            votes_bytes: Bytes,
        ) -> ShardResult<()> {
            self.handle_send_commit_votes
                .write()
                .push((encoder.to_owned(), votes_bytes));
            Ok(())
        }
        async fn handle_send_reveal(
            &self,
            encoder: &EncoderPublicKey,
            reveal_bytes: Bytes,
        ) -> ShardResult<()> {
            self.handle_send_reveal
                .write()
                .push((encoder.to_owned(), reveal_bytes));
            Ok(())
        }

        async fn handle_send_scores(
            &self,
            encoder: &EncoderPublicKey,
            scores_bytes: Bytes,
        ) -> ShardResult<()> {
            self.handle_send_scores
                .write()
                .push((encoder.to_owned(), scores_bytes));
            Ok(())
        }

        async fn handle_send_finality(
            &self,
            encoder: &EncoderPublicKey,
            finality_bytes: Bytes,
        ) -> ShardResult<()> {
            todo!()
        }
    }

    struct MockExternalService {
        pub handle_send_inputs: RwLock<Vec<(PeerPublicKey, Bytes)>>,
    }

    impl MockExternalService {
        fn new() -> Self {
            Self {
                handle_send_inputs: RwLock::new(vec![]),
            }
        }
    }

    #[async_trait]
    impl EncoderExternalNetworkService for MockExternalService {
        async fn handle_send_input(
            &self,
            peer: &PeerPublicKey,
            input_bytes: Bytes,
        ) -> ShardResult<()> {
            self.handle_send_inputs
                .write()
                .push((peer.to_owned(), input_bytes));
            Ok(())
        }
    }

    // TODO: reimplement tests when new_for_test in ShardAuthToken is implemented
    // #[tokio::test]
    // async fn external_tonic_happy_path() {
    //     let mut rng = rand::thread_rng();
    //     let server_tls_key = PeerKeyPair::generate(&mut rng);
    //     let client_tls_key = PeerKeyPair::generate(&mut rng);
    //     let encoder_key = EncoderKeyPair::generate(&mut rng);
    //     let address = get_available_local_address();

    //     let network_mapping = BTreeMap::from([(
    //         encoder_key.public(), // The public key identifying the encoder service
    //         (address.clone(), server_tls_key.public()), // Where and how (TLS identity) to reach it
    //     )]);
    //     let networking_info = NetworkingInfo::new(network_mapping);

    //     let parameters = Arc::new(Parameters::default());
    //     let allower = AllowPublicKeys::new(BTreeSet::from([client_tls_key.public().into_inner()]));
    //     let mut manager =
    //         EncoderExternalTonicManager::new(parameters.clone(), server_tls_key, address, allower);

    //     let service = Arc::new(MockExternalService::new());
    //     manager.start(service.clone()).await;

    //     let client = EncoderExternalTonicClient::new(
    //         networking_info,
    //         client_tls_key.clone(),
    //         parameters,
    //         100,
    //     );

    // let input = Input::V1(InputV1::new(ShardAuthToken::new_for_test()));
    // let inner_keypair = encoder_key.inner().copy();
    // let signed_input = Signed::new(input, Scope::Input, &inner_keypair.private()).unwrap();
    // let verified = Verified::from_trusted(signed_input.clone()).unwrap();

    //     client
    //         .send_input(&encoder_key.public(), &verified, Duration::from_secs(2))
    //         .await
    //         .unwrap();

    //     let (received_peer, received_input) =
    //         service.handle_send_inputs.read().first().cloned().unwrap();

    //     assert_eq!(client_tls_key.public(), received_peer);
    //     assert_eq!(bcs::to_bytes(&signed_input).unwrap(), received_input);
    // }
    // #[tokio::test]
    // async fn internal_tonic_happy_path() {
    //     let mut rng = rand::thread_rng();
    //     let server_tls_key = PeerKeyPair::generate(&mut rng);
    //     let client_tls_key = PeerKeyPair::generate(&mut rng);
    //     let server_encoder_key = EncoderKeyPair::generate(&mut rng);
    //     let client_encoder_key = EncoderKeyPair::generate(&mut rng);
    //     let address = get_available_local_address();

    //     let network_mapping = BTreeMap::from([(
    //         server_encoder_key.public(), // The public key identifying the encoder service
    //         (address.clone(), server_tls_key.public()), // Where and how (TLS identity) to reach it
    //     )]);
    //     let networking_info = NetworkingInfo::new(network_mapping);

    //     let connections_mapping =
    //         BTreeMap::from([(client_tls_key.public(), client_encoder_key.public())]);
    //     let connections_info = ConnectionsInfo::new(connections_mapping);

    //     let parameters = Arc::new(Parameters::default());
    //     let allower = AllowPublicKeys::new(BTreeSet::from([client_tls_key.public().into_inner()]));
    //     let mut manager = EncoderInternalTonicManager::new(
    //         networking_info.clone(),
    //         parameters.clone(),
    //         server_tls_key,
    //         address,
    //         allower,
    //         connections_info,
    //     );

    //     let service = Arc::new(MockInternalService::new());
    //     manager.start(service.clone()).await;

    //     let client = EncoderInternalTonicClient::new(
    //         networking_info,
    //         client_tls_key.clone(),
    //         parameters,
    //         100,
    //     );

    // let commit: Commit = Commit::new_v1(
    //     ShardAuthToken::new_for_test(),
    //     client_encoder_key.public(),
    //     None,
    //     Metadata::default(),
    // );
    // let inner_keypair = client_encoder_key.inner().copy();

    // let signed_commit =
    //     Signed::new(commit, Scope::Commit, &inner_keypair.private()).unwrap();
    // let verified = Verified::from_trusted(signed_commit.clone()).unwrap();

    //     client
    //         .send_commit(
    //             &server_encoder_key.public(),
    //             &verified,
    //             Duration::from_secs(2),
    //         )
    //         .await
    //         .unwrap();

    //     let (from, received_commit) = service.handle_send_commit.read().first().cloned().unwrap();

    //     assert_eq!(client_encoder_key.public(), from);
    //     assert_eq!(bcs::to_bytes(&signed_commit).unwrap(), received_commit);
    // }
}
