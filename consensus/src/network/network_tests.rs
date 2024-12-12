use std::{sync::Arc, time::Duration};

use bytes::Bytes;
use parking_lot::Mutex;
use tokio::time::sleep;
use types::consensus::{
    block::{Round, TestBlock, VerifiedBlock},
    context::Context,
};
use types::crypto::NetworkKeyPair;

use super::{
    test_network::TestService, tonic_network::TonicManager, NetworkClient, NetworkManager,
};

struct TonicManagerBuilder {}

impl TonicManagerBuilder {
    fn build(&self, context: Arc<Context>, network_keypair: NetworkKeyPair) -> TonicManager {
        TonicManager::new(context, network_keypair)
    }
}

fn block_for_round(round: Round) -> Bytes {
    Bytes::from(vec![round as u8; 16])
}

fn service_with_own_blocks() -> Arc<Mutex<TestService>> {
    let service = Arc::new(Mutex::new(TestService::new()));
    {
        let mut service = service.lock();
        let own_blocks = (0..=100u8)
            .map(|i| block_for_round(i as Round))
            .collect::<Vec<_>>();
        service.add_own_blocks(own_blocks);
    }
    service
}

#[tokio::test]
async fn send_and_receive_blocks_with_auth() {
    let manager_builder = TonicManagerBuilder {};

    let (context, keys) = Context::new_for_test(4);

    let context_0 = Arc::new(
        context
            .clone()
            .with_authority_index(context.committee.to_authority_index(0).unwrap()),
    );
    let mut manager_0 = manager_builder.build(context_0.clone(), keys[0].0.clone());
    let client_0 = <TonicManager as NetworkManager<Mutex<TestService>>>::client(&manager_0);
    let service_0 = service_with_own_blocks();
    manager_0.install_service(service_0.clone()).await;

    let context_1 = Arc::new(
        context
            .clone()
            .with_authority_index(context.committee.to_authority_index(1).unwrap()),
    );
    let mut manager_1 = manager_builder.build(context_1.clone(), keys[1].0.clone());
    let client_1 = <TonicManager as NetworkManager<Mutex<TestService>>>::client(&manager_1);
    let service_1 = service_with_own_blocks();
    manager_1.install_service(service_1.clone()).await;

    // Wait for anemo to initialize.
    sleep(Duration::from_secs(5)).await;

    // Test that servers can receive client RPCs.
    let test_block_0 = VerifiedBlock::new_for_test(TestBlock::new(9, 0).build());
    client_0
        .send_block(
            context.committee.to_authority_index(1).unwrap(),
            &test_block_0,
            Duration::from_secs(5),
        )
        .await
        .unwrap();
    let test_block_1 = VerifiedBlock::new_for_test(TestBlock::new(9, 1).build());
    client_1
        .send_block(
            context.committee.to_authority_index(0).unwrap(),
            &test_block_1,
            Duration::from_secs(5),
        )
        .await
        .unwrap();

    assert_eq!(service_0.lock().handle_send_block.len(), 1);
    assert_eq!(service_0.lock().handle_send_block[0].0.value(), 1);
    assert_eq!(
        service_0.lock().handle_send_block[0].1,
        test_block_1.serialized(),
    );
    assert_eq!(service_1.lock().handle_send_block.len(), 1);
    assert_eq!(service_1.lock().handle_send_block[0].0.value(), 0);
    assert_eq!(
        service_1.lock().handle_send_block[0].1,
        test_block_0.serialized(),
    );

    // `Committee` is generated with the same random seed in Context::new_for_test(),
    // so the first 4 authorities are the same.
    let (context_4, keys_4) = Context::new_for_test(5);
    let context_4 = Arc::new(
        context_4
            .clone()
            .with_authority_index(context_4.committee.to_authority_index(4).unwrap()),
    );
    let mut manager_4 = manager_builder.build(context_4.clone(), keys_4[4].0.clone());
    let client_4 = <TonicManager as NetworkManager<Mutex<TestService>>>::client(&manager_4);
    let service_4 = service_with_own_blocks();
    manager_4.install_service(service_4.clone()).await;

    // client_4 should not be able to reach service_0 or service_1, because of the
    // AllowedPeers filter.
    let test_block_2 = VerifiedBlock::new_for_test(TestBlock::new(9, 2).build());
    assert!(client_4
        .send_block(
            context.committee.to_authority_index(0).unwrap(),
            &test_block_2,
            Duration::from_secs(5),
        )
        .await
        .is_err());
    let test_block_3 = VerifiedBlock::new_for_test(TestBlock::new(9, 3).build());
    assert!(client_4
        .send_block(
            context.committee.to_authority_index(1).unwrap(),
            &test_block_3,
            Duration::from_secs(5),
        )
        .await
        .is_err());
}
