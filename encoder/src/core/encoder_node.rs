use std::{future::Future, path::Path, sync::Arc};

use objects::{
    networking::{
        http_network::{ObjectHttpClient, ObjectHttpManager},
        ObjectNetworkManager, ObjectNetworkService,
    },
    storage::filesystem::FilesystemObjectStorage,
};
use quick_cache::sync::Cache;
use shared::{
    crypto::keys::{EncoderKeyPair, PeerKeyPair},
    digest::Digest,
    entropy::EntropyVDF,
};
use soma_network::multiaddr::Multiaddr;
use soma_tls::AllowPublicKeys;
use tokio::sync::Semaphore;

use crate::{
    actors::{
        pipelines::{
            commit::CommitProcessor, commit_votes::CommitVotesProcessor,
            evaluation::EvaluationProcessor, reveal::RevealProcessor,
            reveal_votes::RevealVotesProcessor, scores::ScoresProcessor,
        },
        workers::{
            compression::CompressionProcessor, downloader, encryption::EncryptionProcessor,
            model::ModelProcessor, storage::StorageProcessor, vdf::VDFProcessor,
        },
        ActorManager,
    },
    compression::zstd_compressor::ZstdCompressor,
    datastore::mem_store::MemStore,
    encryption::aes_encryptor::Aes256Ctr64LEEncryptor,
    intelligence::model::python::PythonInterpreter,
    messaging::{
        internal_service::EncoderInternalService,
        tonic::{
            internal::{ConnectionsInfo, EncoderInternalTonicClient, EncoderInternalTonicManager},
            NetworkingInfo,
        },
        EncoderInternalNetworkManager,
    },
    types::{context::Context, parameters::Parameters, shard_verifier},
};

use self::{
    downloader::Downloader,
    shard_verifier::{ShardAuthToken, ShardVerifier, VerificationStatus},
};

use super::{
    internal_broadcaster::Broadcaster, pipeline_dispatcher::InternalPipelineDispatcher,
    shard_tracker::ShardTracker,
};

#[cfg(msim)]
use msim::task::NodeId;
#[cfg(msim)]
use simulator::SimState;

// pub struct Encoder(EncoderNode<ActorInternalPipelineDispatcher<EncoderTonicClient, PythonModule, FilesystemObjectStorage, ObjectHttpClient>, EncoderTonicManager>);

// impl Encoder {
//     pub async fn start(
//         context: Arc<Context>,
//         network_keypair: NetworkKeyPair,
//         protocol_keypair: ProtocolKeyPair,
//         project_root: &Path,
//         entry_point: &Path,
//     ) -> Self {
//         let encoder_node: EncoderNode<ActorInternalPipelineDispatcher<EncoderTonicClient, PythonModule, FilesystemObjectStorage, ObjectHttpClient>, EncoderTonicManager> =
//             EncoderNode::start(
//                 context,
//                 network_keypair,
//                 protocol_keypair,
//                 project_root,
//                 entry_point,
//             )
//             .await;
//         Self(encoder_node)
//     }
//     pub async fn stop(self) {
//         self.0.stop().await;
//     }
// }

pub struct EncoderNode {
    network_manager: EncoderInternalTonicManager,

    #[cfg(msim)]
    sim_state: SimState,
}

impl EncoderNode {
    pub async fn start(
        context: Context,
        encoder_keypair: EncoderKeyPair,
        networking_info: NetworkingInfo,
        parameters: Arc<Parameters>,
        object_parameters: Arc<objects::parameters::Parameters>,
        peer_keypair: PeerKeyPair,
        address: Multiaddr,
        object_address: Multiaddr,
        allower: AllowPublicKeys,
        connections_info: ConnectionsInfo,
        project_root: &Path,
        entry_point: &Path,
    ) -> Self {
        let mut network_manager = EncoderInternalTonicManager::new(
            networking_info,
            parameters,
            peer_keypair.clone(),
            address,
            allower.clone(),
            connections_info,
        );

        let messaging_client = <EncoderInternalTonicManager as EncoderInternalNetworkManager<
            EncoderInternalService<
                InternalPipelineDispatcher<
                    EncoderInternalTonicClient,
                    ObjectHttpClient,
                    FilesystemObjectStorage,
                >,
            >,
        >>::client(&network_manager);

        let object_storage = Arc::new(FilesystemObjectStorage::new("base_path"));
        let object_network_service: ObjectNetworkService<FilesystemObjectStorage> =
            ObjectNetworkService::new(object_storage.clone());

        let mut object_network_manager = <ObjectHttpManager as ObjectNetworkManager<
            FilesystemObjectStorage,
        >>::new(peer_keypair, object_parameters, allower)
        .unwrap();

        object_network_manager
            .start(&object_address, object_network_service)
            .await;

        let object_client =
            <ObjectHttpManager as ObjectNetworkManager<FilesystemObjectStorage>>::client(
                &object_network_manager,
            );

        let encoder_keypair = Arc::new(encoder_keypair);

        let default_buffer = 100_usize;
        let default_concurrency = 100_usize;

        let download_processor = Downloader::new(
            default_concurrency,
            object_client.clone(),
            object_storage.clone(),
        );
        let downloader_manager = ActorManager::new(default_buffer, download_processor);
        let downloader_handle = downloader_manager.handle();

        let encryptor_processor: EncryptionProcessor<Aes256Ctr64LEEncryptor> =
            EncryptionProcessor::new(Arc::new(Aes256Ctr64LEEncryptor::new()));
        let encryptor_manager = ActorManager::new(default_buffer, encryptor_processor);
        let encryptor_handle = encryptor_manager.handle();

        let compressor_processor = CompressionProcessor::new(Arc::new(ZstdCompressor::new()));
        let compressor_manager = ActorManager::new(default_buffer, compressor_processor);
        let compressor_handle = compressor_manager.handle();

        let python_interpreter = PythonInterpreter::new(project_root).unwrap();
        let model = python_interpreter.new_module(entry_point).unwrap();

        let model_processor = ModelProcessor::new(model, None);
        let model_manager = ActorManager::new(default_buffer, model_processor);
        let model_handle = model_manager.handle();

        let storage_processor = StorageProcessor::new(object_storage, None);
        let storage_manager = ActorManager::new(default_buffer, storage_processor);
        let storage_handle = storage_manager.handle();

        let vdf = EntropyVDF::new(1);
        let vdf_processor = VDFProcessor::new(vdf, 1);
        let vdf_handle = ActorManager::new(1, vdf_processor).handle();
        let store = Arc::new(MemStore::new());

        let broadcaster = Arc::new(Broadcaster::new(
            messaging_client.clone(),
            Arc::new(Semaphore::new(default_concurrency)),
        ));

        let evaluation_processor = EvaluationProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            storage_handle.clone(),
        );
        let evaluation_handle = ActorManager::new(default_buffer, evaluation_processor).handle();

        let shard_tracker = Arc::new(ShardTracker::new(
            Arc::new(Semaphore::new(default_concurrency)),
            broadcaster.clone(),
            store.clone(),
            encoder_keypair.clone(),
            evaluation_handle,
        ));

        let commit_processor = CommitProcessor::new(
            store.clone(),
            shard_tracker.clone(),
            downloader_handle.clone(),
        );

        let commit_votes_processor =
            CommitVotesProcessor::new(store.clone(), shard_tracker.clone());

        let reveal_processor = RevealProcessor::new(store.clone(), shard_tracker.clone());
        let reveal_votes_processor =
            RevealVotesProcessor::new(store.clone(), shard_tracker.clone());

        let scores_processor = ScoresProcessor::new(store.clone(), shard_tracker.clone());

        let commit_manager = ActorManager::new(default_buffer, commit_processor);
        let commit_votes_manager = ActorManager::new(default_buffer, commit_votes_processor);
        let reveal_manager = ActorManager::new(default_buffer, reveal_processor);
        let reveal_votes_manager = ActorManager::new(default_buffer, reveal_votes_processor);
        let scores_manager = ActorManager::new(default_buffer, scores_processor);

        let commit_handle = commit_manager.handle();
        let commit_votes_handle = commit_votes_manager.handle();
        let reveal_handle = reveal_manager.handle();
        let reveal_votes_handle = reveal_votes_manager.handle();
        let scores_handle = scores_manager.handle();

        let pipeline_dispatcher = InternalPipelineDispatcher::new(
            commit_handle,
            commit_votes_handle,
            reveal_handle,
            reveal_votes_handle,
            scores_handle,
        );
        let cache: Cache<Digest<ShardAuthToken>, VerificationStatus> = Cache::new(64);
        let verifier = ShardVerifier::new(cache, vdf_handle);

        let network_service = Arc::new(EncoderInternalService::new(
            context,
            store,
            pipeline_dispatcher,
            verifier,
        ));
        network_manager.start(network_service).await;
        Self {
            network_manager,
            #[cfg(msim)]
            sim_state: Default::default(),
        }
    }

    pub(crate) async fn stop(mut self) {
        <EncoderInternalTonicManager as EncoderInternalNetworkManager<
            EncoderInternalService<
                InternalPipelineDispatcher<
                    EncoderInternalTonicClient,
                    ObjectHttpClient,
                    FilesystemObjectStorage,
                >,
            >,
        >>::stop(&mut self.network_manager)
        .await;
    }
}

/// Wrap EncoderNode to allow correct access to EncoderNode in simulator tests.
pub struct EncoderNodeHandle {
    node: Option<Arc<EncoderNode>>,
    shutdown_on_drop: bool,
}

impl EncoderNodeHandle {
    pub fn new(node: Arc<EncoderNode>) -> Self {
        Self {
            node: Some(node),
            shutdown_on_drop: false,
        }
    }

    pub fn inner(&self) -> &Arc<EncoderNode> {
        self.node.as_ref().unwrap()
    }

    pub fn with<T>(&self, cb: impl FnOnce(&EncoderNode) -> T) -> T {
        let _guard = self.guard();
        cb(self.inner())
    }

    // TODO: have some way for simulator tests to inspect the state of the encoder
    // pub fn state(&self) -> Arc<AuthorityState> {
    //     self.with(|soma_node| soma_node.state())
    // }

    pub fn shutdown_on_drop(&mut self) {
        self.shutdown_on_drop = true;
    }
}

impl Clone for EncoderNodeHandle {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            shutdown_on_drop: false,
        }
    }
}

#[cfg(not(msim))]
impl EncoderNodeHandle {
    // Must return something to silence lints above at `let _guard = ...`
    fn guard(&self) -> u32 {
        0
    }

    pub async fn with_async<'a, F, R, T>(&'a self, cb: F) -> T
    where
        F: FnOnce(&'a EncoderNode) -> R,
        R: Future<Output = T>,
    {
        cb(self.inner()).await
    }
}

#[cfg(msim)]
impl EncoderNodeHandle {
    fn guard(&self) -> msim::runtime::NodeEnterGuard {
        self.inner().sim_state.sim_node.enter_node()
    }

    pub async fn with_async<'a, F, R, T>(&'a self, cb: F) -> T
    where
        F: FnOnce(&'a EncoderNode) -> R,
        R: Future<Output = T>,
    {
        let fut = cb(self.node.as_ref().unwrap());
        self.inner()
            .sim_state
            .sim_node
            .await_future_in_node(fut)
            .await
    }
}

#[cfg(msim)]
impl Drop for EncoderNodeHandle {
    fn drop(&mut self) {
        if self.shutdown_on_drop {
            let node_id = self.inner().sim_state.sim_node.id();
            msim::runtime::Handle::try_current().map(|h| h.delete_node(node_id));
        }
    }
}

impl From<Arc<EncoderNode>> for EncoderNodeHandle {
    fn from(node: Arc<EncoderNode>) -> Self {
        EncoderNodeHandle::new(node)
    }
}

#[cfg(msim)]
mod simulator {
    use super::*;
    use std::sync::atomic::AtomicBool;
    pub(super) struct SimState {
        pub sim_node: msim::runtime::NodeHandle,
    }

    impl Default for SimState {
        fn default() -> Self {
            Self {
                sim_node: msim::runtime::NodeHandle::current(),
            }
        }
    }
}
