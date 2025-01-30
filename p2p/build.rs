use tonic_build::manual::{Method, Service};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    let codec_path = "utils::codec::BcsCodec";

    let discovery_service = Service::builder()
        .name("P2p")
        .package("p2p")
        .method(
            Method::builder()
                .name("get_known_peers")
                .route_name("GetKnownPeers")
                .input_type("types::discovery::GetKnownPeersRequest")
                .output_type("types::discovery::GetKnownPeersResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("get_commit_availability")
                .route_name("GetCommitAvailability")
                .input_type("types::state_sync::GetCommitAvailabilityRequest")
                .output_type("types::state_sync::GetCommitAvailabilityResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("push_commit")
                .route_name("PushCommit")
                .input_type("types::state_sync::PushCommitRequest")
                .output_type("types::state_sync::PushCommitResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("get_commit_info")
                .route_name("GetCommitInfo")
                .input_type("types::state_sync::GetCommitInfoRequest")
                .output_type("types::state_sync::GetCommitInfoResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            tonic_build::manual::Method::builder()
                .name("fetch_blocks")
                .route_name("FetchBlocks")
                .input_type("types::state_sync::FetchBlocksRequest")
                .output_type("types::state_sync::FetchBlocksResponse")
                .codec_path(codec_path)
                .server_streaming()
                .build(),
        )
        .method(
            tonic_build::manual::Method::builder()
                .name("fetch_commits")
                .route_name("FetchCommits")
                .input_type("types::state_sync::FetchCommitsRequest")
                .output_type("types::state_sync::FetchCommitsResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();

    tonic_build::manual::Builder::new()
        .out_dir("src/proto")
        .compile(&[discovery_service]);

    Ok(())
}
