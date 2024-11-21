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
        .build();

    // let state_sync_service = Service::builder()
    //     .name("StateSync")
    //     .package("p2p")
    //     .method(
    //         Method::builder()
    //             .name("get_commit_availability")
    //             .route_name("GetCommitAvailability")
    //             .input_type("()")
    //             .output_type("types::grpc::GetCommitAvailabilityResponse")
    //             .codec_path(codec_path)
    //             .build(),
    //     )
    //     .method(
    //         Method::builder()
    //             .name("get_commit_contents")
    //             .route_name("GetCommitContents")
    //             .input_type("types::grpc::GetCommitContentsRequest")
    //             .output_type("types::grpc::GetCommitContentsResponse")
    //             .codec_path(codec_path)
    //             .build(),
    //     )
    //     .build();

    tonic_build::manual::Builder::new()
        .out_dir("src/proto")
        .compile(&[discovery_service]);

    Ok(())
}
