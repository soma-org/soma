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
                .name("get_commit_summary")
                .route_name("GetCommitSummary")
                .input_type("types::state_sync::GetCommitSummaryRequest")
                .output_type("Option<types::state_sync::CertifiedCommitSummary>")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("get_commit_contents")
                .route_name("GetCommitContents")
                .input_type("types::digests::CommitContentsDigest")
                .output_type("Option<types::state_sync::FullCommitContents>")
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
                .name("push_commit_summary")
                .route_name("PushCommitSummary")
                .input_type("types::state_sync::CertifiedCommitSummary")
                .output_type("types::state_sync::PushCommitSummaryResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();

    tonic_build::manual::Builder::new()
        .out_dir("src/proto")
        .compile(&[discovery_service]);

    Ok(())
}
