type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    let codec_path = "tonic::codec::ProstCodec";

    let service = tonic_build::manual::Service::builder()
        .name("ConsensusService")
        .package("consensus")
        .comment("Consensus authority interface")
        .method(
            tonic_build::manual::Method::builder()
                .name("send_block")
                .route_name("SendBlock")
                .input_type("crate::network::tonic_network::SendBlockRequest")
                .output_type("crate::network::tonic_network::SendBlockResponse")
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
        .method(
            tonic_build::manual::Method::builder()
                .name("fetch_latest_blocks")
                .route_name("FetchLatestBlocks")
                .input_type("crate::network::tonic_network::FetchLatestBlocksRequest")
                .output_type("crate::network::tonic_network::FetchLatestBlocksResponse")
                .codec_path(codec_path)
                .server_streaming()
                .build(),
        )
        .build();

    tonic_build::manual::Builder::new()
        .out_dir("src/network/proto")
        .compile(&[service]);

    Ok(())
}
