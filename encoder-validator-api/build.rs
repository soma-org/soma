use tonic_build::manual::{Method, Service};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    let codec_path = "utils::codec::BcsCodec";

    let encoder_validator_service = Service::builder()
        .name("EncoderValidatorApi")
        .package("encoder-validator-api")
        .comment("Interface for encoders to talk to validators")
        .method(
            Method::builder()
                .name("fetch_committees")
                .route_name("FetchCommittees")
                .input_type("types::encoder_validator::FetchCommitteesRequest")
                .output_type("types::encoder_validator::FetchCommitteesResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("get_latest_epoch")
                .route_name("GetLatestEpoch")
                .input_type("types::encoder_validator::GetLatestEpochRequest")
                .output_type("types::encoder_validator::GetLatestEpochResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();

    tonic_build::manual::Builder::new()
        .out_dir("src/proto")
        .compile(&[encoder_validator_service]);

    Ok(())
}
