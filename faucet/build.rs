use tonic_build::manual::{Method, Service};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    let codec_path = "crate::codec::BcsCodec";

    let faucet_service = Service::builder()
        .name("Faucet")
        .package("faucet")
        .comment("Faucet service for localnet token distribution")
        .method(
            Method::builder()
                .name("request_gas")
                .route_name("RequestGas")
                .input_type("crate::faucet_types::GasRequest")
                .output_type("crate::faucet_types::GasResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();

    tonic_build::manual::Builder::new().out_dir("src/proto").compile(&[faucet_service]);

    Ok(())
}
