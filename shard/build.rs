//! build file that builds the tonic related services and loads prost message definitions to
//! allow the types to be written in rust rather that proto definitions.

use std::{
    env,
    path::{Path, PathBuf},
};

/// aliasing long result
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    build_tonic_services(&out_dir);
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}

/// builds the tonic services
fn build_tonic_services(out_dir: &Path) {
    let codec_path = "tonic::codec::ProstCodec";

    let encoder_shard_service = tonic_build::manual::Service::builder()
        .name("EncoderService")
        .package("soma")
        .comment("Soma encoder interface")
        .method(
            tonic_build::manual::Method::builder()
                .name("send_input")
                .route_name("SendInput")
                .input_type("crate::networking::messaging::encoder_tonic_service::SendInputRequest")
                .output_type(
                    "crate::networking::messaging::encoder_tonic_service::SendInputResponse",
                )
                .codec_path(codec_path)
                .build(),
        )
        .method(
            tonic_build::manual::Method::builder()
                .name("send_selection")
                .route_name("SendSelection")
                .input_type(
                    "crate::networking::messaging::encoder_tonic_service::SendSelectionRequest",
                )
                .output_type(
                    "crate::networking::messaging::encoder_tonic_service::SendSelectionResponse",
                )
                .codec_path(codec_path)
                .build(),
        )
        .build();
    tonic_build::manual::Builder::new()
        .out_dir(out_dir)
        .compile(&[encoder_shard_service]);
}
