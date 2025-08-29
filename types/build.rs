use std::{
    env,
    path::{Path, PathBuf},
};

/// aliasing long result
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    // pyo3_build_config::add_python_framework_link_args();
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    build_tonic_services(&out_dir);
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn build_tonic_services(out_dir: &Path) {
    let codec_path = "tonic::codec::ProstCodec";
    let encoder_external_tonic_service = tonic_build::manual::Service::builder()
        .name("EncoderExternalTonicService")
        .package("soma")
        .comment("Soma encoder external interface")
        .method(
            tonic_build::manual::Method::builder()
                .name("send_input")
                .route_name("SendInput")
                .input_type("crate::shard_networking::external::SendInputRequest")
                .output_type("crate::shard_networking::external::SendInputResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();

    tonic_build::manual::Builder::new()
        .out_dir(out_dir)
        .compile(&[encoder_external_tonic_service]);
}
