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
#[allow(clippy::too_many_lines)]
fn build_tonic_services(out_dir: &Path) {
    let codec_path = "tonic::codec::ProstCodec";
    let probe_tonic_service = tonic_build::manual::Service::builder()
        .name("ProbeTonicService")
        .package("soma")
        .comment("soma probe tonic service")
        .method(
            tonic_build::manual::Method::builder()
                .name("probe")
                .route_name("Probe")
                .input_type("crate::messaging::tonic::SendProbeRequest")
                .output_type("crate::messaging::tonic::SendProbeResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();
    tonic_build::manual::Builder::new()
        .out_dir(out_dir)
        .compile(&[probe_tonic_service]);
}
