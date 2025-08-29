use std::{
    env, fs,
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
    let evaluation_tonic_service = tonic_build::manual::Service::builder()
        .name("EvaluationTonicService")
        .package("soma")
        .comment("soma evaluation tonic service")
        .method(
            tonic_build::manual::Method::builder()
                .name("evaluation")
                .route_name("Evaluation")
                .input_type("crate::messaging::tonic::EvaluationRequest")
                .output_type("crate::messaging::tonic::EvaluationResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();
    tonic_build::manual::Builder::new()
        .out_dir(out_dir)
        .compile(&[evaluation_tonic_service]);
}
