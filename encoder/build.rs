//! build file that builds the tonic related services and loads prost message definitions to
//! allow the types to be written in rust rather that proto definitions.

use std::{
    env,
    path::{Path, PathBuf},
};

/// aliasing long result
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    pyo3_build_config::add_extension_module_link_args();
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    build_tonic_services(&out_dir);
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}

/// builds the tonic services
#[allow(clippy::too_many_lines)]
fn build_tonic_services(out_dir: &Path) {
    let codec_path = "tonic::codec::ProstCodec";
    let encoder_internal_tonic_service = tonic_build::manual::Service::builder()
        .name("EncoderInternalTonicService")
        .package("soma")
        .comment("Soma encoder internal interface")
        .method(
            tonic_build::manual::Method::builder()
                .name("send_commit")
                .route_name("SendCommit")
                .input_type("crate::networking::messaging::tonic_network::SendCommitRequest")
                .output_type("crate::networking::messaging::tonic_network::SendCommitResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            tonic_build::manual::Method::builder()
                .name("send_certified_commit")
                .route_name("SendCertifiedCommit")
                .input_type(
                    "crate::networking::messaging::tonic_network::SendCertifiedCommitRequest",
                )
                .output_type(
                    "crate::networking::messaging::tonic_network::SendCertifiedCommitResponse",
                )
                .codec_path(codec_path)
                .build(),
        )
        .method(
            tonic_build::manual::Method::builder()
                .name("send_commit_votes")
                .route_name("SendCommitVotes")
                .input_type("crate::networking::messaging::tonic_network::SendCommitVotesRequest")
                .output_type("crate::networking::messaging::tonic_network::SendCommitVotesResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            tonic_build::manual::Method::builder()
                .name("send_reveal")
                .route_name("SendReveal")
                .input_type("crate::networking::messaging::tonic_network::SendRevealRequest")
                .output_type("crate::networking::messaging::tonic_network::SendRevealResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            tonic_build::manual::Method::builder()
                .name("send_reveal_votes")
                .route_name("SendRevealVotes")
                .input_type("crate::networking::messaging::tonic_network::SendRevealVotesRequest")
                .output_type("crate::networking::messaging::tonic_network::SendRevealVotesResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            tonic_build::manual::Method::builder()
                .name("send_scores")
                .route_name("SendScores")
                .input_type("crate::networking::messaging::tonic_network::SendScoresRequest")
                .output_type("crate::networking::messaging::tonic_network::SendScoresResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();
    tonic_build::manual::Builder::new()
        .out_dir(out_dir)
        .compile(&[encoder_internal_tonic_service]);
}
