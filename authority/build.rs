use tonic_build::manual::{Method, Service};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    let codec_path = "utils::codec::BcsCodec";

    let validator_service = Service::builder()
        .name("Validator")
        .package("validator")
        .comment("The Validator interface")
        .method(
            Method::builder()
                .name("transaction")
                .route_name("Transaction")
                .input_type("types::transaction::Transaction")
                .output_type("types::grpc::HandleTransactionResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("handle_certificate")
                .route_name("CertifiedTransaction")
                .input_type("types::grpc::HandleCertificateRequest")
                .output_type("types::grpc::HandleCertificateResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("submit_certificate")
                .route_name("SubmitCertificate")
                .input_type("types::transaction::CertifiedTransaction")
                .output_type("types::grpc::SubmitCertificateResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();

    tonic_build::manual::Builder::new()
        .out_dir("src/proto")
        .compile(&[validator_service]);

    Ok(())
}
