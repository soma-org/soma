use tonic_build::manual::{Method, Service};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    let codec_path = "utils::codec::BcsCodec";
    let prost_codec_path = "tonic_prost::ProstCodec";

    let validator_service = Service::builder()
        .name("Validator")
        .package("validator")
        .comment("The Validator interface")
        .method(
            Method::builder()
                .name("submit_transaction")
                .route_name("SubmitTransaction")
                .input_type("types::messages_grpc::RawSubmitTxRequest")
                .output_type("types::messages_grpc::RawSubmitTxResponse")
                .codec_path(prost_codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("wait_for_effects")
                .route_name("WaitForEffects")
                .input_type("types::messages_grpc::RawWaitForEffectsRequest")
                .output_type("types::messages_grpc::RawWaitForEffectsResponse")
                .codec_path(prost_codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("transaction")
                .route_name("Transaction")
                .input_type("types::transaction::Transaction")
                .output_type("types::messages_grpc::HandleTransactionResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("handle_certificate")
                .route_name("CertifiedTransaction")
                .input_type("types::messages_grpc::HandleCertificateRequest")
                .output_type("types::messages_grpc::HandleCertificateResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("object_info")
                .route_name("ObjectInfo")
                .input_type("types::messages_grpc::ObjectInfoRequest")
                .output_type("types::messages_grpc::ObjectInfoResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("transaction_info")
                .route_name("TransactionInfo")
                .input_type("types::messages_grpc::TransactionInfoRequest")
                .output_type("types::messages_grpc::TransactionInfoResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("checkpoint")
                .route_name("Checkpoint")
                .input_type("types::checkpoints::CheckpointRequest")
                .output_type("types::checkpoints::CheckpointResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("get_system_state_object")
                .route_name("GetSystemStateObject")
                .input_type("types::messages_grpc::SystemStateRequest")
                .output_type("types::system_state::SystemState")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("validator_health")
                .route_name("ValidatorHealth")
                .input_type("types::messages_grpc::RawValidatorHealthRequest")
                .output_type("types::messages_grpc::RawValidatorHealthResponse")
                .codec_path(prost_codec_path)
                .build(),
        )
        .build();

    tonic_build::manual::Builder::new().out_dir("src/proto").compile(&[validator_service]);

    Ok(())
}
