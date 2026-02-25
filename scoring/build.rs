// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use tonic_build::manual::{Method, Service};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    let codec_path = "utils::codec::BcsCodec";

    let scoring_service = Service::builder()
        .name("Scoring")
        .package("scoring")
        .comment("The Scoring service interface")
        .method(
            Method::builder()
                .name("score")
                .route_name("Score")
                .input_type("crate::types::ScoreRequest")
                .output_type("crate::types::ScoreResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("health")
                .route_name("Health")
                .input_type("crate::types::HealthRequest")
                .output_type("crate::types::HealthResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();

    tonic_build::manual::Builder::new().out_dir("src/proto").compile(&[scoring_service]);

    Ok(())
}
