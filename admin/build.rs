// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use tonic_build::manual::{Method, Service};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    let codec_path = "utils::codec::BcsCodec";

    let admin_service = Service::builder()
        .name("Admin")
        .package("admin")
        .comment("Admin service for localnet epoch management")
        .method(
            Method::builder()
                .name("advance_epoch")
                .route_name("AdvanceEpoch")
                .input_type("crate::admin_types::AdvanceEpochRequest")
                .output_type("crate::admin_types::AdvanceEpochResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();

    tonic_build::manual::Builder::new().out_dir("src/proto").compile(&[admin_service]);

    Ok(())
}
