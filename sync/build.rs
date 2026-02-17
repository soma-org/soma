// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-network/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for the Soma project.

use tonic_build::manual::{Method, Service};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    let codec_path = "utils::codec::BcsCodec";

    let discovery_service = Service::builder()
        .name("P2p")
        .package("p2p")
        .method(
            Method::builder()
                .name("get_known_peers")
                .route_name("GetKnownPeers")
                .input_type("types::sync::GetKnownPeersRequest")
                .output_type("types::sync::GetKnownPeersResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("get_checkpoint_availability")
                .route_name("GetCheckpointAvailability")
                .input_type("types::sync::GetCheckpointAvailabilityRequest")
                .output_type("types::sync::GetCheckpointAvailabilityResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("push_checkpoint_summary")
                .route_name("PushCheckpointSummary")
                .input_type("types::sync::PushCheckpointSummaryRequest")
                .output_type("types::sync::PushCheckpointSummaryResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("get_checkpoint_summary")
                .route_name("GetCheckpointSummary")
                .input_type("types::sync::GetCheckpointSummaryRequest")
                .output_type("types::sync::GetCheckpointSummaryResponse")
                .codec_path(codec_path)
                .build(),
        )
        .method(
            Method::builder()
                .name("get_checkpoint_contents")
                .route_name("GetCheckpointContents")
                .input_type("types::sync::GetCheckpointContentsRequest")
                .output_type("types::sync::GetCheckpointContentsResponse")
                .codec_path(codec_path)
                .build(),
        )
        .build();

    tonic_build::manual::Builder::new().out_dir("src/proto").compile(&[discovery_service]);

    Ok(())
}
