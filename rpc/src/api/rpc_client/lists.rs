// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use futures::stream;
use futures::stream::Stream;
use prost_types::FieldMask;

use super::{Client, Result};
use crate::proto::soma::{ListOwnedObjectsRequest, Object};
use crate::utils::field::FieldMaskUtil as _;

impl Client {
    /// Creates a stream of objects based on the provided request.
    ///
    /// The stream handles pagination automatically by using the page_token from responses
    /// to fetch subsequent pages. The original request's page_token is used as the starting point.
    ///
    /// # Arguments
    /// * `request` - The initial `ListOwnedObjectsRequest` with search criteria
    ///
    /// # Returns
    /// A stream that yields `Result<Object>` instances. If any RPC call fails, the
    /// tonic::Status from that request is returned.
    pub fn list_owned_objects(
        &self,
        request: impl tonic::IntoRequest<ListOwnedObjectsRequest>,
    ) -> impl Stream<Item = Result<Object>> + 'static {
        let client = self.clone();
        let mut request = request.into_request();

        // Ensure read_mask includes all required fields for conversion
        if request.get_ref().read_mask.is_none() {
            request.get_mut().read_mask = Some(FieldMask::from_paths([
                "object_id",
                "version",
                "digest",
                "object_type",
                "owner",
                "contents",
                "previous_transaction",
            ]));
        }

        stream::unfold(
            (
                Vec::new().into_iter(), // current batch of objects
                true,                   // has_next_page
                request,                // request (page_token will be updated as we paginate)
                client,                 // client for making requests
            ),
            move |(mut iter, has_next_page, mut request, mut client)| async move {
                if let Some(item) = iter.next() {
                    return Some((Ok(item), (iter, has_next_page, request, client)));
                }

                if has_next_page {
                    let new_request = tonic::Request::from_parts(
                        request.metadata().clone(),
                        request.extensions().clone(),
                        request.get_ref().clone(),
                    );

                    match client.state_client().list_owned_objects(new_request).await {
                        Ok(response) => {
                            let response = response.into_inner();
                            let mut iter = response.objects.into_iter();

                            let has_next_page = response.next_page_token.is_some();
                            request.get_mut().page_token = response.next_page_token;

                            iter.next()
                                .map(|item| (Ok(item), (iter, has_next_page, request, client)))
                        }
                        Err(e) => {
                            // Return error and terminate stream
                            request.get_mut().page_token = None;
                            Some((Err(e), (Vec::new().into_iter(), false, request, client)))
                        }
                    }
                } else {
                    None
                }
            },
        )
    }
}
