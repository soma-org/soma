// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;

use futures::Stream;

use crate::api::RpcService;
use crate::proto::soma::state_service_server::StateService;
use crate::proto::soma::{
    AskEvent, BidEvent, GetAskRequest, GetAskResponse, GetBalanceRequest, GetBalanceResponse,
    GetBidRequest, GetBidResponse, GetBidsForAskRequest, GetBidsForAskResponse,
    GetOpenAsksRequest, GetOpenAsksResponse, GetProtocolFundRequest, GetProtocolFundResponse,
    GetReputationRequest, GetReputationResponse, GetSettlementRequest, GetSettlementResponse,
    GetSettlementsRequest, GetSettlementsResponse, GetVaultRequest, GetVaultResponse,
    ListOwnedObjectsRequest, ListOwnedObjectsResponse, SubscribeAsksRequest, SubscribeBidsRequest,
};
use crate::utils::field::{FieldMaskTree, FieldMaskUtil as _};
use crate::utils::merge::Merge as _;

mod get_balance;
mod list_owned_objects;

/// Helper: look up an object by ID and return it as a proto Object with standard field masking.
fn lookup_object_by_id(
    service: &RpcService,
    id_str: &str,
    not_found_msg: &str,
) -> Result<crate::proto::soma::Object, tonic::Status> {
    let object_id: types::object::ObjectID = id_str
        .parse()
        .map_err(|_| tonic::Status::invalid_argument("invalid object ID format"))?;

    let object = service
        .reader
        .inner()
        .get_object(&object_id)
        .ok_or_else(|| tonic::Status::not_found(not_found_msg))?;

    let read_mask = prost_types::FieldMask::from_paths([
        "object_id", "version", "digest", "object_type", "owner", "contents",
        "previous_transaction",
    ]);
    let tree = FieldMaskTree::from(read_mask);
    let mut proto_object = crate::proto::soma::Object::default();
    proto_object.merge(&object, &tree);

    Ok(proto_object)
}

#[tonic::async_trait]
impl StateService for RpcService {
    async fn list_owned_objects(
        &self,
        request: tonic::Request<ListOwnedObjectsRequest>,
    ) -> Result<tonic::Response<ListOwnedObjectsResponse>, tonic::Status> {
        list_owned_objects::list_owned_objects(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_balance(
        &self,
        request: tonic::Request<GetBalanceRequest>,
    ) -> Result<tonic::Response<GetBalanceResponse>, tonic::Status> {
        get_balance::get_balance(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_bid(
        &self,
        request: tonic::Request<GetBidRequest>,
    ) -> Result<tonic::Response<GetBidResponse>, tonic::Status> {
        let bid_id = request
            .into_inner()
            .bid_id
            .ok_or_else(|| tonic::Status::invalid_argument("bid_id is required"))?;

        let proto_object = lookup_object_by_id(self, &bid_id, "bid not found")?;

        Ok(tonic::Response::new(GetBidResponse {
            bid: Some(proto_object),
        }))
    }

    async fn get_ask(
        &self,
        request: tonic::Request<GetAskRequest>,
    ) -> Result<tonic::Response<GetAskResponse>, tonic::Status> {
        let ask_id = request
            .into_inner()
            .ask_id
            .ok_or_else(|| tonic::Status::invalid_argument("ask_id is required"))?;

        let proto_object = lookup_object_by_id(self, &ask_id, "ask not found")?;

        Ok(tonic::Response::new(GetAskResponse {
            ask: Some(proto_object),
        }))
    }

    async fn get_settlement(
        &self,
        request: tonic::Request<GetSettlementRequest>,
    ) -> Result<tonic::Response<GetSettlementResponse>, tonic::Status> {
        let settlement_id = request
            .into_inner()
            .settlement_id
            .ok_or_else(|| tonic::Status::invalid_argument("settlement_id is required"))?;

        let proto_object = lookup_object_by_id(self, &settlement_id, "settlement not found")?;

        Ok(tonic::Response::new(GetSettlementResponse {
            settlement: Some(proto_object),
        }))
    }

    async fn get_bids_for_ask(
        &self,
        request: tonic::Request<GetBidsForAskRequest>,
    ) -> Result<tonic::Response<GetBidsForAskResponse>, tonic::Status> {
        let req = request.into_inner();
        let ask_id_str = req
            .ask_id
            .ok_or_else(|| tonic::Status::invalid_argument("ask_id is required"))?;

        let ask_id: types::object::ObjectID = ask_id_str
            .parse()
            .map_err(|_| tonic::Status::invalid_argument("invalid ask_id format"))?;

        let indexes = self
            .reader
            .inner()
            .indexes()
            .ok_or_else(|| tonic::Status::unavailable("indexes not available"))?;

        let bid_ids = indexes
            .bids_for_ask(&ask_id)
            .map_err(|e| tonic::Status::internal(format!("index error: {e}")))?;

        let read_mask = prost_types::FieldMask::from_paths([
            "object_id", "version", "digest", "object_type", "owner", "contents",
            "previous_transaction",
        ]);
        let tree = FieldMaskTree::from(read_mask);

        let mut bids = Vec::new();
        for bid_id in bid_ids {
            if let Some(object) = self.reader.inner().get_object(&bid_id) {
                // Optional status filter: deserialize bid and check status
                if let Some(ref status_filter) = req.status_filter {
                    if let Some(bid) =
                        object.deserialize_contents::<types::bid::Bid>(types::object::ObjectType::Bid)
                    {
                        let status_str = format!("{:?}", bid.status);
                        if !status_str.eq_ignore_ascii_case(status_filter) {
                            continue;
                        }
                    }
                }
                let mut proto_object = crate::proto::soma::Object::default();
                proto_object.merge(&object, &tree);
                bids.push(proto_object);
            }
        }

        Ok(tonic::Response::new(GetBidsForAskResponse { bids }))
    }

    async fn get_open_asks(
        &self,
        request: tonic::Request<GetOpenAsksRequest>,
    ) -> Result<tonic::Response<GetOpenAsksResponse>, tonic::Status> {
        let req = request.into_inner();
        let page_size = req.page_size.unwrap_or(100).min(1000) as usize;

        let indexes = self
            .reader
            .inner()
            .indexes()
            .ok_or_else(|| tonic::Status::unavailable("indexes not available"))?;

        let ask_ids = indexes
            .open_asks()
            .map_err(|e| tonic::Status::internal(format!("index error: {e}")))?;

        let read_mask = prost_types::FieldMask::from_paths([
            "object_id", "version", "digest", "object_type", "owner", "contents",
            "previous_transaction",
        ]);
        let tree = FieldMaskTree::from(read_mask);

        // Optional buyer filter
        let buyer_filter: Option<types::base::SomaAddress> = req
            .buyer
            .as_ref()
            .map(|b| {
                b.parse()
                    .map_err(|_| tonic::Status::invalid_argument("invalid buyer address"))
            })
            .transpose()?;

        let mut asks = Vec::new();
        for ask_id in ask_ids {
            if asks.len() >= page_size {
                break;
            }
            if let Some(object) = self.reader.inner().get_object(&ask_id) {
                // Apply buyer filter if specified
                if let Some(ref buyer) = buyer_filter {
                    if let Some(ask) = object
                        .deserialize_contents::<types::ask::Ask>(types::object::ObjectType::Ask)
                    {
                        if ask.buyer != *buyer {
                            continue;
                        }
                    }
                }
                let mut proto_object = crate::proto::soma::Object::default();
                proto_object.merge(&object, &tree);
                asks.push(proto_object);
            }
        }

        Ok(tonic::Response::new(GetOpenAsksResponse { asks }))
    }

    async fn get_vault(
        &self,
        request: tonic::Request<GetVaultRequest>,
    ) -> Result<tonic::Response<GetVaultResponse>, tonic::Status> {
        let owner_str = request
            .into_inner()
            .owner
            .ok_or_else(|| tonic::Status::invalid_argument("owner is required"))?;

        let owner: types::base::SomaAddress = owner_str
            .parse()
            .map_err(|_| tonic::Status::invalid_argument("invalid owner address"))?;

        let indexes = self
            .reader
            .inner()
            .indexes()
            .ok_or_else(|| tonic::Status::unavailable("indexes not available"))?;

        // SellerVault is an owned object — use owned_objects_iter with type filter
        let iter = indexes
            .owned_objects_iter(owner, Some(types::object::ObjectType::SellerVault), None)
            .map_err(|e| tonic::Status::internal(format!("index error: {e}")))?;

        let read_mask = prost_types::FieldMask::from_paths([
            "object_id", "version", "digest", "object_type", "owner", "contents",
            "previous_transaction",
        ]);
        let tree = FieldMaskTree::from(read_mask);

        let mut vaults = Vec::new();
        for info in iter {
            let info =
                info.map_err(|e| tonic::Status::internal(format!("index iteration error: {e}")))?;
            if let Some(object) = self.reader.inner().get_object(&info.object_id) {
                let mut proto_object = crate::proto::soma::Object::default();
                proto_object.merge(&object, &tree);
                vaults.push(proto_object);
            }
        }

        Ok(tonic::Response::new(GetVaultResponse { vaults }))
    }

    async fn get_settlements(
        &self,
        request: tonic::Request<GetSettlementsRequest>,
    ) -> Result<tonic::Response<GetSettlementsResponse>, tonic::Status> {
        let req = request.into_inner();
        let page_size = req.page_size.unwrap_or(100).min(1000) as usize;

        let indexes = self
            .reader
            .inner()
            .indexes()
            .ok_or_else(|| tonic::Status::unavailable("indexes not available"))?;

        // Parse buyer/seller filters
        let buyer_filter: Option<types::base::SomaAddress> = req
            .buyer
            .as_ref()
            .map(|b| {
                b.parse()
                    .map_err(|_| tonic::Status::invalid_argument("invalid buyer address"))
            })
            .transpose()?;

        let seller_filter: Option<types::base::SomaAddress> = req
            .seller
            .as_ref()
            .map(|s| {
                s.parse()
                    .map_err(|_| tonic::Status::invalid_argument("invalid seller address"))
            })
            .transpose()?;

        // Determine which index to use based on filters
        let settlement_ids: Vec<types::object::ObjectID> = if let Some(ref buyer) = buyer_filter {
            indexes
                .settlements_by_buyer(buyer)
                .map_err(|e| tonic::Status::internal(format!("index error: {e}")))?
        } else if let Some(ref seller) = seller_filter {
            indexes
                .settlements_by_seller(seller)
                .map_err(|e| tonic::Status::internal(format!("index error: {e}")))?
        } else {
            return Err(tonic::Status::invalid_argument(
                "at least one of buyer or seller is required",
            ));
        };

        let read_mask = prost_types::FieldMask::from_paths([
            "object_id", "version", "digest", "object_type", "owner", "contents",
            "previous_transaction",
        ]);
        let tree = FieldMaskTree::from(read_mask);

        let mut settlements = Vec::new();
        for settlement_id in settlement_ids {
            if settlements.len() >= page_size {
                break;
            }
            if let Some(object) = self.reader.inner().get_object(&settlement_id) {
                // If both buyer and seller filters are present, cross-check
                if buyer_filter.is_some() && seller_filter.is_some() {
                    if let Some(settlement) = object
                        .deserialize_contents::<types::settlement::Settlement>(
                            types::object::ObjectType::Settlement,
                        )
                    {
                        if Some(settlement.seller) != seller_filter {
                            continue;
                        }
                    }
                }
                let mut proto_object = crate::proto::soma::Object::default();
                proto_object.merge(&object, &tree);
                settlements.push(proto_object);
            }
        }

        Ok(tonic::Response::new(GetSettlementsResponse { settlements }))
    }

    async fn get_protocol_fund(
        &self,
        _request: tonic::Request<GetProtocolFundRequest>,
    ) -> Result<tonic::Response<GetProtocolFundResponse>, tonic::Status> {
        let system_state = self
            .reader
            .get_system_state()
            .map_err(|e| tonic::Status::internal(format!("failed to get system state: {e}")))?;

        Ok(tonic::Response::new(GetProtocolFundResponse {
            balance: Some(system_state.protocol_fund_balance()),
        }))
    }

    async fn get_reputation(
        &self,
        request: tonic::Request<GetReputationRequest>,
    ) -> Result<tonic::Response<GetReputationResponse>, tonic::Status> {
        let addr_str = request
            .into_inner()
            .address
            .ok_or_else(|| tonic::Status::invalid_argument("address is required"))?;

        let addr: types::base::SomaAddress = addr_str
            .parse()
            .map_err(|_| tonic::Status::invalid_argument("invalid address format"))?;

        let indexes = self
            .reader
            .inner()
            .indexes()
            .ok_or_else(|| tonic::Status::unavailable("indexes not available"))?;

        // Compute buyer reputation from settlements where addr is buyer
        let buyer_settlement_ids = indexes
            .settlements_by_buyer(&addr)
            .map_err(|e| tonic::Status::internal(format!("index error: {e}")))?;

        let mut buyer_volume: u64 = 0;
        let mut buyer_unique_sellers = std::collections::HashSet::new();
        let mut buyer_count: u64 = 0;
        for sid in &buyer_settlement_ids {
            if let Some(obj) = self.reader.inner().get_object(sid) {
                if let Some(s) = obj.deserialize_contents::<types::settlement::Settlement>(
                    types::object::ObjectType::Settlement,
                ) {
                    buyer_volume = buyer_volume.saturating_add(s.amount);
                    buyer_unique_sellers.insert(s.seller);
                    buyer_count += 1;
                }
            }
        }

        // Compute seller reputation from settlements where addr is seller
        let seller_settlement_ids = indexes
            .settlements_by_seller(&addr)
            .map_err(|e| tonic::Status::internal(format!("index error: {e}")))?;

        let mut seller_volume: u64 = 0;
        let mut seller_negative: u64 = 0;
        let mut seller_unique_buyers = std::collections::HashSet::new();
        let mut seller_count: u64 = 0;
        for sid in &seller_settlement_ids {
            if let Some(obj) = self.reader.inner().get_object(sid) {
                if let Some(s) = obj.deserialize_contents::<types::settlement::Settlement>(
                    types::object::ObjectType::Settlement,
                ) {
                    seller_volume = seller_volume.saturating_add(s.amount);
                    seller_unique_buyers.insert(s.buyer);
                    seller_count += 1;
                    if s.seller_rating == types::settlement::SellerRating::Negative {
                        seller_negative += 1;
                    }
                }
            }
        }

        let approval_rate = if seller_count > 0 {
            Some((seller_count - seller_negative) as f64 / seller_count as f64 * 100.0)
        } else {
            None
        };

        Ok(tonic::Response::new(GetReputationResponse {
            address: Some(addr.to_string()),
            buyer_settlements: Some(buyer_count),
            buyer_volume_spent: Some(buyer_volume),
            buyer_unique_sellers: Some(buyer_unique_sellers.len() as u64),
            seller_settlements: Some(seller_count),
            seller_volume_earned: Some(seller_volume),
            seller_negative_ratings: Some(seller_negative),
            seller_approval_rate: approval_rate,
            seller_unique_buyers: Some(seller_unique_buyers.len() as u64),
        }))
    }

    type SubscribeAsksStream =
        Pin<Box<dyn Stream<Item = Result<AskEvent, tonic::Status>> + Send>>;

    async fn subscribe_asks(
        &self,
        _request: tonic::Request<SubscribeAsksRequest>,
    ) -> Result<tonic::Response<Self::SubscribeAsksStream>, tonic::Status> {
        // TODO(Phase 6): Wire broadcast channel to checkpoint/tx processing for real-time events.
        // For now, return an empty stream. Clients should poll GetOpenAsks as a fallback.
        let stream = futures::stream::empty();
        Ok(tonic::Response::new(Box::pin(stream) as Self::SubscribeAsksStream))
    }

    type SubscribeBidsStream =
        Pin<Box<dyn Stream<Item = Result<BidEvent, tonic::Status>> + Send>>;

    async fn subscribe_bids(
        &self,
        _request: tonic::Request<SubscribeBidsRequest>,
    ) -> Result<tonic::Response<Self::SubscribeBidsStream>, tonic::Status> {
        // TODO(Phase 6): Wire broadcast channel to checkpoint/tx processing for real-time events.
        // For now, return an empty stream. Clients should poll GetBidsForAsk as a fallback.
        let stream = futures::stream::empty();
        Ok(tonic::Response::new(Box::pin(stream) as Self::SubscribeBidsStream))
    }
}
