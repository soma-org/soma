impl serde::Serialize for AddStake {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.address.is_some() {
            len += 1;
        }
        if self.coin_ref.is_some() {
            len += 1;
        }
        if self.amount.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.AddStake", len)?;
        if let Some(v) = self.address.as_ref() {
            struct_ser.serialize_field("address", v)?;
        }
        if let Some(v) = self.coin_ref.as_ref() {
            struct_ser.serialize_field("coinRef", v)?;
        }
        if let Some(v) = self.amount.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("amount", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for AddStake {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "address",
            "coin_ref",
            "coinRef",
            "amount",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Address,
            CoinRef,
            Amount,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "address" => Ok(GeneratedField::Address),
                            "coinRef" | "coin_ref" => Ok(GeneratedField::CoinRef),
                            "amount" => Ok(GeneratedField::Amount),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = AddStake;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.AddStake")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<AddStake, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut address__ = None;
                let mut coin_ref__ = None;
                let mut amount__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Address => {
                            if address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("address"));
                            }
                            address__ = map_.next_value()?;
                        }
                        GeneratedField::CoinRef => {
                            if coin_ref__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coinRef"));
                            }
                            coin_ref__ = map_.next_value()?;
                        }
                        GeneratedField::Amount => {
                            if amount__.is_some() {
                                return Err(serde::de::Error::duplicate_field("amount"));
                            }
                            amount__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(AddStake {
                    address: address__,
                    coin_ref: coin_ref__,
                    amount: amount__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.AddStake", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for AddStakeToModel {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.model_id.is_some() {
            len += 1;
        }
        if self.coin_ref.is_some() {
            len += 1;
        }
        if self.amount.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.AddStakeToModel", len)?;
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        if let Some(v) = self.coin_ref.as_ref() {
            struct_ser.serialize_field("coinRef", v)?;
        }
        if let Some(v) = self.amount.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("amount", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for AddStakeToModel {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "model_id",
            "modelId",
            "coin_ref",
            "coinRef",
            "amount",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ModelId,
            CoinRef,
            Amount,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            "coinRef" | "coin_ref" => Ok(GeneratedField::CoinRef),
                            "amount" => Ok(GeneratedField::Amount),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = AddStakeToModel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.AddStakeToModel")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<AddStakeToModel, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut model_id__ = None;
                let mut coin_ref__ = None;
                let mut amount__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::CoinRef => {
                            if coin_ref__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coinRef"));
                            }
                            coin_ref__ = map_.next_value()?;
                        }
                        GeneratedField::Amount => {
                            if amount__.is_some() {
                                return Err(serde::de::Error::duplicate_field("amount"));
                            }
                            amount__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(AddStakeToModel {
                    model_id: model_id__,
                    coin_ref: coin_ref__,
                    amount: amount__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.AddStakeToModel", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for AddValidator {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.pubkey_bytes.is_some() {
            len += 1;
        }
        if self.network_pubkey_bytes.is_some() {
            len += 1;
        }
        if self.worker_pubkey_bytes.is_some() {
            len += 1;
        }
        if self.net_address.is_some() {
            len += 1;
        }
        if self.p2p_address.is_some() {
            len += 1;
        }
        if self.primary_address.is_some() {
            len += 1;
        }
        if self.proxy_address.is_some() {
            len += 1;
        }
        if self.proof_of_possession.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.AddValidator", len)?;
        if let Some(v) = self.pubkey_bytes.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("pubkeyBytes", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.network_pubkey_bytes.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("networkPubkeyBytes", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.worker_pubkey_bytes.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("workerPubkeyBytes", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.net_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("netAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.p2p_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("p2pAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.primary_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("primaryAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.proxy_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("proxyAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.proof_of_possession.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("proofOfPossession", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for AddValidator {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "pubkey_bytes",
            "pubkeyBytes",
            "network_pubkey_bytes",
            "networkPubkeyBytes",
            "worker_pubkey_bytes",
            "workerPubkeyBytes",
            "net_address",
            "netAddress",
            "p2p_address",
            "p2pAddress",
            "primary_address",
            "primaryAddress",
            "proxy_address",
            "proxyAddress",
            "proof_of_possession",
            "proofOfPossession",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            PubkeyBytes,
            NetworkPubkeyBytes,
            WorkerPubkeyBytes,
            NetAddress,
            P2pAddress,
            PrimaryAddress,
            ProxyAddress,
            ProofOfPossession,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "pubkeyBytes" | "pubkey_bytes" => Ok(GeneratedField::PubkeyBytes),
                            "networkPubkeyBytes" | "network_pubkey_bytes" => Ok(GeneratedField::NetworkPubkeyBytes),
                            "workerPubkeyBytes" | "worker_pubkey_bytes" => Ok(GeneratedField::WorkerPubkeyBytes),
                            "netAddress" | "net_address" => Ok(GeneratedField::NetAddress),
                            "p2pAddress" | "p2p_address" => Ok(GeneratedField::P2pAddress),
                            "primaryAddress" | "primary_address" => Ok(GeneratedField::PrimaryAddress),
                            "proxyAddress" | "proxy_address" => Ok(GeneratedField::ProxyAddress),
                            "proofOfPossession" | "proof_of_possession" => Ok(GeneratedField::ProofOfPossession),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = AddValidator;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.AddValidator")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<AddValidator, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut pubkey_bytes__ = None;
                let mut network_pubkey_bytes__ = None;
                let mut worker_pubkey_bytes__ = None;
                let mut net_address__ = None;
                let mut p2p_address__ = None;
                let mut primary_address__ = None;
                let mut proxy_address__ = None;
                let mut proof_of_possession__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::PubkeyBytes => {
                            if pubkey_bytes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pubkeyBytes"));
                            }
                            pubkey_bytes__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NetworkPubkeyBytes => {
                            if network_pubkey_bytes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("networkPubkeyBytes"));
                            }
                            network_pubkey_bytes__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::WorkerPubkeyBytes => {
                            if worker_pubkey_bytes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("workerPubkeyBytes"));
                            }
                            worker_pubkey_bytes__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NetAddress => {
                            if net_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("netAddress"));
                            }
                            net_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::P2pAddress => {
                            if p2p_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("p2pAddress"));
                            }
                            p2p_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PrimaryAddress => {
                            if primary_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("primaryAddress"));
                            }
                            primary_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ProxyAddress => {
                            if proxy_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("proxyAddress"));
                            }
                            proxy_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ProofOfPossession => {
                            if proof_of_possession__.is_some() {
                                return Err(serde::de::Error::duplicate_field("proofOfPossession"));
                            }
                            proof_of_possession__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(AddValidator {
                    pubkey_bytes: pubkey_bytes__,
                    network_pubkey_bytes: network_pubkey_bytes__,
                    worker_pubkey_bytes: worker_pubkey_bytes__,
                    net_address: net_address__,
                    p2p_address: p2p_address__,
                    primary_address: primary_address__,
                    proxy_address: proxy_address__,
                    proof_of_possession: proof_of_possession__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.AddValidator", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for BalanceChange {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.address.is_some() {
            len += 1;
        }
        if self.amount.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.BalanceChange", len)?;
        if let Some(v) = self.address.as_ref() {
            struct_ser.serialize_field("address", v)?;
        }
        if let Some(v) = self.amount.as_ref() {
            struct_ser.serialize_field("amount", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for BalanceChange {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "address",
            "amount",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Address,
            Amount,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "address" => Ok(GeneratedField::Address),
                            "amount" => Ok(GeneratedField::Amount),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = BalanceChange;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.BalanceChange")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<BalanceChange, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut address__ = None;
                let mut amount__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Address => {
                            if address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("address"));
                            }
                            address__ = map_.next_value()?;
                        }
                        GeneratedField::Amount => {
                            if amount__.is_some() {
                                return Err(serde::de::Error::duplicate_field("amount"));
                            }
                            amount__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(BalanceChange {
                    address: address__,
                    amount: amount__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.BalanceChange", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for BatchGetObjectsRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.requests.is_empty() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.BatchGetObjectsRequest", len)?;
        if !self.requests.is_empty() {
            struct_ser.serialize_field("requests", &self.requests)?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for BatchGetObjectsRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "requests",
            "read_mask",
            "readMask",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Requests,
            ReadMask,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "requests" => Ok(GeneratedField::Requests),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = BatchGetObjectsRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.BatchGetObjectsRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<BatchGetObjectsRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut requests__ = None;
                let mut read_mask__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Requests => {
                            if requests__.is_some() {
                                return Err(serde::de::Error::duplicate_field("requests"));
                            }
                            requests__ = Some(map_.next_value()?);
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(BatchGetObjectsRequest {
                    requests: requests__.unwrap_or_default(),
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.BatchGetObjectsRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for BatchGetObjectsResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.objects.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.BatchGetObjectsResponse", len)?;
        if !self.objects.is_empty() {
            struct_ser.serialize_field("objects", &self.objects)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for BatchGetObjectsResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "objects",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Objects,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "objects" => Ok(GeneratedField::Objects),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = BatchGetObjectsResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.BatchGetObjectsResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<BatchGetObjectsResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut objects__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Objects => {
                            if objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objects"));
                            }
                            objects__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(BatchGetObjectsResponse {
                    objects: objects__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.BatchGetObjectsResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for BatchGetTransactionsRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.digests.is_empty() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.BatchGetTransactionsRequest", len)?;
        if !self.digests.is_empty() {
            struct_ser.serialize_field("digests", &self.digests)?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for BatchGetTransactionsRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "digests",
            "read_mask",
            "readMask",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Digests,
            ReadMask,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "digests" => Ok(GeneratedField::Digests),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = BatchGetTransactionsRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.BatchGetTransactionsRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<BatchGetTransactionsRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut digests__ = None;
                let mut read_mask__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Digests => {
                            if digests__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digests"));
                            }
                            digests__ = Some(map_.next_value()?);
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(BatchGetTransactionsRequest {
                    digests: digests__.unwrap_or_default(),
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.BatchGetTransactionsRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for BatchGetTransactionsResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.transactions.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.BatchGetTransactionsResponse", len)?;
        if !self.transactions.is_empty() {
            struct_ser.serialize_field("transactions", &self.transactions)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for BatchGetTransactionsResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "transactions",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Transactions,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "transactions" => Ok(GeneratedField::Transactions),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = BatchGetTransactionsResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.BatchGetTransactionsResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<BatchGetTransactionsResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut transactions__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Transactions => {
                            if transactions__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transactions"));
                            }
                            transactions__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(BatchGetTransactionsResponse {
                    transactions: transactions__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.BatchGetTransactionsResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Challenge {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.id.is_some() {
            len += 1;
        }
        if self.target_id.is_some() {
            len += 1;
        }
        if self.challenger.is_some() {
            len += 1;
        }
        if self.challenger_bond.is_some() {
            len += 1;
        }
        if self.challenge_epoch.is_some() {
            len += 1;
        }
        if self.status.is_some() {
            len += 1;
        }
        if self.verdict.is_some() {
            len += 1;
        }
        if self.win_reason.is_some() {
            len += 1;
        }
        if self.distance_threshold.is_some() {
            len += 1;
        }
        if self.winning_distance_score.is_some() {
            len += 1;
        }
        if self.winning_model_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Challenge", len)?;
        if let Some(v) = self.id.as_ref() {
            struct_ser.serialize_field("id", v)?;
        }
        if let Some(v) = self.target_id.as_ref() {
            struct_ser.serialize_field("targetId", v)?;
        }
        if let Some(v) = self.challenger.as_ref() {
            struct_ser.serialize_field("challenger", v)?;
        }
        if let Some(v) = self.challenger_bond.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("challengerBond", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.challenge_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("challengeEpoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.status.as_ref() {
            struct_ser.serialize_field("status", v)?;
        }
        if let Some(v) = self.verdict.as_ref() {
            struct_ser.serialize_field("verdict", v)?;
        }
        if let Some(v) = self.win_reason.as_ref() {
            struct_ser.serialize_field("winReason", v)?;
        }
        if let Some(v) = self.distance_threshold.as_ref() {
            struct_ser.serialize_field("distanceThreshold", v)?;
        }
        if let Some(v) = self.winning_distance_score.as_ref() {
            struct_ser.serialize_field("winningDistanceScore", v)?;
        }
        if let Some(v) = self.winning_model_id.as_ref() {
            struct_ser.serialize_field("winningModelId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Challenge {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "id",
            "target_id",
            "targetId",
            "challenger",
            "challenger_bond",
            "challengerBond",
            "challenge_epoch",
            "challengeEpoch",
            "status",
            "verdict",
            "win_reason",
            "winReason",
            "distance_threshold",
            "distanceThreshold",
            "winning_distance_score",
            "winningDistanceScore",
            "winning_model_id",
            "winningModelId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Id,
            TargetId,
            Challenger,
            ChallengerBond,
            ChallengeEpoch,
            Status,
            Verdict,
            WinReason,
            DistanceThreshold,
            WinningDistanceScore,
            WinningModelId,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "id" => Ok(GeneratedField::Id),
                            "targetId" | "target_id" => Ok(GeneratedField::TargetId),
                            "challenger" => Ok(GeneratedField::Challenger),
                            "challengerBond" | "challenger_bond" => Ok(GeneratedField::ChallengerBond),
                            "challengeEpoch" | "challenge_epoch" => Ok(GeneratedField::ChallengeEpoch),
                            "status" => Ok(GeneratedField::Status),
                            "verdict" => Ok(GeneratedField::Verdict),
                            "winReason" | "win_reason" => Ok(GeneratedField::WinReason),
                            "distanceThreshold" | "distance_threshold" => Ok(GeneratedField::DistanceThreshold),
                            "winningDistanceScore" | "winning_distance_score" => Ok(GeneratedField::WinningDistanceScore),
                            "winningModelId" | "winning_model_id" => Ok(GeneratedField::WinningModelId),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Challenge;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Challenge")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Challenge, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut id__ = None;
                let mut target_id__ = None;
                let mut challenger__ = None;
                let mut challenger_bond__ = None;
                let mut challenge_epoch__ = None;
                let mut status__ = None;
                let mut verdict__ = None;
                let mut win_reason__ = None;
                let mut distance_threshold__ = None;
                let mut winning_distance_score__ = None;
                let mut winning_model_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Id => {
                            if id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("id"));
                            }
                            id__ = map_.next_value()?;
                        }
                        GeneratedField::TargetId => {
                            if target_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetId"));
                            }
                            target_id__ = map_.next_value()?;
                        }
                        GeneratedField::Challenger => {
                            if challenger__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challenger"));
                            }
                            challenger__ = map_.next_value()?;
                        }
                        GeneratedField::ChallengerBond => {
                            if challenger_bond__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challengerBond"));
                            }
                            challenger_bond__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ChallengeEpoch => {
                            if challenge_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challengeEpoch"));
                            }
                            challenge_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Status => {
                            if status__.is_some() {
                                return Err(serde::de::Error::duplicate_field("status"));
                            }
                            status__ = map_.next_value()?;
                        }
                        GeneratedField::Verdict => {
                            if verdict__.is_some() {
                                return Err(serde::de::Error::duplicate_field("verdict"));
                            }
                            verdict__ = map_.next_value()?;
                        }
                        GeneratedField::WinReason => {
                            if win_reason__.is_some() {
                                return Err(serde::de::Error::duplicate_field("winReason"));
                            }
                            win_reason__ = map_.next_value()?;
                        }
                        GeneratedField::DistanceThreshold => {
                            if distance_threshold__.is_some() {
                                return Err(serde::de::Error::duplicate_field("distanceThreshold"));
                            }
                            distance_threshold__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::WinningDistanceScore => {
                            if winning_distance_score__.is_some() {
                                return Err(serde::de::Error::duplicate_field("winningDistanceScore"));
                            }
                            winning_distance_score__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::WinningModelId => {
                            if winning_model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("winningModelId"));
                            }
                            winning_model_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Challenge {
                    id: id__,
                    target_id: target_id__,
                    challenger: challenger__,
                    challenger_bond: challenger_bond__,
                    challenge_epoch: challenge_epoch__,
                    status: status__,
                    verdict: verdict__,
                    win_reason: win_reason__,
                    distance_threshold: distance_threshold__,
                    winning_distance_score: winning_distance_score__,
                    winning_model_id: winning_model_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Challenge", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ChangeEpoch {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.epoch.is_some() {
            len += 1;
        }
        if self.epoch_start_timestamp.is_some() {
            len += 1;
        }
        if self.protocol_version.is_some() {
            len += 1;
        }
        if self.fees.is_some() {
            len += 1;
        }
        if self.epoch_randomness.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ChangeEpoch", len)?;
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.epoch_start_timestamp.as_ref() {
            struct_ser.serialize_field("epochStartTimestamp", &crate::utils::_serde::TimestampSerializer(v))?;
        }
        if let Some(v) = self.protocol_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("protocolVersion", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.fees.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("fees", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.epoch_randomness.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epochRandomness", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ChangeEpoch {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epoch",
            "epoch_start_timestamp",
            "epochStartTimestamp",
            "protocol_version",
            "protocolVersion",
            "fees",
            "epoch_randomness",
            "epochRandomness",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            EpochStartTimestamp,
            ProtocolVersion,
            Fees,
            EpochRandomness,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "epoch" => Ok(GeneratedField::Epoch),
                            "epochStartTimestamp" | "epoch_start_timestamp" => Ok(GeneratedField::EpochStartTimestamp),
                            "protocolVersion" | "protocol_version" => Ok(GeneratedField::ProtocolVersion),
                            "fees" => Ok(GeneratedField::Fees),
                            "epochRandomness" | "epoch_randomness" => Ok(GeneratedField::EpochRandomness),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ChangeEpoch;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ChangeEpoch")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ChangeEpoch, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epoch__ = None;
                let mut epoch_start_timestamp__ = None;
                let mut protocol_version__ = None;
                let mut fees__ = None;
                let mut epoch_randomness__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::EpochStartTimestamp => {
                            if epoch_start_timestamp__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epochStartTimestamp"));
                            }
                            epoch_start_timestamp__ = map_.next_value::<::std::option::Option<crate::utils::_serde::TimestampDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::ProtocolVersion => {
                            if protocol_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("protocolVersion"));
                            }
                            protocol_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Fees => {
                            if fees__.is_some() {
                                return Err(serde::de::Error::duplicate_field("fees"));
                            }
                            fees__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::EpochRandomness => {
                            if epoch_randomness__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epochRandomness"));
                            }
                            epoch_randomness__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ChangeEpoch {
                    epoch: epoch__,
                    epoch_start_timestamp: epoch_start_timestamp__,
                    protocol_version: protocol_version__,
                    fees: fees__,
                    epoch_randomness: epoch_randomness__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ChangeEpoch", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ChangedObject {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.object_id.is_some() {
            len += 1;
        }
        if self.input_state.is_some() {
            len += 1;
        }
        if self.input_version.is_some() {
            len += 1;
        }
        if self.input_digest.is_some() {
            len += 1;
        }
        if self.input_owner.is_some() {
            len += 1;
        }
        if self.output_state.is_some() {
            len += 1;
        }
        if self.output_version.is_some() {
            len += 1;
        }
        if self.output_digest.is_some() {
            len += 1;
        }
        if self.output_owner.is_some() {
            len += 1;
        }
        if self.id_operation.is_some() {
            len += 1;
        }
        if self.object_type.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ChangedObject", len)?;
        if let Some(v) = self.object_id.as_ref() {
            struct_ser.serialize_field("objectId", v)?;
        }
        if let Some(v) = self.input_state.as_ref() {
            let v = changed_object::InputObjectState::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("inputState", &v)?;
        }
        if let Some(v) = self.input_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("inputVersion", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.input_digest.as_ref() {
            struct_ser.serialize_field("inputDigest", v)?;
        }
        if let Some(v) = self.input_owner.as_ref() {
            struct_ser.serialize_field("inputOwner", v)?;
        }
        if let Some(v) = self.output_state.as_ref() {
            let v = changed_object::OutputObjectState::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("outputState", &v)?;
        }
        if let Some(v) = self.output_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("outputVersion", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.output_digest.as_ref() {
            struct_ser.serialize_field("outputDigest", v)?;
        }
        if let Some(v) = self.output_owner.as_ref() {
            struct_ser.serialize_field("outputOwner", v)?;
        }
        if let Some(v) = self.id_operation.as_ref() {
            let v = changed_object::IdOperation::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("idOperation", &v)?;
        }
        if let Some(v) = self.object_type.as_ref() {
            struct_ser.serialize_field("objectType", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ChangedObject {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "object_id",
            "objectId",
            "input_state",
            "inputState",
            "input_version",
            "inputVersion",
            "input_digest",
            "inputDigest",
            "input_owner",
            "inputOwner",
            "output_state",
            "outputState",
            "output_version",
            "outputVersion",
            "output_digest",
            "outputDigest",
            "output_owner",
            "outputOwner",
            "id_operation",
            "idOperation",
            "object_type",
            "objectType",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ObjectId,
            InputState,
            InputVersion,
            InputDigest,
            InputOwner,
            OutputState,
            OutputVersion,
            OutputDigest,
            OutputOwner,
            IdOperation,
            ObjectType,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "objectId" | "object_id" => Ok(GeneratedField::ObjectId),
                            "inputState" | "input_state" => Ok(GeneratedField::InputState),
                            "inputVersion" | "input_version" => Ok(GeneratedField::InputVersion),
                            "inputDigest" | "input_digest" => Ok(GeneratedField::InputDigest),
                            "inputOwner" | "input_owner" => Ok(GeneratedField::InputOwner),
                            "outputState" | "output_state" => Ok(GeneratedField::OutputState),
                            "outputVersion" | "output_version" => Ok(GeneratedField::OutputVersion),
                            "outputDigest" | "output_digest" => Ok(GeneratedField::OutputDigest),
                            "outputOwner" | "output_owner" => Ok(GeneratedField::OutputOwner),
                            "idOperation" | "id_operation" => Ok(GeneratedField::IdOperation),
                            "objectType" | "object_type" => Ok(GeneratedField::ObjectType),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ChangedObject;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ChangedObject")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ChangedObject, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut object_id__ = None;
                let mut input_state__ = None;
                let mut input_version__ = None;
                let mut input_digest__ = None;
                let mut input_owner__ = None;
                let mut output_state__ = None;
                let mut output_version__ = None;
                let mut output_digest__ = None;
                let mut output_owner__ = None;
                let mut id_operation__ = None;
                let mut object_type__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ObjectId => {
                            if object_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectId"));
                            }
                            object_id__ = map_.next_value()?;
                        }
                        GeneratedField::InputState => {
                            if input_state__.is_some() {
                                return Err(serde::de::Error::duplicate_field("inputState"));
                            }
                            input_state__ = map_.next_value::<::std::option::Option<changed_object::InputObjectState>>()?.map(|x| x as i32);
                        }
                        GeneratedField::InputVersion => {
                            if input_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("inputVersion"));
                            }
                            input_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::InputDigest => {
                            if input_digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("inputDigest"));
                            }
                            input_digest__ = map_.next_value()?;
                        }
                        GeneratedField::InputOwner => {
                            if input_owner__.is_some() {
                                return Err(serde::de::Error::duplicate_field("inputOwner"));
                            }
                            input_owner__ = map_.next_value()?;
                        }
                        GeneratedField::OutputState => {
                            if output_state__.is_some() {
                                return Err(serde::de::Error::duplicate_field("outputState"));
                            }
                            output_state__ = map_.next_value::<::std::option::Option<changed_object::OutputObjectState>>()?.map(|x| x as i32);
                        }
                        GeneratedField::OutputVersion => {
                            if output_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("outputVersion"));
                            }
                            output_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::OutputDigest => {
                            if output_digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("outputDigest"));
                            }
                            output_digest__ = map_.next_value()?;
                        }
                        GeneratedField::OutputOwner => {
                            if output_owner__.is_some() {
                                return Err(serde::de::Error::duplicate_field("outputOwner"));
                            }
                            output_owner__ = map_.next_value()?;
                        }
                        GeneratedField::IdOperation => {
                            if id_operation__.is_some() {
                                return Err(serde::de::Error::duplicate_field("idOperation"));
                            }
                            id_operation__ = map_.next_value::<::std::option::Option<changed_object::IdOperation>>()?.map(|x| x as i32);
                        }
                        GeneratedField::ObjectType => {
                            if object_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectType"));
                            }
                            object_type__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ChangedObject {
                    object_id: object_id__,
                    input_state: input_state__,
                    input_version: input_version__,
                    input_digest: input_digest__,
                    input_owner: input_owner__,
                    output_state: output_state__,
                    output_version: output_version__,
                    output_digest: output_digest__,
                    output_owner: output_owner__,
                    id_operation: id_operation__,
                    object_type: object_type__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ChangedObject", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for changed_object::IdOperation {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "ID_OPERATION_UNKNOWN",
            Self::None => "NONE",
            Self::Created => "CREATED",
            Self::Deleted => "DELETED",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for changed_object::IdOperation {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "ID_OPERATION_UNKNOWN",
            "NONE",
            "CREATED",
            "DELETED",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = changed_object::IdOperation;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "ID_OPERATION_UNKNOWN" => Ok(changed_object::IdOperation::Unknown),
                    "NONE" => Ok(changed_object::IdOperation::None),
                    "CREATED" => Ok(changed_object::IdOperation::Created),
                    "DELETED" => Ok(changed_object::IdOperation::Deleted),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for changed_object::InputObjectState {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "INPUT_OBJECT_STATE_UNKNOWN",
            Self::DoesNotExist => "INPUT_OBJECT_STATE_DOES_NOT_EXIST",
            Self::Exists => "INPUT_OBJECT_STATE_EXISTS",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for changed_object::InputObjectState {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "INPUT_OBJECT_STATE_UNKNOWN",
            "INPUT_OBJECT_STATE_DOES_NOT_EXIST",
            "INPUT_OBJECT_STATE_EXISTS",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = changed_object::InputObjectState;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "INPUT_OBJECT_STATE_UNKNOWN" => Ok(changed_object::InputObjectState::Unknown),
                    "INPUT_OBJECT_STATE_DOES_NOT_EXIST" => Ok(changed_object::InputObjectState::DoesNotExist),
                    "INPUT_OBJECT_STATE_EXISTS" => Ok(changed_object::InputObjectState::Exists),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for changed_object::OutputObjectState {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "OUTPUT_OBJECT_STATE_UNKNOWN",
            Self::DoesNotExist => "OUTPUT_OBJECT_STATE_DOES_NOT_EXIST",
            Self::ObjectWrite => "OUTPUT_OBJECT_STATE_OBJECT_WRITE",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for changed_object::OutputObjectState {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "OUTPUT_OBJECT_STATE_UNKNOWN",
            "OUTPUT_OBJECT_STATE_DOES_NOT_EXIST",
            "OUTPUT_OBJECT_STATE_OBJECT_WRITE",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = changed_object::OutputObjectState;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "OUTPUT_OBJECT_STATE_UNKNOWN" => Ok(changed_object::OutputObjectState::Unknown),
                    "OUTPUT_OBJECT_STATE_DOES_NOT_EXIST" => Ok(changed_object::OutputObjectState::DoesNotExist),
                    "OUTPUT_OBJECT_STATE_OBJECT_WRITE" => Ok(changed_object::OutputObjectState::ObjectWrite),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for Checkpoint {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.sequence_number.is_some() {
            len += 1;
        }
        if self.digest.is_some() {
            len += 1;
        }
        if self.summary.is_some() {
            len += 1;
        }
        if self.signature.is_some() {
            len += 1;
        }
        if self.contents.is_some() {
            len += 1;
        }
        if !self.transactions.is_empty() {
            len += 1;
        }
        if self.objects.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Checkpoint", len)?;
        if let Some(v) = self.sequence_number.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("sequenceNumber", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.summary.as_ref() {
            struct_ser.serialize_field("summary", v)?;
        }
        if let Some(v) = self.signature.as_ref() {
            struct_ser.serialize_field("signature", v)?;
        }
        if let Some(v) = self.contents.as_ref() {
            struct_ser.serialize_field("contents", v)?;
        }
        if !self.transactions.is_empty() {
            struct_ser.serialize_field("transactions", &self.transactions)?;
        }
        if let Some(v) = self.objects.as_ref() {
            struct_ser.serialize_field("objects", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Checkpoint {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "sequence_number",
            "sequenceNumber",
            "digest",
            "summary",
            "signature",
            "contents",
            "transactions",
            "objects",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            SequenceNumber,
            Digest,
            Summary,
            Signature,
            Contents,
            Transactions,
            Objects,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "sequenceNumber" | "sequence_number" => Ok(GeneratedField::SequenceNumber),
                            "digest" => Ok(GeneratedField::Digest),
                            "summary" => Ok(GeneratedField::Summary),
                            "signature" => Ok(GeneratedField::Signature),
                            "contents" => Ok(GeneratedField::Contents),
                            "transactions" => Ok(GeneratedField::Transactions),
                            "objects" => Ok(GeneratedField::Objects),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Checkpoint;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Checkpoint")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Checkpoint, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut sequence_number__ = None;
                let mut digest__ = None;
                let mut summary__ = None;
                let mut signature__ = None;
                let mut contents__ = None;
                let mut transactions__ = None;
                let mut objects__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::SequenceNumber => {
                            if sequence_number__.is_some() {
                                return Err(serde::de::Error::duplicate_field("sequenceNumber"));
                            }
                            sequence_number__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::Summary => {
                            if summary__.is_some() {
                                return Err(serde::de::Error::duplicate_field("summary"));
                            }
                            summary__ = map_.next_value()?;
                        }
                        GeneratedField::Signature => {
                            if signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signature"));
                            }
                            signature__ = map_.next_value()?;
                        }
                        GeneratedField::Contents => {
                            if contents__.is_some() {
                                return Err(serde::de::Error::duplicate_field("contents"));
                            }
                            contents__ = map_.next_value()?;
                        }
                        GeneratedField::Transactions => {
                            if transactions__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transactions"));
                            }
                            transactions__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Objects => {
                            if objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objects"));
                            }
                            objects__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Checkpoint {
                    sequence_number: sequence_number__,
                    digest: digest__,
                    summary: summary__,
                    signature: signature__,
                    contents: contents__,
                    transactions: transactions__.unwrap_or_default(),
                    objects: objects__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Checkpoint", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CheckpointCommitment {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.kind.is_some() {
            len += 1;
        }
        if self.digest.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CheckpointCommitment", len)?;
        if let Some(v) = self.kind.as_ref() {
            let v = checkpoint_commitment::CheckpointCommitmentKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CheckpointCommitment {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "digest",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            Digest,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "kind" => Ok(GeneratedField::Kind),
                            "digest" => Ok(GeneratedField::Digest),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = CheckpointCommitment;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CheckpointCommitment")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CheckpointCommitment, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut digest__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<checkpoint_commitment::CheckpointCommitmentKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CheckpointCommitment {
                    kind: kind__,
                    digest: digest__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CheckpointCommitment", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for checkpoint_commitment::CheckpointCommitmentKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "CHECKPOINT_COMMITMENT_KIND_UNKNOWN",
            Self::EcmhLiveObjectSet => "ECMH_LIVE_OBJECT_SET",
            Self::CheckpointArtifacts => "CHECKPOINT_ARTIFACTS",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for checkpoint_commitment::CheckpointCommitmentKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "CHECKPOINT_COMMITMENT_KIND_UNKNOWN",
            "ECMH_LIVE_OBJECT_SET",
            "CHECKPOINT_ARTIFACTS",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = checkpoint_commitment::CheckpointCommitmentKind;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "CHECKPOINT_COMMITMENT_KIND_UNKNOWN" => Ok(checkpoint_commitment::CheckpointCommitmentKind::Unknown),
                    "ECMH_LIVE_OBJECT_SET" => Ok(checkpoint_commitment::CheckpointCommitmentKind::EcmhLiveObjectSet),
                    "CHECKPOINT_ARTIFACTS" => Ok(checkpoint_commitment::CheckpointCommitmentKind::CheckpointArtifacts),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for CheckpointContents {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.digest.is_some() {
            len += 1;
        }
        if self.version.is_some() {
            len += 1;
        }
        if !self.transactions.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CheckpointContents", len)?;
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.version.as_ref() {
            struct_ser.serialize_field("version", v)?;
        }
        if !self.transactions.is_empty() {
            struct_ser.serialize_field("transactions", &self.transactions)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CheckpointContents {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "digest",
            "version",
            "transactions",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Digest,
            Version,
            Transactions,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "digest" => Ok(GeneratedField::Digest),
                            "version" => Ok(GeneratedField::Version),
                            "transactions" => Ok(GeneratedField::Transactions),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = CheckpointContents;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CheckpointContents")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CheckpointContents, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut digest__ = None;
                let mut version__ = None;
                let mut transactions__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Transactions => {
                            if transactions__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transactions"));
                            }
                            transactions__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CheckpointContents {
                    digest: digest__,
                    version: version__,
                    transactions: transactions__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CheckpointContents", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CheckpointSummary {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.digest.is_some() {
            len += 1;
        }
        if self.epoch.is_some() {
            len += 1;
        }
        if self.sequence_number.is_some() {
            len += 1;
        }
        if self.total_network_transactions.is_some() {
            len += 1;
        }
        if self.content_digest.is_some() {
            len += 1;
        }
        if self.previous_digest.is_some() {
            len += 1;
        }
        if self.epoch_rolling_transaction_fees.is_some() {
            len += 1;
        }
        if self.timestamp.is_some() {
            len += 1;
        }
        if !self.commitments.is_empty() {
            len += 1;
        }
        if self.end_of_epoch_data.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CheckpointSummary", len)?;
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.sequence_number.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("sequenceNumber", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.total_network_transactions.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("totalNetworkTransactions", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.content_digest.as_ref() {
            struct_ser.serialize_field("contentDigest", v)?;
        }
        if let Some(v) = self.previous_digest.as_ref() {
            struct_ser.serialize_field("previousDigest", v)?;
        }
        if let Some(v) = self.epoch_rolling_transaction_fees.as_ref() {
            struct_ser.serialize_field("epochRollingTransactionFees", v)?;
        }
        if let Some(v) = self.timestamp.as_ref() {
            struct_ser.serialize_field("timestamp", &crate::utils::_serde::TimestampSerializer(v))?;
        }
        if !self.commitments.is_empty() {
            struct_ser.serialize_field("commitments", &self.commitments)?;
        }
        if let Some(v) = self.end_of_epoch_data.as_ref() {
            struct_ser.serialize_field("endOfEpochData", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CheckpointSummary {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "digest",
            "epoch",
            "sequence_number",
            "sequenceNumber",
            "total_network_transactions",
            "totalNetworkTransactions",
            "content_digest",
            "contentDigest",
            "previous_digest",
            "previousDigest",
            "epoch_rolling_transaction_fees",
            "epochRollingTransactionFees",
            "timestamp",
            "commitments",
            "end_of_epoch_data",
            "endOfEpochData",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Digest,
            Epoch,
            SequenceNumber,
            TotalNetworkTransactions,
            ContentDigest,
            PreviousDigest,
            EpochRollingTransactionFees,
            Timestamp,
            Commitments,
            EndOfEpochData,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "digest" => Ok(GeneratedField::Digest),
                            "epoch" => Ok(GeneratedField::Epoch),
                            "sequenceNumber" | "sequence_number" => Ok(GeneratedField::SequenceNumber),
                            "totalNetworkTransactions" | "total_network_transactions" => Ok(GeneratedField::TotalNetworkTransactions),
                            "contentDigest" | "content_digest" => Ok(GeneratedField::ContentDigest),
                            "previousDigest" | "previous_digest" => Ok(GeneratedField::PreviousDigest),
                            "epochRollingTransactionFees" | "epoch_rolling_transaction_fees" => Ok(GeneratedField::EpochRollingTransactionFees),
                            "timestamp" => Ok(GeneratedField::Timestamp),
                            "commitments" => Ok(GeneratedField::Commitments),
                            "endOfEpochData" | "end_of_epoch_data" => Ok(GeneratedField::EndOfEpochData),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = CheckpointSummary;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CheckpointSummary")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CheckpointSummary, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut digest__ = None;
                let mut epoch__ = None;
                let mut sequence_number__ = None;
                let mut total_network_transactions__ = None;
                let mut content_digest__ = None;
                let mut previous_digest__ = None;
                let mut epoch_rolling_transaction_fees__ = None;
                let mut timestamp__ = None;
                let mut commitments__ = None;
                let mut end_of_epoch_data__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::SequenceNumber => {
                            if sequence_number__.is_some() {
                                return Err(serde::de::Error::duplicate_field("sequenceNumber"));
                            }
                            sequence_number__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TotalNetworkTransactions => {
                            if total_network_transactions__.is_some() {
                                return Err(serde::de::Error::duplicate_field("totalNetworkTransactions"));
                            }
                            total_network_transactions__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ContentDigest => {
                            if content_digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("contentDigest"));
                            }
                            content_digest__ = map_.next_value()?;
                        }
                        GeneratedField::PreviousDigest => {
                            if previous_digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("previousDigest"));
                            }
                            previous_digest__ = map_.next_value()?;
                        }
                        GeneratedField::EpochRollingTransactionFees => {
                            if epoch_rolling_transaction_fees__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epochRollingTransactionFees"));
                            }
                            epoch_rolling_transaction_fees__ = map_.next_value()?;
                        }
                        GeneratedField::Timestamp => {
                            if timestamp__.is_some() {
                                return Err(serde::de::Error::duplicate_field("timestamp"));
                            }
                            timestamp__ = map_.next_value::<::std::option::Option<crate::utils::_serde::TimestampDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::Commitments => {
                            if commitments__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commitments"));
                            }
                            commitments__ = Some(map_.next_value()?);
                        }
                        GeneratedField::EndOfEpochData => {
                            if end_of_epoch_data__.is_some() {
                                return Err(serde::de::Error::duplicate_field("endOfEpochData"));
                            }
                            end_of_epoch_data__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CheckpointSummary {
                    digest: digest__,
                    epoch: epoch__,
                    sequence_number: sequence_number__,
                    total_network_transactions: total_network_transactions__,
                    content_digest: content_digest__,
                    previous_digest: previous_digest__,
                    epoch_rolling_transaction_fees: epoch_rolling_transaction_fees__,
                    timestamp: timestamp__,
                    commitments: commitments__.unwrap_or_default(),
                    end_of_epoch_data: end_of_epoch_data__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CheckpointSummary", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CheckpointedTransactionInfo {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.transaction.is_some() {
            len += 1;
        }
        if self.effects.is_some() {
            len += 1;
        }
        if !self.signatures.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CheckpointedTransactionInfo", len)?;
        if let Some(v) = self.transaction.as_ref() {
            struct_ser.serialize_field("transaction", v)?;
        }
        if let Some(v) = self.effects.as_ref() {
            struct_ser.serialize_field("effects", v)?;
        }
        if !self.signatures.is_empty() {
            struct_ser.serialize_field("signatures", &self.signatures)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CheckpointedTransactionInfo {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "transaction",
            "effects",
            "signatures",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Transaction,
            Effects,
            Signatures,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "transaction" => Ok(GeneratedField::Transaction),
                            "effects" => Ok(GeneratedField::Effects),
                            "signatures" => Ok(GeneratedField::Signatures),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = CheckpointedTransactionInfo;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CheckpointedTransactionInfo")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CheckpointedTransactionInfo, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut transaction__ = None;
                let mut effects__ = None;
                let mut signatures__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Transaction => {
                            if transaction__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transaction"));
                            }
                            transaction__ = map_.next_value()?;
                        }
                        GeneratedField::Effects => {
                            if effects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("effects"));
                            }
                            effects__ = map_.next_value()?;
                        }
                        GeneratedField::Signatures => {
                            if signatures__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signatures"));
                            }
                            signatures__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CheckpointedTransactionInfo {
                    transaction: transaction__,
                    effects: effects__,
                    signatures: signatures__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CheckpointedTransactionInfo", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ClaimChallengeBond {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.challenge_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ClaimChallengeBond", len)?;
        if let Some(v) = self.challenge_id.as_ref() {
            struct_ser.serialize_field("challengeId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ClaimChallengeBond {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "challenge_id",
            "challengeId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ChallengeId,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "challengeId" | "challenge_id" => Ok(GeneratedField::ChallengeId),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ClaimChallengeBond;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ClaimChallengeBond")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ClaimChallengeBond, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut challenge_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ChallengeId => {
                            if challenge_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challengeId"));
                            }
                            challenge_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ClaimChallengeBond {
                    challenge_id: challenge_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ClaimChallengeBond", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ClaimRewards {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.target_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ClaimRewards", len)?;
        if let Some(v) = self.target_id.as_ref() {
            struct_ser.serialize_field("targetId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ClaimRewards {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "target_id",
            "targetId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TargetId,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "targetId" | "target_id" => Ok(GeneratedField::TargetId),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ClaimRewards;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ClaimRewards")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ClaimRewards, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut target_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::TargetId => {
                            if target_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetId"));
                            }
                            target_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ClaimRewards {
                    target_id: target_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ClaimRewards", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CommitModel {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.model_id.is_some() {
            len += 1;
        }
        if self.weights_url_commitment.is_some() {
            len += 1;
        }
        if self.weights_commitment.is_some() {
            len += 1;
        }
        if self.architecture_version.is_some() {
            len += 1;
        }
        if self.stake_amount.is_some() {
            len += 1;
        }
        if self.commission_rate.is_some() {
            len += 1;
        }
        if self.staking_pool_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CommitModel", len)?;
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        if let Some(v) = self.weights_url_commitment.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("weightsUrlCommitment", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.weights_commitment.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("weightsCommitment", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.architecture_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("architectureVersion", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.stake_amount.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("stakeAmount", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.commission_rate.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("commissionRate", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.staking_pool_id.as_ref() {
            struct_ser.serialize_field("stakingPoolId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CommitModel {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "model_id",
            "modelId",
            "weights_url_commitment",
            "weightsUrlCommitment",
            "weights_commitment",
            "weightsCommitment",
            "architecture_version",
            "architectureVersion",
            "stake_amount",
            "stakeAmount",
            "commission_rate",
            "commissionRate",
            "staking_pool_id",
            "stakingPoolId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ModelId,
            WeightsUrlCommitment,
            WeightsCommitment,
            ArchitectureVersion,
            StakeAmount,
            CommissionRate,
            StakingPoolId,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            "weightsUrlCommitment" | "weights_url_commitment" => Ok(GeneratedField::WeightsUrlCommitment),
                            "weightsCommitment" | "weights_commitment" => Ok(GeneratedField::WeightsCommitment),
                            "architectureVersion" | "architecture_version" => Ok(GeneratedField::ArchitectureVersion),
                            "stakeAmount" | "stake_amount" => Ok(GeneratedField::StakeAmount),
                            "commissionRate" | "commission_rate" => Ok(GeneratedField::CommissionRate),
                            "stakingPoolId" | "staking_pool_id" => Ok(GeneratedField::StakingPoolId),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = CommitModel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CommitModel")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CommitModel, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut model_id__ = None;
                let mut weights_url_commitment__ = None;
                let mut weights_commitment__ = None;
                let mut architecture_version__ = None;
                let mut stake_amount__ = None;
                let mut commission_rate__ = None;
                let mut staking_pool_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::WeightsUrlCommitment => {
                            if weights_url_commitment__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weightsUrlCommitment"));
                            }
                            weights_url_commitment__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::WeightsCommitment => {
                            if weights_commitment__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weightsCommitment"));
                            }
                            weights_commitment__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ArchitectureVersion => {
                            if architecture_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("architectureVersion"));
                            }
                            architecture_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::StakeAmount => {
                            if stake_amount__.is_some() {
                                return Err(serde::de::Error::duplicate_field("stakeAmount"));
                            }
                            stake_amount__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::CommissionRate => {
                            if commission_rate__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commissionRate"));
                            }
                            commission_rate__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::StakingPoolId => {
                            if staking_pool_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("stakingPoolId"));
                            }
                            staking_pool_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CommitModel {
                    model_id: model_id__,
                    weights_url_commitment: weights_url_commitment__,
                    weights_commitment: weights_commitment__,
                    architecture_version: architecture_version__,
                    stake_amount: stake_amount__,
                    commission_rate: commission_rate__,
                    staking_pool_id: staking_pool_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CommitModel", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CommitModelUpdate {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.model_id.is_some() {
            len += 1;
        }
        if self.weights_url_commitment.is_some() {
            len += 1;
        }
        if self.weights_commitment.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CommitModelUpdate", len)?;
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        if let Some(v) = self.weights_url_commitment.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("weightsUrlCommitment", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.weights_commitment.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("weightsCommitment", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CommitModelUpdate {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "model_id",
            "modelId",
            "weights_url_commitment",
            "weightsUrlCommitment",
            "weights_commitment",
            "weightsCommitment",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ModelId,
            WeightsUrlCommitment,
            WeightsCommitment,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            "weightsUrlCommitment" | "weights_url_commitment" => Ok(GeneratedField::WeightsUrlCommitment),
                            "weightsCommitment" | "weights_commitment" => Ok(GeneratedField::WeightsCommitment),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = CommitModelUpdate;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CommitModelUpdate")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CommitModelUpdate, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut model_id__ = None;
                let mut weights_url_commitment__ = None;
                let mut weights_commitment__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::WeightsUrlCommitment => {
                            if weights_url_commitment__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weightsUrlCommitment"));
                            }
                            weights_url_commitment__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::WeightsCommitment => {
                            if weights_commitment__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weightsCommitment"));
                            }
                            weights_commitment__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CommitModelUpdate {
                    model_id: model_id__,
                    weights_url_commitment: weights_url_commitment__,
                    weights_commitment: weights_commitment__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CommitModelUpdate", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ConsensusCommitPrologue {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.epoch.is_some() {
            len += 1;
        }
        if self.round.is_some() {
            len += 1;
        }
        if self.sub_dag_index.is_some() {
            len += 1;
        }
        if self.commit_timestamp.is_some() {
            len += 1;
        }
        if self.consensus_commit_digest.is_some() {
            len += 1;
        }
        if self.additional_state_digest.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ConsensusCommitPrologue", len)?;
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.round.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("round", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.sub_dag_index.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("subDagIndex", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.commit_timestamp.as_ref() {
            struct_ser.serialize_field("commitTimestamp", &crate::utils::_serde::TimestampSerializer(v))?;
        }
        if let Some(v) = self.consensus_commit_digest.as_ref() {
            struct_ser.serialize_field("consensusCommitDigest", v)?;
        }
        if let Some(v) = self.additional_state_digest.as_ref() {
            struct_ser.serialize_field("additionalStateDigest", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ConsensusCommitPrologue {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epoch",
            "round",
            "sub_dag_index",
            "subDagIndex",
            "commit_timestamp",
            "commitTimestamp",
            "consensus_commit_digest",
            "consensusCommitDigest",
            "additional_state_digest",
            "additionalStateDigest",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            Round,
            SubDagIndex,
            CommitTimestamp,
            ConsensusCommitDigest,
            AdditionalStateDigest,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "epoch" => Ok(GeneratedField::Epoch),
                            "round" => Ok(GeneratedField::Round),
                            "subDagIndex" | "sub_dag_index" => Ok(GeneratedField::SubDagIndex),
                            "commitTimestamp" | "commit_timestamp" => Ok(GeneratedField::CommitTimestamp),
                            "consensusCommitDigest" | "consensus_commit_digest" => Ok(GeneratedField::ConsensusCommitDigest),
                            "additionalStateDigest" | "additional_state_digest" => Ok(GeneratedField::AdditionalStateDigest),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ConsensusCommitPrologue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ConsensusCommitPrologue")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ConsensusCommitPrologue, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epoch__ = None;
                let mut round__ = None;
                let mut sub_dag_index__ = None;
                let mut commit_timestamp__ = None;
                let mut consensus_commit_digest__ = None;
                let mut additional_state_digest__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Round => {
                            if round__.is_some() {
                                return Err(serde::de::Error::duplicate_field("round"));
                            }
                            round__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::SubDagIndex => {
                            if sub_dag_index__.is_some() {
                                return Err(serde::de::Error::duplicate_field("subDagIndex"));
                            }
                            sub_dag_index__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::CommitTimestamp => {
                            if commit_timestamp__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commitTimestamp"));
                            }
                            commit_timestamp__ = map_.next_value::<::std::option::Option<crate::utils::_serde::TimestampDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::ConsensusCommitDigest => {
                            if consensus_commit_digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("consensusCommitDigest"));
                            }
                            consensus_commit_digest__ = map_.next_value()?;
                        }
                        GeneratedField::AdditionalStateDigest => {
                            if additional_state_digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("additionalStateDigest"));
                            }
                            additional_state_digest__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ConsensusCommitPrologue {
                    epoch: epoch__,
                    round: round__,
                    sub_dag_index: sub_dag_index__,
                    commit_timestamp: commit_timestamp__,
                    consensus_commit_digest: consensus_commit_digest__,
                    additional_state_digest: additional_state_digest__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ConsensusCommitPrologue", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for DeactivateModel {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.model_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.DeactivateModel", len)?;
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for DeactivateModel {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "model_id",
            "modelId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ModelId,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = DeactivateModel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.DeactivateModel")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<DeactivateModel, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut model_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(DeactivateModel {
                    model_id: model_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.DeactivateModel", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for EmissionPool {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.balance.is_some() {
            len += 1;
        }
        if self.emission_per_epoch.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.EmissionPool", len)?;
        if let Some(v) = self.balance.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("balance", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.emission_per_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("emissionPerEpoch", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for EmissionPool {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "balance",
            "emission_per_epoch",
            "emissionPerEpoch",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Balance,
            EmissionPerEpoch,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "balance" => Ok(GeneratedField::Balance),
                            "emissionPerEpoch" | "emission_per_epoch" => Ok(GeneratedField::EmissionPerEpoch),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = EmissionPool;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.EmissionPool")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<EmissionPool, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut balance__ = None;
                let mut emission_per_epoch__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Balance => {
                            if balance__.is_some() {
                                return Err(serde::de::Error::duplicate_field("balance"));
                            }
                            balance__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::EmissionPerEpoch => {
                            if emission_per_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("emissionPerEpoch"));
                            }
                            emission_per_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(EmissionPool {
                    balance: balance__,
                    emission_per_epoch: emission_per_epoch__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.EmissionPool", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for EndOfEpochData {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.next_epoch_validator_committee.is_some() {
            len += 1;
        }
        if self.next_epoch_protocol_version.is_some() {
            len += 1;
        }
        if !self.epoch_commitments.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.EndOfEpochData", len)?;
        if let Some(v) = self.next_epoch_validator_committee.as_ref() {
            struct_ser.serialize_field("nextEpochValidatorCommittee", v)?;
        }
        if let Some(v) = self.next_epoch_protocol_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochProtocolVersion", ToString::to_string(&v).as_str())?;
        }
        if !self.epoch_commitments.is_empty() {
            struct_ser.serialize_field("epochCommitments", &self.epoch_commitments)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for EndOfEpochData {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "next_epoch_validator_committee",
            "nextEpochValidatorCommittee",
            "next_epoch_protocol_version",
            "nextEpochProtocolVersion",
            "epoch_commitments",
            "epochCommitments",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            NextEpochValidatorCommittee,
            NextEpochProtocolVersion,
            EpochCommitments,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "nextEpochValidatorCommittee" | "next_epoch_validator_committee" => Ok(GeneratedField::NextEpochValidatorCommittee),
                            "nextEpochProtocolVersion" | "next_epoch_protocol_version" => Ok(GeneratedField::NextEpochProtocolVersion),
                            "epochCommitments" | "epoch_commitments" => Ok(GeneratedField::EpochCommitments),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = EndOfEpochData;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.EndOfEpochData")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<EndOfEpochData, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut next_epoch_validator_committee__ = None;
                let mut next_epoch_protocol_version__ = None;
                let mut epoch_commitments__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::NextEpochValidatorCommittee => {
                            if next_epoch_validator_committee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochValidatorCommittee"));
                            }
                            next_epoch_validator_committee__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochProtocolVersion => {
                            if next_epoch_protocol_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochProtocolVersion"));
                            }
                            next_epoch_protocol_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::EpochCommitments => {
                            if epoch_commitments__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epochCommitments"));
                            }
                            epoch_commitments__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(EndOfEpochData {
                    next_epoch_validator_committee: next_epoch_validator_committee__,
                    next_epoch_protocol_version: next_epoch_protocol_version__,
                    epoch_commitments: epoch_commitments__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.EndOfEpochData", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Epoch {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.epoch.is_some() {
            len += 1;
        }
        if self.committee.is_some() {
            len += 1;
        }
        if self.system_state.is_some() {
            len += 1;
        }
        if self.first_checkpoint.is_some() {
            len += 1;
        }
        if self.last_checkpoint.is_some() {
            len += 1;
        }
        if self.start.is_some() {
            len += 1;
        }
        if self.end.is_some() {
            len += 1;
        }
        if self.protocol_config.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Epoch", len)?;
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.committee.as_ref() {
            struct_ser.serialize_field("committee", v)?;
        }
        if let Some(v) = self.system_state.as_ref() {
            struct_ser.serialize_field("systemState", v)?;
        }
        if let Some(v) = self.first_checkpoint.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("firstCheckpoint", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.last_checkpoint.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("lastCheckpoint", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.start.as_ref() {
            struct_ser.serialize_field("start", &crate::utils::_serde::TimestampSerializer(v))?;
        }
        if let Some(v) = self.end.as_ref() {
            struct_ser.serialize_field("end", &crate::utils::_serde::TimestampSerializer(v))?;
        }
        if let Some(v) = self.protocol_config.as_ref() {
            struct_ser.serialize_field("protocolConfig", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Epoch {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epoch",
            "committee",
            "system_state",
            "systemState",
            "first_checkpoint",
            "firstCheckpoint",
            "last_checkpoint",
            "lastCheckpoint",
            "start",
            "end",
            "protocol_config",
            "protocolConfig",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            Committee,
            SystemState,
            FirstCheckpoint,
            LastCheckpoint,
            Start,
            End,
            ProtocolConfig,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "epoch" => Ok(GeneratedField::Epoch),
                            "committee" => Ok(GeneratedField::Committee),
                            "systemState" | "system_state" => Ok(GeneratedField::SystemState),
                            "firstCheckpoint" | "first_checkpoint" => Ok(GeneratedField::FirstCheckpoint),
                            "lastCheckpoint" | "last_checkpoint" => Ok(GeneratedField::LastCheckpoint),
                            "start" => Ok(GeneratedField::Start),
                            "end" => Ok(GeneratedField::End),
                            "protocolConfig" | "protocol_config" => Ok(GeneratedField::ProtocolConfig),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Epoch;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Epoch")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Epoch, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epoch__ = None;
                let mut committee__ = None;
                let mut system_state__ = None;
                let mut first_checkpoint__ = None;
                let mut last_checkpoint__ = None;
                let mut start__ = None;
                let mut end__ = None;
                let mut protocol_config__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Committee => {
                            if committee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("committee"));
                            }
                            committee__ = map_.next_value()?;
                        }
                        GeneratedField::SystemState => {
                            if system_state__.is_some() {
                                return Err(serde::de::Error::duplicate_field("systemState"));
                            }
                            system_state__ = map_.next_value()?;
                        }
                        GeneratedField::FirstCheckpoint => {
                            if first_checkpoint__.is_some() {
                                return Err(serde::de::Error::duplicate_field("firstCheckpoint"));
                            }
                            first_checkpoint__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::LastCheckpoint => {
                            if last_checkpoint__.is_some() {
                                return Err(serde::de::Error::duplicate_field("lastCheckpoint"));
                            }
                            last_checkpoint__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Start => {
                            if start__.is_some() {
                                return Err(serde::de::Error::duplicate_field("start"));
                            }
                            start__ = map_.next_value::<::std::option::Option<crate::utils::_serde::TimestampDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::End => {
                            if end__.is_some() {
                                return Err(serde::de::Error::duplicate_field("end"));
                            }
                            end__ = map_.next_value::<::std::option::Option<crate::utils::_serde::TimestampDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::ProtocolConfig => {
                            if protocol_config__.is_some() {
                                return Err(serde::de::Error::duplicate_field("protocolConfig"));
                            }
                            protocol_config__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Epoch {
                    epoch: epoch__,
                    committee: committee__,
                    system_state: system_state__,
                    first_checkpoint: first_checkpoint__,
                    last_checkpoint: last_checkpoint__,
                    start: start__,
                    end: end__,
                    protocol_config: protocol_config__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Epoch", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ErrorReason {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "ERROR_REASON_UNKNOWN",
            Self::FieldInvalid => "FIELD_INVALID",
            Self::FieldMissing => "FIELD_MISSING",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for ErrorReason {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "ERROR_REASON_UNKNOWN",
            "FIELD_INVALID",
            "FIELD_MISSING",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ErrorReason;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "ERROR_REASON_UNKNOWN" => Ok(ErrorReason::Unknown),
                    "FIELD_INVALID" => Ok(ErrorReason::FieldInvalid),
                    "FIELD_MISSING" => Ok(ErrorReason::FieldMissing),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for ExecuteTransactionRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.transaction.is_some() {
            len += 1;
        }
        if !self.signatures.is_empty() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ExecuteTransactionRequest", len)?;
        if let Some(v) = self.transaction.as_ref() {
            struct_ser.serialize_field("transaction", v)?;
        }
        if !self.signatures.is_empty() {
            struct_ser.serialize_field("signatures", &self.signatures)?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ExecuteTransactionRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "transaction",
            "signatures",
            "read_mask",
            "readMask",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Transaction,
            Signatures,
            ReadMask,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "transaction" => Ok(GeneratedField::Transaction),
                            "signatures" => Ok(GeneratedField::Signatures),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ExecuteTransactionRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ExecuteTransactionRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ExecuteTransactionRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut transaction__ = None;
                let mut signatures__ = None;
                let mut read_mask__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Transaction => {
                            if transaction__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transaction"));
                            }
                            transaction__ = map_.next_value()?;
                        }
                        GeneratedField::Signatures => {
                            if signatures__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signatures"));
                            }
                            signatures__ = Some(map_.next_value()?);
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ExecuteTransactionRequest {
                    transaction: transaction__,
                    signatures: signatures__.unwrap_or_default(),
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ExecuteTransactionRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ExecuteTransactionResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.transaction.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ExecuteTransactionResponse", len)?;
        if let Some(v) = self.transaction.as_ref() {
            struct_ser.serialize_field("transaction", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ExecuteTransactionResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "transaction",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Transaction,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "transaction" => Ok(GeneratedField::Transaction),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ExecuteTransactionResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ExecuteTransactionResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ExecuteTransactionResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut transaction__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Transaction => {
                            if transaction__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transaction"));
                            }
                            transaction__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ExecuteTransactionResponse {
                    transaction: transaction__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ExecuteTransactionResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ExecutedTransaction {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.digest.is_some() {
            len += 1;
        }
        if self.transaction.is_some() {
            len += 1;
        }
        if !self.signatures.is_empty() {
            len += 1;
        }
        if self.effects.is_some() {
            len += 1;
        }
        if self.checkpoint.is_some() {
            len += 1;
        }
        if self.timestamp.is_some() {
            len += 1;
        }
        if !self.balance_changes.is_empty() {
            len += 1;
        }
        if self.objects.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ExecutedTransaction", len)?;
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.transaction.as_ref() {
            struct_ser.serialize_field("transaction", v)?;
        }
        if !self.signatures.is_empty() {
            struct_ser.serialize_field("signatures", &self.signatures)?;
        }
        if let Some(v) = self.effects.as_ref() {
            struct_ser.serialize_field("effects", v)?;
        }
        if let Some(v) = self.checkpoint.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("checkpoint", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.timestamp.as_ref() {
            struct_ser.serialize_field("timestamp", &crate::utils::_serde::TimestampSerializer(v))?;
        }
        if !self.balance_changes.is_empty() {
            struct_ser.serialize_field("balanceChanges", &self.balance_changes)?;
        }
        if let Some(v) = self.objects.as_ref() {
            struct_ser.serialize_field("objects", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ExecutedTransaction {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "digest",
            "transaction",
            "signatures",
            "effects",
            "checkpoint",
            "timestamp",
            "balance_changes",
            "balanceChanges",
            "objects",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Digest,
            Transaction,
            Signatures,
            Effects,
            Checkpoint,
            Timestamp,
            BalanceChanges,
            Objects,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "digest" => Ok(GeneratedField::Digest),
                            "transaction" => Ok(GeneratedField::Transaction),
                            "signatures" => Ok(GeneratedField::Signatures),
                            "effects" => Ok(GeneratedField::Effects),
                            "checkpoint" => Ok(GeneratedField::Checkpoint),
                            "timestamp" => Ok(GeneratedField::Timestamp),
                            "balanceChanges" | "balance_changes" => Ok(GeneratedField::BalanceChanges),
                            "objects" => Ok(GeneratedField::Objects),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ExecutedTransaction;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ExecutedTransaction")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ExecutedTransaction, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut digest__ = None;
                let mut transaction__ = None;
                let mut signatures__ = None;
                let mut effects__ = None;
                let mut checkpoint__ = None;
                let mut timestamp__ = None;
                let mut balance_changes__ = None;
                let mut objects__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::Transaction => {
                            if transaction__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transaction"));
                            }
                            transaction__ = map_.next_value()?;
                        }
                        GeneratedField::Signatures => {
                            if signatures__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signatures"));
                            }
                            signatures__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Effects => {
                            if effects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("effects"));
                            }
                            effects__ = map_.next_value()?;
                        }
                        GeneratedField::Checkpoint => {
                            if checkpoint__.is_some() {
                                return Err(serde::de::Error::duplicate_field("checkpoint"));
                            }
                            checkpoint__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Timestamp => {
                            if timestamp__.is_some() {
                                return Err(serde::de::Error::duplicate_field("timestamp"));
                            }
                            timestamp__ = map_.next_value::<::std::option::Option<crate::utils::_serde::TimestampDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::BalanceChanges => {
                            if balance_changes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("balanceChanges"));
                            }
                            balance_changes__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Objects => {
                            if objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objects"));
                            }
                            objects__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ExecutedTransaction {
                    digest: digest__,
                    transaction: transaction__,
                    signatures: signatures__.unwrap_or_default(),
                    effects: effects__,
                    checkpoint: checkpoint__,
                    timestamp: timestamp__,
                    balance_changes: balance_changes__.unwrap_or_default(),
                    objects: objects__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ExecutedTransaction", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ExecutionError {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.description.is_some() {
            len += 1;
        }
        if self.kind.is_some() {
            len += 1;
        }
        if self.error_details.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ExecutionError", len)?;
        if let Some(v) = self.description.as_ref() {
            struct_ser.serialize_field("description", v)?;
        }
        if let Some(v) = self.kind.as_ref() {
            let v = execution_error::ExecutionErrorKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.error_details.as_ref() {
            match v {
                execution_error::ErrorDetails::ObjectId(v) => {
                    struct_ser.serialize_field("objectId", v)?;
                }
                execution_error::ErrorDetails::OtherError(v) => {
                    struct_ser.serialize_field("otherError", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ExecutionError {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "description",
            "kind",
            "object_id",
            "objectId",
            "other_error",
            "otherError",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Description,
            Kind,
            ObjectId,
            OtherError,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "description" => Ok(GeneratedField::Description),
                            "kind" => Ok(GeneratedField::Kind),
                            "objectId" | "object_id" => Ok(GeneratedField::ObjectId),
                            "otherError" | "other_error" => Ok(GeneratedField::OtherError),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ExecutionError;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ExecutionError")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ExecutionError, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut description__ = None;
                let mut kind__ = None;
                let mut error_details__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Description => {
                            if description__.is_some() {
                                return Err(serde::de::Error::duplicate_field("description"));
                            }
                            description__ = map_.next_value()?;
                        }
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<execution_error::ExecutionErrorKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::ObjectId => {
                            if error_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectId"));
                            }
                            error_details__ = map_.next_value::<::std::option::Option<_>>()?.map(execution_error::ErrorDetails::ObjectId);
                        }
                        GeneratedField::OtherError => {
                            if error_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("otherError"));
                            }
                            error_details__ = map_.next_value::<::std::option::Option<_>>()?.map(execution_error::ErrorDetails::OtherError);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ExecutionError {
                    description: description__,
                    kind: kind__,
                    error_details: error_details__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ExecutionError", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for execution_error::ExecutionErrorKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "EXECUTION_ERROR_KIND_UNKNOWN",
            Self::InsufficientGas => "INSUFFICIENT_GAS",
            Self::InvalidOwnership => "INVALID_OWNERSHIP",
            Self::ObjectNotFound => "OBJECT_NOT_FOUND",
            Self::InvalidObjectType => "INVALID_OBJECT_TYPE",
            Self::InvalidTransactionType => "INVALID_TRANSACTION_TYPE",
            Self::InvalidArguments => "INVALID_ARGUMENTS",
            Self::DuplicateValidator => "DUPLICATE_VALIDATOR",
            Self::NotAValidator => "NOT_A_VALIDATOR",
            Self::ValidatorAlreadyRemoved => "VALIDATOR_ALREADY_REMOVED",
            Self::AdvancedToWrongEpoch => "ADVANCED_TO_WRONG_EPOCH",
            Self::ModelNotFound => "MODEL_NOT_FOUND",
            Self::NotModelOwner => "NOT_MODEL_OWNER",
            Self::ModelNotActive => "MODEL_NOT_ACTIVE",
            Self::InsufficientCoinBalance => "INSUFFICIENT_COIN_BALANCE",
            Self::CoinBalanceOverflow => "COIN_BALANCE_OVERFLOW",
            Self::ValidatorNotFound => "VALIDATOR_NOT_FOUND",
            Self::StakingPoolNotFound => "STAKING_POOL_NOT_FOUND",
            Self::CannotReportOneself => "CANNOT_REPORT_ONESELF",
            Self::ReportRecordNotFound => "REPORT_RECORD_NOT_FOUND",
            Self::InputObjectDeleted => "INPUT_OBJECT_DELETED",
            Self::CertificateDenied => "CERTIFICATE_DENIED",
            Self::SharedObjectCongestion => "SHARED_OBJECT_CONGESTION",
            Self::OtherError => "OTHER_ERROR",
            Self::ModelNotPending => "MODEL_NOT_PENDING",
            Self::ModelAlreadyInactive => "MODEL_ALREADY_INACTIVE",
            Self::ModelRevealEpochMismatch => "MODEL_REVEAL_EPOCH_MISMATCH",
            Self::ModelWeightsUrlMismatch => "MODEL_WEIGHTS_URL_MISMATCH",
            Self::ModelNoPendingUpdate => "MODEL_NO_PENDING_UPDATE",
            Self::ModelArchitectureVersionMismatch => "MODEL_ARCHITECTURE_VERSION_MISMATCH",
            Self::ModelCommissionRateTooHigh => "MODEL_COMMISSION_RATE_TOO_HIGH",
            Self::ModelMinStakeNotMet => "MODEL_MIN_STAKE_NOT_MET",
            Self::NoActiveModels => "NO_ACTIVE_MODELS",
            Self::TargetNotFound => "TARGET_NOT_FOUND",
            Self::TargetNotOpen => "TARGET_NOT_OPEN",
            Self::TargetExpired => "TARGET_EXPIRED",
            Self::TargetNotFilled => "TARGET_NOT_FILLED",
            Self::ChallengeWindowOpen => "CHALLENGE_WINDOW_OPEN",
            Self::TargetAlreadyClaimed => "TARGET_ALREADY_CLAIMED",
            Self::ModelNotInTarget => "MODEL_NOT_IN_TARGET",
            Self::EmbeddingDimensionMismatch => "EMBEDDING_DIMENSION_MISMATCH",
            Self::DistanceExceedsThreshold => "DISTANCE_EXCEEDS_THRESHOLD",
            Self::InsufficientBond => "INSUFFICIENT_BOND",
            Self::InsufficientEmissionBalance => "INSUFFICIENT_EMISSION_BALANCE",
            Self::ChallengeWindowClosed => "CHALLENGE_WINDOW_CLOSED",
            Self::InsufficientChallengerBond => "INSUFFICIENT_CHALLENGER_BOND",
            Self::ChallengeNotFound => "CHALLENGE_NOT_FOUND",
            Self::ChallengeNotPending => "CHALLENGE_NOT_PENDING",
            Self::ChallengeExpired => "CHALLENGE_EXPIRED",
            Self::InvalidChallengeResult => "INVALID_CHALLENGE_RESULT",
            Self::InvalidChallengeQuorum => "INVALID_CHALLENGE_QUORUM",
            Self::DataExceedsMaxSize => "DATA_EXCEEDS_MAX_SIZE",
            Self::ChallengeAlreadyExists => "CHALLENGE_ALREADY_EXISTS",
            Self::DuplicateValidatorMetadata => "DUPLICATE_VALIDATOR_METADATA",
            Self::MissingProofOfPossession => "MISSING_PROOF_OF_POSSESSION",
            Self::InvalidProofOfPossession => "INVALID_PROOF_OF_POSSESSION",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for execution_error::ExecutionErrorKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "EXECUTION_ERROR_KIND_UNKNOWN",
            "INSUFFICIENT_GAS",
            "INVALID_OWNERSHIP",
            "OBJECT_NOT_FOUND",
            "INVALID_OBJECT_TYPE",
            "INVALID_TRANSACTION_TYPE",
            "INVALID_ARGUMENTS",
            "DUPLICATE_VALIDATOR",
            "NOT_A_VALIDATOR",
            "VALIDATOR_ALREADY_REMOVED",
            "ADVANCED_TO_WRONG_EPOCH",
            "MODEL_NOT_FOUND",
            "NOT_MODEL_OWNER",
            "MODEL_NOT_ACTIVE",
            "INSUFFICIENT_COIN_BALANCE",
            "COIN_BALANCE_OVERFLOW",
            "VALIDATOR_NOT_FOUND",
            "STAKING_POOL_NOT_FOUND",
            "CANNOT_REPORT_ONESELF",
            "REPORT_RECORD_NOT_FOUND",
            "INPUT_OBJECT_DELETED",
            "CERTIFICATE_DENIED",
            "SHARED_OBJECT_CONGESTION",
            "OTHER_ERROR",
            "MODEL_NOT_PENDING",
            "MODEL_ALREADY_INACTIVE",
            "MODEL_REVEAL_EPOCH_MISMATCH",
            "MODEL_WEIGHTS_URL_MISMATCH",
            "MODEL_NO_PENDING_UPDATE",
            "MODEL_ARCHITECTURE_VERSION_MISMATCH",
            "MODEL_COMMISSION_RATE_TOO_HIGH",
            "MODEL_MIN_STAKE_NOT_MET",
            "NO_ACTIVE_MODELS",
            "TARGET_NOT_FOUND",
            "TARGET_NOT_OPEN",
            "TARGET_EXPIRED",
            "TARGET_NOT_FILLED",
            "CHALLENGE_WINDOW_OPEN",
            "TARGET_ALREADY_CLAIMED",
            "MODEL_NOT_IN_TARGET",
            "EMBEDDING_DIMENSION_MISMATCH",
            "DISTANCE_EXCEEDS_THRESHOLD",
            "INSUFFICIENT_BOND",
            "INSUFFICIENT_EMISSION_BALANCE",
            "CHALLENGE_WINDOW_CLOSED",
            "INSUFFICIENT_CHALLENGER_BOND",
            "CHALLENGE_NOT_FOUND",
            "CHALLENGE_NOT_PENDING",
            "CHALLENGE_EXPIRED",
            "INVALID_CHALLENGE_RESULT",
            "INVALID_CHALLENGE_QUORUM",
            "DATA_EXCEEDS_MAX_SIZE",
            "CHALLENGE_ALREADY_EXISTS",
            "DUPLICATE_VALIDATOR_METADATA",
            "MISSING_PROOF_OF_POSSESSION",
            "INVALID_PROOF_OF_POSSESSION",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = execution_error::ExecutionErrorKind;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "EXECUTION_ERROR_KIND_UNKNOWN" => Ok(execution_error::ExecutionErrorKind::Unknown),
                    "INSUFFICIENT_GAS" => Ok(execution_error::ExecutionErrorKind::InsufficientGas),
                    "INVALID_OWNERSHIP" => Ok(execution_error::ExecutionErrorKind::InvalidOwnership),
                    "OBJECT_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::ObjectNotFound),
                    "INVALID_OBJECT_TYPE" => Ok(execution_error::ExecutionErrorKind::InvalidObjectType),
                    "INVALID_TRANSACTION_TYPE" => Ok(execution_error::ExecutionErrorKind::InvalidTransactionType),
                    "INVALID_ARGUMENTS" => Ok(execution_error::ExecutionErrorKind::InvalidArguments),
                    "DUPLICATE_VALIDATOR" => Ok(execution_error::ExecutionErrorKind::DuplicateValidator),
                    "NOT_A_VALIDATOR" => Ok(execution_error::ExecutionErrorKind::NotAValidator),
                    "VALIDATOR_ALREADY_REMOVED" => Ok(execution_error::ExecutionErrorKind::ValidatorAlreadyRemoved),
                    "ADVANCED_TO_WRONG_EPOCH" => Ok(execution_error::ExecutionErrorKind::AdvancedToWrongEpoch),
                    "MODEL_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::ModelNotFound),
                    "NOT_MODEL_OWNER" => Ok(execution_error::ExecutionErrorKind::NotModelOwner),
                    "MODEL_NOT_ACTIVE" => Ok(execution_error::ExecutionErrorKind::ModelNotActive),
                    "INSUFFICIENT_COIN_BALANCE" => Ok(execution_error::ExecutionErrorKind::InsufficientCoinBalance),
                    "COIN_BALANCE_OVERFLOW" => Ok(execution_error::ExecutionErrorKind::CoinBalanceOverflow),
                    "VALIDATOR_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::ValidatorNotFound),
                    "STAKING_POOL_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::StakingPoolNotFound),
                    "CANNOT_REPORT_ONESELF" => Ok(execution_error::ExecutionErrorKind::CannotReportOneself),
                    "REPORT_RECORD_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::ReportRecordNotFound),
                    "INPUT_OBJECT_DELETED" => Ok(execution_error::ExecutionErrorKind::InputObjectDeleted),
                    "CERTIFICATE_DENIED" => Ok(execution_error::ExecutionErrorKind::CertificateDenied),
                    "SHARED_OBJECT_CONGESTION" => Ok(execution_error::ExecutionErrorKind::SharedObjectCongestion),
                    "OTHER_ERROR" => Ok(execution_error::ExecutionErrorKind::OtherError),
                    "MODEL_NOT_PENDING" => Ok(execution_error::ExecutionErrorKind::ModelNotPending),
                    "MODEL_ALREADY_INACTIVE" => Ok(execution_error::ExecutionErrorKind::ModelAlreadyInactive),
                    "MODEL_REVEAL_EPOCH_MISMATCH" => Ok(execution_error::ExecutionErrorKind::ModelRevealEpochMismatch),
                    "MODEL_WEIGHTS_URL_MISMATCH" => Ok(execution_error::ExecutionErrorKind::ModelWeightsUrlMismatch),
                    "MODEL_NO_PENDING_UPDATE" => Ok(execution_error::ExecutionErrorKind::ModelNoPendingUpdate),
                    "MODEL_ARCHITECTURE_VERSION_MISMATCH" => Ok(execution_error::ExecutionErrorKind::ModelArchitectureVersionMismatch),
                    "MODEL_COMMISSION_RATE_TOO_HIGH" => Ok(execution_error::ExecutionErrorKind::ModelCommissionRateTooHigh),
                    "MODEL_MIN_STAKE_NOT_MET" => Ok(execution_error::ExecutionErrorKind::ModelMinStakeNotMet),
                    "NO_ACTIVE_MODELS" => Ok(execution_error::ExecutionErrorKind::NoActiveModels),
                    "TARGET_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::TargetNotFound),
                    "TARGET_NOT_OPEN" => Ok(execution_error::ExecutionErrorKind::TargetNotOpen),
                    "TARGET_EXPIRED" => Ok(execution_error::ExecutionErrorKind::TargetExpired),
                    "TARGET_NOT_FILLED" => Ok(execution_error::ExecutionErrorKind::TargetNotFilled),
                    "CHALLENGE_WINDOW_OPEN" => Ok(execution_error::ExecutionErrorKind::ChallengeWindowOpen),
                    "TARGET_ALREADY_CLAIMED" => Ok(execution_error::ExecutionErrorKind::TargetAlreadyClaimed),
                    "MODEL_NOT_IN_TARGET" => Ok(execution_error::ExecutionErrorKind::ModelNotInTarget),
                    "EMBEDDING_DIMENSION_MISMATCH" => Ok(execution_error::ExecutionErrorKind::EmbeddingDimensionMismatch),
                    "DISTANCE_EXCEEDS_THRESHOLD" => Ok(execution_error::ExecutionErrorKind::DistanceExceedsThreshold),
                    "INSUFFICIENT_BOND" => Ok(execution_error::ExecutionErrorKind::InsufficientBond),
                    "INSUFFICIENT_EMISSION_BALANCE" => Ok(execution_error::ExecutionErrorKind::InsufficientEmissionBalance),
                    "CHALLENGE_WINDOW_CLOSED" => Ok(execution_error::ExecutionErrorKind::ChallengeWindowClosed),
                    "INSUFFICIENT_CHALLENGER_BOND" => Ok(execution_error::ExecutionErrorKind::InsufficientChallengerBond),
                    "CHALLENGE_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::ChallengeNotFound),
                    "CHALLENGE_NOT_PENDING" => Ok(execution_error::ExecutionErrorKind::ChallengeNotPending),
                    "CHALLENGE_EXPIRED" => Ok(execution_error::ExecutionErrorKind::ChallengeExpired),
                    "INVALID_CHALLENGE_RESULT" => Ok(execution_error::ExecutionErrorKind::InvalidChallengeResult),
                    "INVALID_CHALLENGE_QUORUM" => Ok(execution_error::ExecutionErrorKind::InvalidChallengeQuorum),
                    "DATA_EXCEEDS_MAX_SIZE" => Ok(execution_error::ExecutionErrorKind::DataExceedsMaxSize),
                    "CHALLENGE_ALREADY_EXISTS" => Ok(execution_error::ExecutionErrorKind::ChallengeAlreadyExists),
                    "DUPLICATE_VALIDATOR_METADATA" => Ok(execution_error::ExecutionErrorKind::DuplicateValidatorMetadata),
                    "MISSING_PROOF_OF_POSSESSION" => Ok(execution_error::ExecutionErrorKind::MissingProofOfPossession),
                    "INVALID_PROOF_OF_POSSESSION" => Ok(execution_error::ExecutionErrorKind::InvalidProofOfPossession),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for ExecutionStatus {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.success.is_some() {
            len += 1;
        }
        if self.error.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ExecutionStatus", len)?;
        if let Some(v) = self.success.as_ref() {
            struct_ser.serialize_field("success", v)?;
        }
        if let Some(v) = self.error.as_ref() {
            struct_ser.serialize_field("error", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ExecutionStatus {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "success",
            "error",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Success,
            Error,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "success" => Ok(GeneratedField::Success),
                            "error" => Ok(GeneratedField::Error),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ExecutionStatus;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ExecutionStatus")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ExecutionStatus, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut success__ = None;
                let mut error__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Success => {
                            if success__.is_some() {
                                return Err(serde::de::Error::duplicate_field("success"));
                            }
                            success__ = map_.next_value()?;
                        }
                        GeneratedField::Error => {
                            if error__.is_some() {
                                return Err(serde::de::Error::duplicate_field("error"));
                            }
                            error__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ExecutionStatus {
                    success: success__,
                    error: error__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ExecutionStatus", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GenesisTransaction {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.objects.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GenesisTransaction", len)?;
        if !self.objects.is_empty() {
            struct_ser.serialize_field("objects", &self.objects)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GenesisTransaction {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "objects",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Objects,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "objects" => Ok(GeneratedField::Objects),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GenesisTransaction;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GenesisTransaction")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GenesisTransaction, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut objects__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Objects => {
                            if objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objects"));
                            }
                            objects__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GenesisTransaction {
                    objects: objects__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GenesisTransaction", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetBalanceRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.owner.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetBalanceRequest", len)?;
        if let Some(v) = self.owner.as_ref() {
            struct_ser.serialize_field("owner", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetBalanceRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "owner",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Owner,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "owner" => Ok(GeneratedField::Owner),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetBalanceRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetBalanceRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetBalanceRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut owner__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Owner => {
                            if owner__.is_some() {
                                return Err(serde::de::Error::duplicate_field("owner"));
                            }
                            owner__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetBalanceRequest {
                    owner: owner__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetBalanceRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetBalanceResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.balance.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetBalanceResponse", len)?;
        if let Some(v) = self.balance.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("balance", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetBalanceResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "balance",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Balance,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "balance" => Ok(GeneratedField::Balance),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetBalanceResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetBalanceResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetBalanceResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut balance__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Balance => {
                            if balance__.is_some() {
                                return Err(serde::de::Error::duplicate_field("balance"));
                            }
                            balance__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetBalanceResponse {
                    balance: balance__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetBalanceResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetChallengeRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.challenge_id.is_some() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetChallengeRequest", len)?;
        if let Some(v) = self.challenge_id.as_ref() {
            struct_ser.serialize_field("challengeId", v)?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetChallengeRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "challenge_id",
            "challengeId",
            "read_mask",
            "readMask",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ChallengeId,
            ReadMask,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "challengeId" | "challenge_id" => Ok(GeneratedField::ChallengeId),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetChallengeRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetChallengeRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetChallengeRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut challenge_id__ = None;
                let mut read_mask__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ChallengeId => {
                            if challenge_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challengeId"));
                            }
                            challenge_id__ = map_.next_value()?;
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetChallengeRequest {
                    challenge_id: challenge_id__,
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetChallengeRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetChallengeResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.challenge.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetChallengeResponse", len)?;
        if let Some(v) = self.challenge.as_ref() {
            struct_ser.serialize_field("challenge", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetChallengeResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "challenge",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Challenge,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "challenge" => Ok(GeneratedField::Challenge),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetChallengeResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetChallengeResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetChallengeResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut challenge__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Challenge => {
                            if challenge__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challenge"));
                            }
                            challenge__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetChallengeResponse {
                    challenge: challenge__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetChallengeResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetCheckpointRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.read_mask.is_some() {
            len += 1;
        }
        if self.checkpoint_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetCheckpointRequest", len)?;
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        if let Some(v) = self.checkpoint_id.as_ref() {
            match v {
                get_checkpoint_request::CheckpointId::SequenceNumber(v) => {
                    #[allow(clippy::needless_borrow)]
                    #[allow(clippy::needless_borrows_for_generic_args)]
                    struct_ser.serialize_field("sequenceNumber", ToString::to_string(&v).as_str())?;
                }
                get_checkpoint_request::CheckpointId::Digest(v) => {
                    struct_ser.serialize_field("digest", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetCheckpointRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "read_mask",
            "readMask",
            "sequence_number",
            "sequenceNumber",
            "digest",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ReadMask,
            SequenceNumber,
            Digest,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            "sequenceNumber" | "sequence_number" => Ok(GeneratedField::SequenceNumber),
                            "digest" => Ok(GeneratedField::Digest),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetCheckpointRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetCheckpointRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetCheckpointRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut read_mask__ = None;
                let mut checkpoint_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::SequenceNumber => {
                            if checkpoint_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("sequenceNumber"));
                            }
                            checkpoint_id__ = map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| get_checkpoint_request::CheckpointId::SequenceNumber(x.0));
                        }
                        GeneratedField::Digest => {
                            if checkpoint_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            checkpoint_id__ = map_.next_value::<::std::option::Option<_>>()?.map(get_checkpoint_request::CheckpointId::Digest);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetCheckpointRequest {
                    read_mask: read_mask__,
                    checkpoint_id: checkpoint_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetCheckpointRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetCheckpointResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.checkpoint.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetCheckpointResponse", len)?;
        if let Some(v) = self.checkpoint.as_ref() {
            struct_ser.serialize_field("checkpoint", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetCheckpointResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "checkpoint",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Checkpoint,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "checkpoint" => Ok(GeneratedField::Checkpoint),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetCheckpointResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetCheckpointResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetCheckpointResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut checkpoint__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Checkpoint => {
                            if checkpoint__.is_some() {
                                return Err(serde::de::Error::duplicate_field("checkpoint"));
                            }
                            checkpoint__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetCheckpointResponse {
                    checkpoint: checkpoint__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetCheckpointResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetEpochRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.epoch.is_some() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetEpochRequest", len)?;
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetEpochRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epoch",
            "read_mask",
            "readMask",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            ReadMask,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "epoch" => Ok(GeneratedField::Epoch),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetEpochRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetEpochRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetEpochRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epoch__ = None;
                let mut read_mask__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetEpochRequest {
                    epoch: epoch__,
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetEpochRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetEpochResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.epoch.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetEpochResponse", len)?;
        if let Some(v) = self.epoch.as_ref() {
            struct_ser.serialize_field("epoch", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetEpochResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epoch",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "epoch" => Ok(GeneratedField::Epoch),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetEpochResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetEpochResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetEpochResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epoch__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetEpochResponse {
                    epoch: epoch__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetEpochResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetObjectRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.object_id.is_some() {
            len += 1;
        }
        if self.version.is_some() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetObjectRequest", len)?;
        if let Some(v) = self.object_id.as_ref() {
            struct_ser.serialize_field("objectId", v)?;
        }
        if let Some(v) = self.version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("version", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetObjectRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "object_id",
            "objectId",
            "version",
            "read_mask",
            "readMask",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ObjectId,
            Version,
            ReadMask,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "objectId" | "object_id" => Ok(GeneratedField::ObjectId),
                            "version" => Ok(GeneratedField::Version),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetObjectRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetObjectRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetObjectRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut object_id__ = None;
                let mut version__ = None;
                let mut read_mask__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ObjectId => {
                            if object_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectId"));
                            }
                            object_id__ = map_.next_value()?;
                        }
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetObjectRequest {
                    object_id: object_id__,
                    version: version__,
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetObjectRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetObjectResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.object.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetObjectResponse", len)?;
        if let Some(v) = self.object.as_ref() {
            struct_ser.serialize_field("object", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetObjectResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "object",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Object,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "object" => Ok(GeneratedField::Object),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetObjectResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetObjectResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetObjectResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut object__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Object => {
                            if object__.is_some() {
                                return Err(serde::de::Error::duplicate_field("object"));
                            }
                            object__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetObjectResponse {
                    object: object__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetObjectResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetObjectResult {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.result.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetObjectResult", len)?;
        if let Some(v) = self.result.as_ref() {
            match v {
                get_object_result::Result::Object(v) => {
                    struct_ser.serialize_field("object", v)?;
                }
                get_object_result::Result::Error(v) => {
                    struct_ser.serialize_field("error", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetObjectResult {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "object",
            "error",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Object,
            Error,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "object" => Ok(GeneratedField::Object),
                            "error" => Ok(GeneratedField::Error),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetObjectResult;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetObjectResult")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetObjectResult, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut result__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Object => {
                            if result__.is_some() {
                                return Err(serde::de::Error::duplicate_field("object"));
                            }
                            result__ = map_.next_value::<::std::option::Option<_>>()?.map(get_object_result::Result::Object)
;
                        }
                        GeneratedField::Error => {
                            if result__.is_some() {
                                return Err(serde::de::Error::duplicate_field("error"));
                            }
                            result__ = map_.next_value::<::std::option::Option<_>>()?.map(get_object_result::Result::Error)
;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetObjectResult {
                    result: result__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetObjectResult", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetServiceInfoRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let len = 0;
        let struct_ser = serializer.serialize_struct("soma.rpc.GetServiceInfoRequest", len)?;
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetServiceInfoRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                            Ok(GeneratedField::__SkipField__)
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetServiceInfoRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetServiceInfoRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetServiceInfoRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                while map_.next_key::<GeneratedField>()?.is_some() {
                    let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                }
                Ok(GetServiceInfoRequest {
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetServiceInfoRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetServiceInfoResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.chain_id.is_some() {
            len += 1;
        }
        if self.chain.is_some() {
            len += 1;
        }
        if self.epoch.is_some() {
            len += 1;
        }
        if self.checkpoint_height.is_some() {
            len += 1;
        }
        if self.timestamp.is_some() {
            len += 1;
        }
        if self.lowest_available_checkpoint.is_some() {
            len += 1;
        }
        if self.lowest_available_checkpoint_objects.is_some() {
            len += 1;
        }
        if self.server.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetServiceInfoResponse", len)?;
        if let Some(v) = self.chain_id.as_ref() {
            struct_ser.serialize_field("chainId", v)?;
        }
        if let Some(v) = self.chain.as_ref() {
            struct_ser.serialize_field("chain", v)?;
        }
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.checkpoint_height.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("checkpointHeight", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.timestamp.as_ref() {
            struct_ser.serialize_field("timestamp", &crate::utils::_serde::TimestampSerializer(v))?;
        }
        if let Some(v) = self.lowest_available_checkpoint.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("lowestAvailableCheckpoint", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.lowest_available_checkpoint_objects.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("lowestAvailableCheckpointObjects", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.server.as_ref() {
            struct_ser.serialize_field("server", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetServiceInfoResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "chain_id",
            "chainId",
            "chain",
            "epoch",
            "checkpoint_height",
            "checkpointHeight",
            "timestamp",
            "lowest_available_checkpoint",
            "lowestAvailableCheckpoint",
            "lowest_available_checkpoint_objects",
            "lowestAvailableCheckpointObjects",
            "server",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ChainId,
            Chain,
            Epoch,
            CheckpointHeight,
            Timestamp,
            LowestAvailableCheckpoint,
            LowestAvailableCheckpointObjects,
            Server,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "chainId" | "chain_id" => Ok(GeneratedField::ChainId),
                            "chain" => Ok(GeneratedField::Chain),
                            "epoch" => Ok(GeneratedField::Epoch),
                            "checkpointHeight" | "checkpoint_height" => Ok(GeneratedField::CheckpointHeight),
                            "timestamp" => Ok(GeneratedField::Timestamp),
                            "lowestAvailableCheckpoint" | "lowest_available_checkpoint" => Ok(GeneratedField::LowestAvailableCheckpoint),
                            "lowestAvailableCheckpointObjects" | "lowest_available_checkpoint_objects" => Ok(GeneratedField::LowestAvailableCheckpointObjects),
                            "server" => Ok(GeneratedField::Server),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetServiceInfoResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetServiceInfoResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetServiceInfoResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut chain_id__ = None;
                let mut chain__ = None;
                let mut epoch__ = None;
                let mut checkpoint_height__ = None;
                let mut timestamp__ = None;
                let mut lowest_available_checkpoint__ = None;
                let mut lowest_available_checkpoint_objects__ = None;
                let mut server__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ChainId => {
                            if chain_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("chainId"));
                            }
                            chain_id__ = map_.next_value()?;
                        }
                        GeneratedField::Chain => {
                            if chain__.is_some() {
                                return Err(serde::de::Error::duplicate_field("chain"));
                            }
                            chain__ = map_.next_value()?;
                        }
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::CheckpointHeight => {
                            if checkpoint_height__.is_some() {
                                return Err(serde::de::Error::duplicate_field("checkpointHeight"));
                            }
                            checkpoint_height__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Timestamp => {
                            if timestamp__.is_some() {
                                return Err(serde::de::Error::duplicate_field("timestamp"));
                            }
                            timestamp__ = map_.next_value::<::std::option::Option<crate::utils::_serde::TimestampDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::LowestAvailableCheckpoint => {
                            if lowest_available_checkpoint__.is_some() {
                                return Err(serde::de::Error::duplicate_field("lowestAvailableCheckpoint"));
                            }
                            lowest_available_checkpoint__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::LowestAvailableCheckpointObjects => {
                            if lowest_available_checkpoint_objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("lowestAvailableCheckpointObjects"));
                            }
                            lowest_available_checkpoint_objects__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Server => {
                            if server__.is_some() {
                                return Err(serde::de::Error::duplicate_field("server"));
                            }
                            server__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetServiceInfoResponse {
                    chain_id: chain_id__,
                    chain: chain__,
                    epoch: epoch__,
                    checkpoint_height: checkpoint_height__,
                    timestamp: timestamp__,
                    lowest_available_checkpoint: lowest_available_checkpoint__,
                    lowest_available_checkpoint_objects: lowest_available_checkpoint_objects__,
                    server: server__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetServiceInfoResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetTargetRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.target_id.is_some() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetTargetRequest", len)?;
        if let Some(v) = self.target_id.as_ref() {
            struct_ser.serialize_field("targetId", v)?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetTargetRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "target_id",
            "targetId",
            "read_mask",
            "readMask",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TargetId,
            ReadMask,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "targetId" | "target_id" => Ok(GeneratedField::TargetId),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetTargetRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetTargetRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetTargetRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut target_id__ = None;
                let mut read_mask__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::TargetId => {
                            if target_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetId"));
                            }
                            target_id__ = map_.next_value()?;
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetTargetRequest {
                    target_id: target_id__,
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetTargetRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetTargetResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.target.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetTargetResponse", len)?;
        if let Some(v) = self.target.as_ref() {
            struct_ser.serialize_field("target", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetTargetResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "target",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Target,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "target" => Ok(GeneratedField::Target),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetTargetResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetTargetResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetTargetResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut target__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Target => {
                            if target__.is_some() {
                                return Err(serde::de::Error::duplicate_field("target"));
                            }
                            target__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetTargetResponse {
                    target: target__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetTargetResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetTransactionRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.digest.is_some() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetTransactionRequest", len)?;
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetTransactionRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "digest",
            "read_mask",
            "readMask",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Digest,
            ReadMask,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "digest" => Ok(GeneratedField::Digest),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetTransactionRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetTransactionRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetTransactionRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut digest__ = None;
                let mut read_mask__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetTransactionRequest {
                    digest: digest__,
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetTransactionRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetTransactionResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.transaction.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetTransactionResponse", len)?;
        if let Some(v) = self.transaction.as_ref() {
            struct_ser.serialize_field("transaction", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetTransactionResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "transaction",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Transaction,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "transaction" => Ok(GeneratedField::Transaction),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetTransactionResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetTransactionResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetTransactionResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut transaction__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Transaction => {
                            if transaction__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transaction"));
                            }
                            transaction__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetTransactionResponse {
                    transaction: transaction__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetTransactionResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetTransactionResult {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.result.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetTransactionResult", len)?;
        if let Some(v) = self.result.as_ref() {
            match v {
                get_transaction_result::Result::Transaction(v) => {
                    struct_ser.serialize_field("transaction", v)?;
                }
                get_transaction_result::Result::Error(v) => {
                    struct_ser.serialize_field("error", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetTransactionResult {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "transaction",
            "error",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Transaction,
            Error,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "transaction" => Ok(GeneratedField::Transaction),
                            "error" => Ok(GeneratedField::Error),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = GetTransactionResult;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetTransactionResult")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetTransactionResult, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut result__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Transaction => {
                            if result__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transaction"));
                            }
                            result__ = map_.next_value::<::std::option::Option<_>>()?.map(get_transaction_result::Result::Transaction)
;
                        }
                        GeneratedField::Error => {
                            if result__.is_some() {
                                return Err(serde::de::Error::duplicate_field("error"));
                            }
                            result__ = map_.next_value::<::std::option::Option<_>>()?.map(get_transaction_result::Result::Error)
;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetTransactionResult {
                    result: result__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetTransactionResult", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for InitiateChallenge {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.target_id.is_some() {
            len += 1;
        }
        if self.challenge_type.is_some() {
            len += 1;
        }
        if self.model_id.is_some() {
            len += 1;
        }
        if self.bond_coin.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.InitiateChallenge", len)?;
        if let Some(v) = self.target_id.as_ref() {
            struct_ser.serialize_field("targetId", v)?;
        }
        if let Some(v) = self.challenge_type.as_ref() {
            struct_ser.serialize_field("challengeType", v)?;
        }
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        if let Some(v) = self.bond_coin.as_ref() {
            struct_ser.serialize_field("bondCoin", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for InitiateChallenge {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "target_id",
            "targetId",
            "challenge_type",
            "challengeType",
            "model_id",
            "modelId",
            "bond_coin",
            "bondCoin",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TargetId,
            ChallengeType,
            ModelId,
            BondCoin,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "targetId" | "target_id" => Ok(GeneratedField::TargetId),
                            "challengeType" | "challenge_type" => Ok(GeneratedField::ChallengeType),
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            "bondCoin" | "bond_coin" => Ok(GeneratedField::BondCoin),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = InitiateChallenge;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.InitiateChallenge")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<InitiateChallenge, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut target_id__ = None;
                let mut challenge_type__ = None;
                let mut model_id__ = None;
                let mut bond_coin__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::TargetId => {
                            if target_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetId"));
                            }
                            target_id__ = map_.next_value()?;
                        }
                        GeneratedField::ChallengeType => {
                            if challenge_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challengeType"));
                            }
                            challenge_type__ = map_.next_value()?;
                        }
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::BondCoin => {
                            if bond_coin__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bondCoin"));
                            }
                            bond_coin__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(InitiateChallenge {
                    target_id: target_id__,
                    challenge_type: challenge_type__,
                    model_id: model_id__,
                    bond_coin: bond_coin__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.InitiateChallenge", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ListChallengesRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.target_id.is_some() {
            len += 1;
        }
        if self.status_filter.is_some() {
            len += 1;
        }
        if self.epoch_filter.is_some() {
            len += 1;
        }
        if self.page_size.is_some() {
            len += 1;
        }
        if self.page_token.is_some() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ListChallengesRequest", len)?;
        if let Some(v) = self.target_id.as_ref() {
            struct_ser.serialize_field("targetId", v)?;
        }
        if let Some(v) = self.status_filter.as_ref() {
            struct_ser.serialize_field("statusFilter", v)?;
        }
        if let Some(v) = self.epoch_filter.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epochFilter", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.page_size.as_ref() {
            struct_ser.serialize_field("pageSize", v)?;
        }
        if let Some(v) = self.page_token.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("pageToken", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ListChallengesRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "target_id",
            "targetId",
            "status_filter",
            "statusFilter",
            "epoch_filter",
            "epochFilter",
            "page_size",
            "pageSize",
            "page_token",
            "pageToken",
            "read_mask",
            "readMask",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TargetId,
            StatusFilter,
            EpochFilter,
            PageSize,
            PageToken,
            ReadMask,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "targetId" | "target_id" => Ok(GeneratedField::TargetId),
                            "statusFilter" | "status_filter" => Ok(GeneratedField::StatusFilter),
                            "epochFilter" | "epoch_filter" => Ok(GeneratedField::EpochFilter),
                            "pageSize" | "page_size" => Ok(GeneratedField::PageSize),
                            "pageToken" | "page_token" => Ok(GeneratedField::PageToken),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ListChallengesRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ListChallengesRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ListChallengesRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut target_id__ = None;
                let mut status_filter__ = None;
                let mut epoch_filter__ = None;
                let mut page_size__ = None;
                let mut page_token__ = None;
                let mut read_mask__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::TargetId => {
                            if target_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetId"));
                            }
                            target_id__ = map_.next_value()?;
                        }
                        GeneratedField::StatusFilter => {
                            if status_filter__.is_some() {
                                return Err(serde::de::Error::duplicate_field("statusFilter"));
                            }
                            status_filter__ = map_.next_value()?;
                        }
                        GeneratedField::EpochFilter => {
                            if epoch_filter__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epochFilter"));
                            }
                            epoch_filter__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PageSize => {
                            if page_size__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pageSize"));
                            }
                            page_size__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PageToken => {
                            if page_token__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pageToken"));
                            }
                            page_token__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ListChallengesRequest {
                    target_id: target_id__,
                    status_filter: status_filter__,
                    epoch_filter: epoch_filter__,
                    page_size: page_size__,
                    page_token: page_token__,
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ListChallengesRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ListChallengesResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.challenges.is_empty() {
            len += 1;
        }
        if self.next_page_token.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ListChallengesResponse", len)?;
        if !self.challenges.is_empty() {
            struct_ser.serialize_field("challenges", &self.challenges)?;
        }
        if let Some(v) = self.next_page_token.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextPageToken", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ListChallengesResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "challenges",
            "next_page_token",
            "nextPageToken",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Challenges,
            NextPageToken,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "challenges" => Ok(GeneratedField::Challenges),
                            "nextPageToken" | "next_page_token" => Ok(GeneratedField::NextPageToken),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ListChallengesResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ListChallengesResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ListChallengesResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut challenges__ = None;
                let mut next_page_token__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Challenges => {
                            if challenges__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challenges"));
                            }
                            challenges__ = Some(map_.next_value()?);
                        }
                        GeneratedField::NextPageToken => {
                            if next_page_token__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextPageToken"));
                            }
                            next_page_token__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ListChallengesResponse {
                    challenges: challenges__.unwrap_or_default(),
                    next_page_token: next_page_token__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ListChallengesResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ListOwnedObjectsRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.owner.is_some() {
            len += 1;
        }
        if self.page_size.is_some() {
            len += 1;
        }
        if self.page_token.is_some() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        if self.object_type.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ListOwnedObjectsRequest", len)?;
        if let Some(v) = self.owner.as_ref() {
            struct_ser.serialize_field("owner", v)?;
        }
        if let Some(v) = self.page_size.as_ref() {
            struct_ser.serialize_field("pageSize", v)?;
        }
        if let Some(v) = self.page_token.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("pageToken", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        if let Some(v) = self.object_type.as_ref() {
            struct_ser.serialize_field("objectType", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ListOwnedObjectsRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "owner",
            "page_size",
            "pageSize",
            "page_token",
            "pageToken",
            "read_mask",
            "readMask",
            "object_type",
            "objectType",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Owner,
            PageSize,
            PageToken,
            ReadMask,
            ObjectType,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "owner" => Ok(GeneratedField::Owner),
                            "pageSize" | "page_size" => Ok(GeneratedField::PageSize),
                            "pageToken" | "page_token" => Ok(GeneratedField::PageToken),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            "objectType" | "object_type" => Ok(GeneratedField::ObjectType),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ListOwnedObjectsRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ListOwnedObjectsRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ListOwnedObjectsRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut owner__ = None;
                let mut page_size__ = None;
                let mut page_token__ = None;
                let mut read_mask__ = None;
                let mut object_type__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Owner => {
                            if owner__.is_some() {
                                return Err(serde::de::Error::duplicate_field("owner"));
                            }
                            owner__ = map_.next_value()?;
                        }
                        GeneratedField::PageSize => {
                            if page_size__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pageSize"));
                            }
                            page_size__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PageToken => {
                            if page_token__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pageToken"));
                            }
                            page_token__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::ObjectType => {
                            if object_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectType"));
                            }
                            object_type__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ListOwnedObjectsRequest {
                    owner: owner__,
                    page_size: page_size__,
                    page_token: page_token__,
                    read_mask: read_mask__,
                    object_type: object_type__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ListOwnedObjectsRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ListOwnedObjectsResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.objects.is_empty() {
            len += 1;
        }
        if self.next_page_token.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ListOwnedObjectsResponse", len)?;
        if !self.objects.is_empty() {
            struct_ser.serialize_field("objects", &self.objects)?;
        }
        if let Some(v) = self.next_page_token.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextPageToken", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ListOwnedObjectsResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "objects",
            "next_page_token",
            "nextPageToken",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Objects,
            NextPageToken,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "objects" => Ok(GeneratedField::Objects),
                            "nextPageToken" | "next_page_token" => Ok(GeneratedField::NextPageToken),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ListOwnedObjectsResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ListOwnedObjectsResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ListOwnedObjectsResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut objects__ = None;
                let mut next_page_token__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Objects => {
                            if objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objects"));
                            }
                            objects__ = Some(map_.next_value()?);
                        }
                        GeneratedField::NextPageToken => {
                            if next_page_token__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextPageToken"));
                            }
                            next_page_token__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ListOwnedObjectsResponse {
                    objects: objects__.unwrap_or_default(),
                    next_page_token: next_page_token__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ListOwnedObjectsResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ListTargetsRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.status_filter.is_some() {
            len += 1;
        }
        if self.epoch_filter.is_some() {
            len += 1;
        }
        if self.page_size.is_some() {
            len += 1;
        }
        if self.page_token.is_some() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ListTargetsRequest", len)?;
        if let Some(v) = self.status_filter.as_ref() {
            struct_ser.serialize_field("statusFilter", v)?;
        }
        if let Some(v) = self.epoch_filter.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epochFilter", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.page_size.as_ref() {
            struct_ser.serialize_field("pageSize", v)?;
        }
        if let Some(v) = self.page_token.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("pageToken", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ListTargetsRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "status_filter",
            "statusFilter",
            "epoch_filter",
            "epochFilter",
            "page_size",
            "pageSize",
            "page_token",
            "pageToken",
            "read_mask",
            "readMask",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            StatusFilter,
            EpochFilter,
            PageSize,
            PageToken,
            ReadMask,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "statusFilter" | "status_filter" => Ok(GeneratedField::StatusFilter),
                            "epochFilter" | "epoch_filter" => Ok(GeneratedField::EpochFilter),
                            "pageSize" | "page_size" => Ok(GeneratedField::PageSize),
                            "pageToken" | "page_token" => Ok(GeneratedField::PageToken),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ListTargetsRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ListTargetsRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ListTargetsRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut status_filter__ = None;
                let mut epoch_filter__ = None;
                let mut page_size__ = None;
                let mut page_token__ = None;
                let mut read_mask__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::StatusFilter => {
                            if status_filter__.is_some() {
                                return Err(serde::de::Error::duplicate_field("statusFilter"));
                            }
                            status_filter__ = map_.next_value()?;
                        }
                        GeneratedField::EpochFilter => {
                            if epoch_filter__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epochFilter"));
                            }
                            epoch_filter__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PageSize => {
                            if page_size__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pageSize"));
                            }
                            page_size__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PageToken => {
                            if page_token__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pageToken"));
                            }
                            page_token__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ListTargetsRequest {
                    status_filter: status_filter__,
                    epoch_filter: epoch_filter__,
                    page_size: page_size__,
                    page_token: page_token__,
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ListTargetsRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ListTargetsResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.targets.is_empty() {
            len += 1;
        }
        if self.next_page_token.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ListTargetsResponse", len)?;
        if !self.targets.is_empty() {
            struct_ser.serialize_field("targets", &self.targets)?;
        }
        if let Some(v) = self.next_page_token.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextPageToken", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ListTargetsResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "targets",
            "next_page_token",
            "nextPageToken",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Targets,
            NextPageToken,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "targets" => Ok(GeneratedField::Targets),
                            "nextPageToken" | "next_page_token" => Ok(GeneratedField::NextPageToken),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ListTargetsResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ListTargetsResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ListTargetsResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut targets__ = None;
                let mut next_page_token__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Targets => {
                            if targets__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targets"));
                            }
                            targets__ = Some(map_.next_value()?);
                        }
                        GeneratedField::NextPageToken => {
                            if next_page_token__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextPageToken"));
                            }
                            next_page_token__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ListTargetsResponse {
                    targets: targets__.unwrap_or_default(),
                    next_page_token: next_page_token__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ListTargetsResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Manifest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.version.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Manifest", len)?;
        if let Some(v) = self.version.as_ref() {
            match v {
                manifest::Version::V1(v) => {
                    struct_ser.serialize_field("v1", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Manifest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "v1",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            V1,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "v1" => Ok(GeneratedField::V1),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Manifest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Manifest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Manifest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut version__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::V1 => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("v1"));
                            }
                            version__ = map_.next_value::<::std::option::Option<_>>()?.map(manifest::Version::V1)
;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Manifest {
                    version: version__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Manifest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ManifestV1 {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.url.is_some() {
            len += 1;
        }
        if self.metadata.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ManifestV1", len)?;
        if let Some(v) = self.url.as_ref() {
            struct_ser.serialize_field("url", v)?;
        }
        if let Some(v) = self.metadata.as_ref() {
            struct_ser.serialize_field("metadata", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ManifestV1 {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "url",
            "metadata",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Url,
            Metadata,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "url" => Ok(GeneratedField::Url),
                            "metadata" => Ok(GeneratedField::Metadata),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ManifestV1;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ManifestV1")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ManifestV1, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut url__ = None;
                let mut metadata__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Url => {
                            if url__.is_some() {
                                return Err(serde::de::Error::duplicate_field("url"));
                            }
                            url__ = map_.next_value()?;
                        }
                        GeneratedField::Metadata => {
                            if metadata__.is_some() {
                                return Err(serde::de::Error::duplicate_field("metadata"));
                            }
                            metadata__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ManifestV1 {
                    url: url__,
                    metadata: metadata__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ManifestV1", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Metadata {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.version.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Metadata", len)?;
        if let Some(v) = self.version.as_ref() {
            match v {
                metadata::Version::V1(v) => {
                    struct_ser.serialize_field("v1", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Metadata {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "v1",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            V1,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "v1" => Ok(GeneratedField::V1),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Metadata;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Metadata")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Metadata, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut version__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::V1 => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("v1"));
                            }
                            version__ = map_.next_value::<::std::option::Option<_>>()?.map(metadata::Version::V1)
;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Metadata {
                    version: version__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Metadata", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for MetadataV1 {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.checksum.is_some() {
            len += 1;
        }
        if self.size.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.MetadataV1", len)?;
        if let Some(v) = self.checksum.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("checksum", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.size.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("size", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MetadataV1 {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "checksum",
            "size",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Checksum,
            Size,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "checksum" => Ok(GeneratedField::Checksum),
                            "size" => Ok(GeneratedField::Size),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = MetadataV1;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.MetadataV1")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MetadataV1, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut checksum__ = None;
                let mut size__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Checksum => {
                            if checksum__.is_some() {
                                return Err(serde::de::Error::duplicate_field("checksum"));
                            }
                            checksum__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Size => {
                            if size__.is_some() {
                                return Err(serde::de::Error::duplicate_field("size"));
                            }
                            size__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MetadataV1 {
                    checksum: checksum__,
                    size: size__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.MetadataV1", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Model {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.owner.is_some() {
            len += 1;
        }
        if self.architecture_version.is_some() {
            len += 1;
        }
        if self.weights_url_commitment.is_some() {
            len += 1;
        }
        if self.weights_commitment.is_some() {
            len += 1;
        }
        if self.commit_epoch.is_some() {
            len += 1;
        }
        if self.weights_manifest.is_some() {
            len += 1;
        }
        if !self.embedding.is_empty() {
            len += 1;
        }
        if self.staking_pool.is_some() {
            len += 1;
        }
        if self.commission_rate.is_some() {
            len += 1;
        }
        if self.next_epoch_commission_rate.is_some() {
            len += 1;
        }
        if self.pending_update.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Model", len)?;
        if let Some(v) = self.owner.as_ref() {
            struct_ser.serialize_field("owner", v)?;
        }
        if let Some(v) = self.architecture_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("architectureVersion", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.weights_url_commitment.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("weightsUrlCommitment", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.weights_commitment.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("weightsCommitment", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.commit_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("commitEpoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.weights_manifest.as_ref() {
            struct_ser.serialize_field("weightsManifest", v)?;
        }
        if !self.embedding.is_empty() {
            struct_ser.serialize_field("embedding", &self.embedding)?;
        }
        if let Some(v) = self.staking_pool.as_ref() {
            struct_ser.serialize_field("stakingPool", v)?;
        }
        if let Some(v) = self.commission_rate.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("commissionRate", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_commission_rate.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochCommissionRate", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.pending_update.as_ref() {
            struct_ser.serialize_field("pendingUpdate", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Model {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "owner",
            "architecture_version",
            "architectureVersion",
            "weights_url_commitment",
            "weightsUrlCommitment",
            "weights_commitment",
            "weightsCommitment",
            "commit_epoch",
            "commitEpoch",
            "weights_manifest",
            "weightsManifest",
            "embedding",
            "staking_pool",
            "stakingPool",
            "commission_rate",
            "commissionRate",
            "next_epoch_commission_rate",
            "nextEpochCommissionRate",
            "pending_update",
            "pendingUpdate",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Owner,
            ArchitectureVersion,
            WeightsUrlCommitment,
            WeightsCommitment,
            CommitEpoch,
            WeightsManifest,
            Embedding,
            StakingPool,
            CommissionRate,
            NextEpochCommissionRate,
            PendingUpdate,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "owner" => Ok(GeneratedField::Owner),
                            "architectureVersion" | "architecture_version" => Ok(GeneratedField::ArchitectureVersion),
                            "weightsUrlCommitment" | "weights_url_commitment" => Ok(GeneratedField::WeightsUrlCommitment),
                            "weightsCommitment" | "weights_commitment" => Ok(GeneratedField::WeightsCommitment),
                            "commitEpoch" | "commit_epoch" => Ok(GeneratedField::CommitEpoch),
                            "weightsManifest" | "weights_manifest" => Ok(GeneratedField::WeightsManifest),
                            "embedding" => Ok(GeneratedField::Embedding),
                            "stakingPool" | "staking_pool" => Ok(GeneratedField::StakingPool),
                            "commissionRate" | "commission_rate" => Ok(GeneratedField::CommissionRate),
                            "nextEpochCommissionRate" | "next_epoch_commission_rate" => Ok(GeneratedField::NextEpochCommissionRate),
                            "pendingUpdate" | "pending_update" => Ok(GeneratedField::PendingUpdate),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Model;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Model")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Model, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut owner__ = None;
                let mut architecture_version__ = None;
                let mut weights_url_commitment__ = None;
                let mut weights_commitment__ = None;
                let mut commit_epoch__ = None;
                let mut weights_manifest__ = None;
                let mut embedding__ = None;
                let mut staking_pool__ = None;
                let mut commission_rate__ = None;
                let mut next_epoch_commission_rate__ = None;
                let mut pending_update__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Owner => {
                            if owner__.is_some() {
                                return Err(serde::de::Error::duplicate_field("owner"));
                            }
                            owner__ = map_.next_value()?;
                        }
                        GeneratedField::ArchitectureVersion => {
                            if architecture_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("architectureVersion"));
                            }
                            architecture_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::WeightsUrlCommitment => {
                            if weights_url_commitment__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weightsUrlCommitment"));
                            }
                            weights_url_commitment__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::WeightsCommitment => {
                            if weights_commitment__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weightsCommitment"));
                            }
                            weights_commitment__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::CommitEpoch => {
                            if commit_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commitEpoch"));
                            }
                            commit_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::WeightsManifest => {
                            if weights_manifest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weightsManifest"));
                            }
                            weights_manifest__ = map_.next_value()?;
                        }
                        GeneratedField::Embedding => {
                            if embedding__.is_some() {
                                return Err(serde::de::Error::duplicate_field("embedding"));
                            }
                            embedding__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::StakingPool => {
                            if staking_pool__.is_some() {
                                return Err(serde::de::Error::duplicate_field("stakingPool"));
                            }
                            staking_pool__ = map_.next_value()?;
                        }
                        GeneratedField::CommissionRate => {
                            if commission_rate__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commissionRate"));
                            }
                            commission_rate__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochCommissionRate => {
                            if next_epoch_commission_rate__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochCommissionRate"));
                            }
                            next_epoch_commission_rate__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PendingUpdate => {
                            if pending_update__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pendingUpdate"));
                            }
                            pending_update__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Model {
                    owner: owner__,
                    architecture_version: architecture_version__,
                    weights_url_commitment: weights_url_commitment__,
                    weights_commitment: weights_commitment__,
                    commit_epoch: commit_epoch__,
                    weights_manifest: weights_manifest__,
                    embedding: embedding__.unwrap_or_default(),
                    staking_pool: staking_pool__,
                    commission_rate: commission_rate__,
                    next_epoch_commission_rate: next_epoch_commission_rate__,
                    pending_update: pending_update__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Model", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ModelRegistry {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.active_models.is_empty() {
            len += 1;
        }
        if !self.pending_models.is_empty() {
            len += 1;
        }
        if !self.staking_pool_mappings.is_empty() {
            len += 1;
        }
        if !self.inactive_models.is_empty() {
            len += 1;
        }
        if self.total_model_stake.is_some() {
            len += 1;
        }
        if !self.model_report_records.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ModelRegistry", len)?;
        if !self.active_models.is_empty() {
            struct_ser.serialize_field("activeModels", &self.active_models)?;
        }
        if !self.pending_models.is_empty() {
            struct_ser.serialize_field("pendingModels", &self.pending_models)?;
        }
        if !self.staking_pool_mappings.is_empty() {
            struct_ser.serialize_field("stakingPoolMappings", &self.staking_pool_mappings)?;
        }
        if !self.inactive_models.is_empty() {
            struct_ser.serialize_field("inactiveModels", &self.inactive_models)?;
        }
        if let Some(v) = self.total_model_stake.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("totalModelStake", ToString::to_string(&v).as_str())?;
        }
        if !self.model_report_records.is_empty() {
            struct_ser.serialize_field("modelReportRecords", &self.model_report_records)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ModelRegistry {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "active_models",
            "activeModels",
            "pending_models",
            "pendingModels",
            "staking_pool_mappings",
            "stakingPoolMappings",
            "inactive_models",
            "inactiveModels",
            "total_model_stake",
            "totalModelStake",
            "model_report_records",
            "modelReportRecords",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ActiveModels,
            PendingModels,
            StakingPoolMappings,
            InactiveModels,
            TotalModelStake,
            ModelReportRecords,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "activeModels" | "active_models" => Ok(GeneratedField::ActiveModels),
                            "pendingModels" | "pending_models" => Ok(GeneratedField::PendingModels),
                            "stakingPoolMappings" | "staking_pool_mappings" => Ok(GeneratedField::StakingPoolMappings),
                            "inactiveModels" | "inactive_models" => Ok(GeneratedField::InactiveModels),
                            "totalModelStake" | "total_model_stake" => Ok(GeneratedField::TotalModelStake),
                            "modelReportRecords" | "model_report_records" => Ok(GeneratedField::ModelReportRecords),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ModelRegistry;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ModelRegistry")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ModelRegistry, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut active_models__ = None;
                let mut pending_models__ = None;
                let mut staking_pool_mappings__ = None;
                let mut inactive_models__ = None;
                let mut total_model_stake__ = None;
                let mut model_report_records__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ActiveModels => {
                            if active_models__.is_some() {
                                return Err(serde::de::Error::duplicate_field("activeModels"));
                            }
                            active_models__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::PendingModels => {
                            if pending_models__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pendingModels"));
                            }
                            pending_models__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::StakingPoolMappings => {
                            if staking_pool_mappings__.is_some() {
                                return Err(serde::de::Error::duplicate_field("stakingPoolMappings"));
                            }
                            staking_pool_mappings__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::InactiveModels => {
                            if inactive_models__.is_some() {
                                return Err(serde::de::Error::duplicate_field("inactiveModels"));
                            }
                            inactive_models__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::TotalModelStake => {
                            if total_model_stake__.is_some() {
                                return Err(serde::de::Error::duplicate_field("totalModelStake"));
                            }
                            total_model_stake__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ModelReportRecords => {
                            if model_report_records__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelReportRecords"));
                            }
                            model_report_records__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ModelRegistry {
                    active_models: active_models__.unwrap_or_default(),
                    pending_models: pending_models__.unwrap_or_default(),
                    staking_pool_mappings: staking_pool_mappings__.unwrap_or_default(),
                    inactive_models: inactive_models__.unwrap_or_default(),
                    total_model_stake: total_model_stake__,
                    model_report_records: model_report_records__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ModelRegistry", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ModelWeightsManifest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.manifest.is_some() {
            len += 1;
        }
        if self.decryption_key.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ModelWeightsManifest", len)?;
        if let Some(v) = self.manifest.as_ref() {
            struct_ser.serialize_field("manifest", v)?;
        }
        if let Some(v) = self.decryption_key.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("decryptionKey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ModelWeightsManifest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "manifest",
            "decryption_key",
            "decryptionKey",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Manifest,
            DecryptionKey,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "manifest" => Ok(GeneratedField::Manifest),
                            "decryptionKey" | "decryption_key" => Ok(GeneratedField::DecryptionKey),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ModelWeightsManifest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ModelWeightsManifest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ModelWeightsManifest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut manifest__ = None;
                let mut decryption_key__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Manifest => {
                            if manifest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("manifest"));
                            }
                            manifest__ = map_.next_value()?;
                        }
                        GeneratedField::DecryptionKey => {
                            if decryption_key__.is_some() {
                                return Err(serde::de::Error::duplicate_field("decryptionKey"));
                            }
                            decryption_key__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ModelWeightsManifest {
                    manifest: manifest__,
                    decryption_key: decryption_key__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ModelWeightsManifest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for MultisigAggregatedSignature {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.signatures.is_empty() {
            len += 1;
        }
        if self.bitmap.is_some() {
            len += 1;
        }
        if self.committee.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.MultisigAggregatedSignature", len)?;
        if !self.signatures.is_empty() {
            struct_ser.serialize_field("signatures", &self.signatures)?;
        }
        if let Some(v) = self.bitmap.as_ref() {
            struct_ser.serialize_field("bitmap", v)?;
        }
        if let Some(v) = self.committee.as_ref() {
            struct_ser.serialize_field("committee", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MultisigAggregatedSignature {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "signatures",
            "bitmap",
            "committee",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Signatures,
            Bitmap,
            Committee,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "signatures" => Ok(GeneratedField::Signatures),
                            "bitmap" => Ok(GeneratedField::Bitmap),
                            "committee" => Ok(GeneratedField::Committee),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = MultisigAggregatedSignature;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.MultisigAggregatedSignature")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MultisigAggregatedSignature, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut signatures__ = None;
                let mut bitmap__ = None;
                let mut committee__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Signatures => {
                            if signatures__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signatures"));
                            }
                            signatures__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Bitmap => {
                            if bitmap__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bitmap"));
                            }
                            bitmap__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Committee => {
                            if committee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("committee"));
                            }
                            committee__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MultisigAggregatedSignature {
                    signatures: signatures__.unwrap_or_default(),
                    bitmap: bitmap__,
                    committee: committee__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.MultisigAggregatedSignature", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for MultisigCommittee {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.members.is_empty() {
            len += 1;
        }
        if self.threshold.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.MultisigCommittee", len)?;
        if !self.members.is_empty() {
            struct_ser.serialize_field("members", &self.members)?;
        }
        if let Some(v) = self.threshold.as_ref() {
            struct_ser.serialize_field("threshold", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MultisigCommittee {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "members",
            "threshold",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Members,
            Threshold,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "members" => Ok(GeneratedField::Members),
                            "threshold" => Ok(GeneratedField::Threshold),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = MultisigCommittee;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.MultisigCommittee")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MultisigCommittee, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut members__ = None;
                let mut threshold__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Members => {
                            if members__.is_some() {
                                return Err(serde::de::Error::duplicate_field("members"));
                            }
                            members__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Threshold => {
                            if threshold__.is_some() {
                                return Err(serde::de::Error::duplicate_field("threshold"));
                            }
                            threshold__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MultisigCommittee {
                    members: members__.unwrap_or_default(),
                    threshold: threshold__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.MultisigCommittee", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for MultisigMember {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.public_key.is_some() {
            len += 1;
        }
        if self.weight.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.MultisigMember", len)?;
        if let Some(v) = self.public_key.as_ref() {
            struct_ser.serialize_field("publicKey", v)?;
        }
        if let Some(v) = self.weight.as_ref() {
            struct_ser.serialize_field("weight", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MultisigMember {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "public_key",
            "publicKey",
            "weight",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            PublicKey,
            Weight,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "publicKey" | "public_key" => Ok(GeneratedField::PublicKey),
                            "weight" => Ok(GeneratedField::Weight),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = MultisigMember;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.MultisigMember")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MultisigMember, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut public_key__ = None;
                let mut weight__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::PublicKey => {
                            if public_key__.is_some() {
                                return Err(serde::de::Error::duplicate_field("publicKey"));
                            }
                            public_key__ = map_.next_value()?;
                        }
                        GeneratedField::Weight => {
                            if weight__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weight"));
                            }
                            weight__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MultisigMember {
                    public_key: public_key__,
                    weight: weight__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.MultisigMember", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for MultisigMemberPublicKey {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.scheme.is_some() {
            len += 1;
        }
        if self.public_key.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.MultisigMemberPublicKey", len)?;
        if let Some(v) = self.scheme.as_ref() {
            let v = SignatureScheme::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("scheme", &v)?;
        }
        if let Some(v) = self.public_key.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("publicKey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MultisigMemberPublicKey {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "scheme",
            "public_key",
            "publicKey",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Scheme,
            PublicKey,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "scheme" => Ok(GeneratedField::Scheme),
                            "publicKey" | "public_key" => Ok(GeneratedField::PublicKey),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = MultisigMemberPublicKey;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.MultisigMemberPublicKey")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MultisigMemberPublicKey, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut scheme__ = None;
                let mut public_key__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Scheme => {
                            if scheme__.is_some() {
                                return Err(serde::de::Error::duplicate_field("scheme"));
                            }
                            scheme__ = map_.next_value::<::std::option::Option<SignatureScheme>>()?.map(|x| x as i32);
                        }
                        GeneratedField::PublicKey => {
                            if public_key__.is_some() {
                                return Err(serde::de::Error::duplicate_field("publicKey"));
                            }
                            public_key__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MultisigMemberPublicKey {
                    scheme: scheme__,
                    public_key: public_key__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.MultisigMemberPublicKey", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for MultisigMemberSignature {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.scheme.is_some() {
            len += 1;
        }
        if self.signature.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.MultisigMemberSignature", len)?;
        if let Some(v) = self.scheme.as_ref() {
            let v = SignatureScheme::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("scheme", &v)?;
        }
        if let Some(v) = self.signature.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("signature", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MultisigMemberSignature {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "scheme",
            "signature",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Scheme,
            Signature,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "scheme" => Ok(GeneratedField::Scheme),
                            "signature" => Ok(GeneratedField::Signature),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = MultisigMemberSignature;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.MultisigMemberSignature")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MultisigMemberSignature, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut scheme__ = None;
                let mut signature__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Scheme => {
                            if scheme__.is_some() {
                                return Err(serde::de::Error::duplicate_field("scheme"));
                            }
                            scheme__ = map_.next_value::<::std::option::Option<SignatureScheme>>()?.map(|x| x as i32);
                        }
                        GeneratedField::Signature => {
                            if signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signature"));
                            }
                            signature__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MultisigMemberSignature {
                    scheme: scheme__,
                    signature: signature__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.MultisigMemberSignature", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Object {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.object_id.is_some() {
            len += 1;
        }
        if self.version.is_some() {
            len += 1;
        }
        if self.digest.is_some() {
            len += 1;
        }
        if self.owner.is_some() {
            len += 1;
        }
        if self.object_type.is_some() {
            len += 1;
        }
        if self.contents.is_some() {
            len += 1;
        }
        if self.previous_transaction.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Object", len)?;
        if let Some(v) = self.object_id.as_ref() {
            struct_ser.serialize_field("objectId", v)?;
        }
        if let Some(v) = self.version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("version", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.owner.as_ref() {
            struct_ser.serialize_field("owner", v)?;
        }
        if let Some(v) = self.object_type.as_ref() {
            struct_ser.serialize_field("objectType", v)?;
        }
        if let Some(v) = self.contents.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("contents", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.previous_transaction.as_ref() {
            struct_ser.serialize_field("previousTransaction", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Object {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "object_id",
            "objectId",
            "version",
            "digest",
            "owner",
            "object_type",
            "objectType",
            "contents",
            "previous_transaction",
            "previousTransaction",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ObjectId,
            Version,
            Digest,
            Owner,
            ObjectType,
            Contents,
            PreviousTransaction,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "objectId" | "object_id" => Ok(GeneratedField::ObjectId),
                            "version" => Ok(GeneratedField::Version),
                            "digest" => Ok(GeneratedField::Digest),
                            "owner" => Ok(GeneratedField::Owner),
                            "objectType" | "object_type" => Ok(GeneratedField::ObjectType),
                            "contents" => Ok(GeneratedField::Contents),
                            "previousTransaction" | "previous_transaction" => Ok(GeneratedField::PreviousTransaction),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Object;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Object")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Object, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut object_id__ = None;
                let mut version__ = None;
                let mut digest__ = None;
                let mut owner__ = None;
                let mut object_type__ = None;
                let mut contents__ = None;
                let mut previous_transaction__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ObjectId => {
                            if object_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectId"));
                            }
                            object_id__ = map_.next_value()?;
                        }
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::Owner => {
                            if owner__.is_some() {
                                return Err(serde::de::Error::duplicate_field("owner"));
                            }
                            owner__ = map_.next_value()?;
                        }
                        GeneratedField::ObjectType => {
                            if object_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectType"));
                            }
                            object_type__ = map_.next_value()?;
                        }
                        GeneratedField::Contents => {
                            if contents__.is_some() {
                                return Err(serde::de::Error::duplicate_field("contents"));
                            }
                            contents__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PreviousTransaction => {
                            if previous_transaction__.is_some() {
                                return Err(serde::de::Error::duplicate_field("previousTransaction"));
                            }
                            previous_transaction__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Object {
                    object_id: object_id__,
                    version: version__,
                    digest: digest__,
                    owner: owner__,
                    object_type: object_type__,
                    contents: contents__,
                    previous_transaction: previous_transaction__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Object", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ObjectReference {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.object_id.is_some() {
            len += 1;
        }
        if self.version.is_some() {
            len += 1;
        }
        if self.digest.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ObjectReference", len)?;
        if let Some(v) = self.object_id.as_ref() {
            struct_ser.serialize_field("objectId", v)?;
        }
        if let Some(v) = self.version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("version", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ObjectReference {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "object_id",
            "objectId",
            "version",
            "digest",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ObjectId,
            Version,
            Digest,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "objectId" | "object_id" => Ok(GeneratedField::ObjectId),
                            "version" => Ok(GeneratedField::Version),
                            "digest" => Ok(GeneratedField::Digest),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ObjectReference;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ObjectReference")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ObjectReference, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut object_id__ = None;
                let mut version__ = None;
                let mut digest__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ObjectId => {
                            if object_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectId"));
                            }
                            object_id__ = map_.next_value()?;
                        }
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ObjectReference {
                    object_id: object_id__,
                    version: version__,
                    digest: digest__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ObjectReference", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ObjectSet {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.objects.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ObjectSet", len)?;
        if !self.objects.is_empty() {
            struct_ser.serialize_field("objects", &self.objects)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ObjectSet {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "objects",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Objects,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "objects" => Ok(GeneratedField::Objects),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ObjectSet;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ObjectSet")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ObjectSet, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut objects__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Objects => {
                            if objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objects"));
                            }
                            objects__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ObjectSet {
                    objects: objects__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ObjectSet", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Owner {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.kind.is_some() {
            len += 1;
        }
        if self.address.is_some() {
            len += 1;
        }
        if self.version.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Owner", len)?;
        if let Some(v) = self.kind.as_ref() {
            let v = owner::OwnerKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.address.as_ref() {
            struct_ser.serialize_field("address", v)?;
        }
        if let Some(v) = self.version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("version", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Owner {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "address",
            "version",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            Address,
            Version,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "kind" => Ok(GeneratedField::Kind),
                            "address" => Ok(GeneratedField::Address),
                            "version" => Ok(GeneratedField::Version),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Owner;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Owner")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Owner, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut address__ = None;
                let mut version__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<owner::OwnerKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::Address => {
                            if address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("address"));
                            }
                            address__ = map_.next_value()?;
                        }
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Owner {
                    kind: kind__,
                    address: address__,
                    version: version__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Owner", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for owner::OwnerKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "OWNER_KIND_UNKNOWN",
            Self::Address => "ADDRESS",
            Self::Shared => "SHARED",
            Self::Immutable => "IMMUTABLE",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for owner::OwnerKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "OWNER_KIND_UNKNOWN",
            "ADDRESS",
            "SHARED",
            "IMMUTABLE",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = owner::OwnerKind;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "OWNER_KIND_UNKNOWN" => Ok(owner::OwnerKind::Unknown),
                    "ADDRESS" => Ok(owner::OwnerKind::Address),
                    "SHARED" => Ok(owner::OwnerKind::Shared),
                    "IMMUTABLE" => Ok(owner::OwnerKind::Immutable),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for PayCoins {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.coins.is_empty() {
            len += 1;
        }
        if !self.amounts.is_empty() {
            len += 1;
        }
        if !self.recipients.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.PayCoins", len)?;
        if !self.coins.is_empty() {
            struct_ser.serialize_field("coins", &self.coins)?;
        }
        if !self.amounts.is_empty() {
            struct_ser.serialize_field("amounts", &self.amounts.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        if !self.recipients.is_empty() {
            struct_ser.serialize_field("recipients", &self.recipients)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for PayCoins {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "coins",
            "amounts",
            "recipients",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Coins,
            Amounts,
            Recipients,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "coins" => Ok(GeneratedField::Coins),
                            "amounts" => Ok(GeneratedField::Amounts),
                            "recipients" => Ok(GeneratedField::Recipients),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = PayCoins;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.PayCoins")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<PayCoins, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut coins__ = None;
                let mut amounts__ = None;
                let mut recipients__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Coins => {
                            if coins__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coins"));
                            }
                            coins__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Amounts => {
                            if amounts__.is_some() {
                                return Err(serde::de::Error::duplicate_field("amounts"));
                            }
                            amounts__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::Recipients => {
                            if recipients__.is_some() {
                                return Err(serde::de::Error::duplicate_field("recipients"));
                            }
                            recipients__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(PayCoins {
                    coins: coins__.unwrap_or_default(),
                    amounts: amounts__.unwrap_or_default(),
                    recipients: recipients__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.PayCoins", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for PendingModelUpdate {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.weights_url_commitment.is_some() {
            len += 1;
        }
        if self.weights_commitment.is_some() {
            len += 1;
        }
        if self.commit_epoch.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.PendingModelUpdate", len)?;
        if let Some(v) = self.weights_url_commitment.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("weightsUrlCommitment", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.weights_commitment.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("weightsCommitment", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.commit_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("commitEpoch", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for PendingModelUpdate {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "weights_url_commitment",
            "weightsUrlCommitment",
            "weights_commitment",
            "weightsCommitment",
            "commit_epoch",
            "commitEpoch",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            WeightsUrlCommitment,
            WeightsCommitment,
            CommitEpoch,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "weightsUrlCommitment" | "weights_url_commitment" => Ok(GeneratedField::WeightsUrlCommitment),
                            "weightsCommitment" | "weights_commitment" => Ok(GeneratedField::WeightsCommitment),
                            "commitEpoch" | "commit_epoch" => Ok(GeneratedField::CommitEpoch),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = PendingModelUpdate;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.PendingModelUpdate")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<PendingModelUpdate, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut weights_url_commitment__ = None;
                let mut weights_commitment__ = None;
                let mut commit_epoch__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::WeightsUrlCommitment => {
                            if weights_url_commitment__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weightsUrlCommitment"));
                            }
                            weights_url_commitment__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::WeightsCommitment => {
                            if weights_commitment__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weightsCommitment"));
                            }
                            weights_commitment__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::CommitEpoch => {
                            if commit_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commitEpoch"));
                            }
                            commit_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(PendingModelUpdate {
                    weights_url_commitment: weights_url_commitment__,
                    weights_commitment: weights_commitment__,
                    commit_epoch: commit_epoch__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.PendingModelUpdate", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for PoolTokenExchangeRate {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.soma_amount.is_some() {
            len += 1;
        }
        if self.pool_token_amount.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.PoolTokenExchangeRate", len)?;
        if let Some(v) = self.soma_amount.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("somaAmount", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.pool_token_amount.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("poolTokenAmount", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for PoolTokenExchangeRate {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "soma_amount",
            "somaAmount",
            "pool_token_amount",
            "poolTokenAmount",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            SomaAmount,
            PoolTokenAmount,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "somaAmount" | "soma_amount" => Ok(GeneratedField::SomaAmount),
                            "poolTokenAmount" | "pool_token_amount" => Ok(GeneratedField::PoolTokenAmount),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = PoolTokenExchangeRate;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.PoolTokenExchangeRate")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<PoolTokenExchangeRate, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut soma_amount__ = None;
                let mut pool_token_amount__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::SomaAmount => {
                            if soma_amount__.is_some() {
                                return Err(serde::de::Error::duplicate_field("somaAmount"));
                            }
                            soma_amount__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PoolTokenAmount => {
                            if pool_token_amount__.is_some() {
                                return Err(serde::de::Error::duplicate_field("poolTokenAmount"));
                            }
                            pool_token_amount__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(PoolTokenExchangeRate {
                    soma_amount: soma_amount__,
                    pool_token_amount: pool_token_amount__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.PoolTokenExchangeRate", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ProtocolConfig {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.protocol_version.is_some() {
            len += 1;
        }
        if !self.feature_flags.is_empty() {
            len += 1;
        }
        if !self.attributes.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ProtocolConfig", len)?;
        if let Some(v) = self.protocol_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("protocolVersion", ToString::to_string(&v).as_str())?;
        }
        if !self.feature_flags.is_empty() {
            struct_ser.serialize_field("featureFlags", &self.feature_flags)?;
        }
        if !self.attributes.is_empty() {
            struct_ser.serialize_field("attributes", &self.attributes)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ProtocolConfig {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "protocol_version",
            "protocolVersion",
            "feature_flags",
            "featureFlags",
            "attributes",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ProtocolVersion,
            FeatureFlags,
            Attributes,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "protocolVersion" | "protocol_version" => Ok(GeneratedField::ProtocolVersion),
                            "featureFlags" | "feature_flags" => Ok(GeneratedField::FeatureFlags),
                            "attributes" => Ok(GeneratedField::Attributes),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ProtocolConfig;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ProtocolConfig")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ProtocolConfig, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut protocol_version__ = None;
                let mut feature_flags__ = None;
                let mut attributes__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ProtocolVersion => {
                            if protocol_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("protocolVersion"));
                            }
                            protocol_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::FeatureFlags => {
                            if feature_flags__.is_some() {
                                return Err(serde::de::Error::duplicate_field("featureFlags"));
                            }
                            feature_flags__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::Attributes => {
                            if attributes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("attributes"));
                            }
                            attributes__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ProtocolConfig {
                    protocol_version: protocol_version__,
                    feature_flags: feature_flags__.unwrap_or_default(),
                    attributes: attributes__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ProtocolConfig", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for RemoveValidator {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.pubkey_bytes.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.RemoveValidator", len)?;
        if let Some(v) = self.pubkey_bytes.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("pubkeyBytes", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for RemoveValidator {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "pubkey_bytes",
            "pubkeyBytes",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            PubkeyBytes,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "pubkeyBytes" | "pubkey_bytes" => Ok(GeneratedField::PubkeyBytes),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = RemoveValidator;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.RemoveValidator")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<RemoveValidator, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut pubkey_bytes__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::PubkeyBytes => {
                            if pubkey_bytes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pubkeyBytes"));
                            }
                            pubkey_bytes__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(RemoveValidator {
                    pubkey_bytes: pubkey_bytes__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.RemoveValidator", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ReportChallenge {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.challenge_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ReportChallenge", len)?;
        if let Some(v) = self.challenge_id.as_ref() {
            struct_ser.serialize_field("challengeId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ReportChallenge {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "challenge_id",
            "challengeId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ChallengeId,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "challengeId" | "challenge_id" => Ok(GeneratedField::ChallengeId),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ReportChallenge;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ReportChallenge")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ReportChallenge, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut challenge_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ChallengeId => {
                            if challenge_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challengeId"));
                            }
                            challenge_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ReportChallenge {
                    challenge_id: challenge_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ReportChallenge", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ReportModel {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.model_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ReportModel", len)?;
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ReportModel {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "model_id",
            "modelId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ModelId,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ReportModel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ReportModel")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ReportModel, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut model_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ReportModel {
                    model_id: model_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ReportModel", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ReportSubmission {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.target_id.is_some() {
            len += 1;
        }
        if self.challenger.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ReportSubmission", len)?;
        if let Some(v) = self.target_id.as_ref() {
            struct_ser.serialize_field("targetId", v)?;
        }
        if let Some(v) = self.challenger.as_ref() {
            struct_ser.serialize_field("challenger", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ReportSubmission {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "target_id",
            "targetId",
            "challenger",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TargetId,
            Challenger,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "targetId" | "target_id" => Ok(GeneratedField::TargetId),
                            "challenger" => Ok(GeneratedField::Challenger),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ReportSubmission;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ReportSubmission")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ReportSubmission, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut target_id__ = None;
                let mut challenger__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::TargetId => {
                            if target_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetId"));
                            }
                            target_id__ = map_.next_value()?;
                        }
                        GeneratedField::Challenger => {
                            if challenger__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challenger"));
                            }
                            challenger__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ReportSubmission {
                    target_id: target_id__,
                    challenger: challenger__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ReportSubmission", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ReportValidator {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.reportee.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ReportValidator", len)?;
        if let Some(v) = self.reportee.as_ref() {
            struct_ser.serialize_field("reportee", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ReportValidator {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "reportee",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Reportee,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "reportee" => Ok(GeneratedField::Reportee),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ReportValidator;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ReportValidator")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ReportValidator, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut reportee__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Reportee => {
                            if reportee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reportee"));
                            }
                            reportee__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ReportValidator {
                    reportee: reportee__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ReportValidator", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ReporterSet {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.reporters.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ReporterSet", len)?;
        if !self.reporters.is_empty() {
            struct_ser.serialize_field("reporters", &self.reporters)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ReporterSet {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "reporters",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Reporters,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "reporters" => Ok(GeneratedField::Reporters),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ReporterSet;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ReporterSet")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ReporterSet, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut reporters__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Reporters => {
                            if reporters__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reporters"));
                            }
                            reporters__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ReporterSet {
                    reporters: reporters__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ReporterSet", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for RevealModel {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.model_id.is_some() {
            len += 1;
        }
        if self.weights_manifest.is_some() {
            len += 1;
        }
        if !self.embedding.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.RevealModel", len)?;
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        if let Some(v) = self.weights_manifest.as_ref() {
            struct_ser.serialize_field("weightsManifest", v)?;
        }
        if !self.embedding.is_empty() {
            struct_ser.serialize_field("embedding", &self.embedding)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for RevealModel {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "model_id",
            "modelId",
            "weights_manifest",
            "weightsManifest",
            "embedding",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ModelId,
            WeightsManifest,
            Embedding,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            "weightsManifest" | "weights_manifest" => Ok(GeneratedField::WeightsManifest),
                            "embedding" => Ok(GeneratedField::Embedding),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = RevealModel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.RevealModel")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<RevealModel, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut model_id__ = None;
                let mut weights_manifest__ = None;
                let mut embedding__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::WeightsManifest => {
                            if weights_manifest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weightsManifest"));
                            }
                            weights_manifest__ = map_.next_value()?;
                        }
                        GeneratedField::Embedding => {
                            if embedding__.is_some() {
                                return Err(serde::de::Error::duplicate_field("embedding"));
                            }
                            embedding__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(RevealModel {
                    model_id: model_id__,
                    weights_manifest: weights_manifest__,
                    embedding: embedding__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.RevealModel", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for RevealModelUpdate {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.model_id.is_some() {
            len += 1;
        }
        if self.weights_manifest.is_some() {
            len += 1;
        }
        if !self.embedding.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.RevealModelUpdate", len)?;
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        if let Some(v) = self.weights_manifest.as_ref() {
            struct_ser.serialize_field("weightsManifest", v)?;
        }
        if !self.embedding.is_empty() {
            struct_ser.serialize_field("embedding", &self.embedding)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for RevealModelUpdate {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "model_id",
            "modelId",
            "weights_manifest",
            "weightsManifest",
            "embedding",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ModelId,
            WeightsManifest,
            Embedding,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            "weightsManifest" | "weights_manifest" => Ok(GeneratedField::WeightsManifest),
                            "embedding" => Ok(GeneratedField::Embedding),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = RevealModelUpdate;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.RevealModelUpdate")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<RevealModelUpdate, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut model_id__ = None;
                let mut weights_manifest__ = None;
                let mut embedding__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::WeightsManifest => {
                            if weights_manifest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weightsManifest"));
                            }
                            weights_manifest__ = map_.next_value()?;
                        }
                        GeneratedField::Embedding => {
                            if embedding__.is_some() {
                                return Err(serde::de::Error::duplicate_field("embedding"));
                            }
                            embedding__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(RevealModelUpdate {
                    model_id: model_id__,
                    weights_manifest: weights_manifest__,
                    embedding: embedding__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.RevealModelUpdate", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SetCommissionRate {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.new_rate.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SetCommissionRate", len)?;
        if let Some(v) = self.new_rate.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("newRate", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SetCommissionRate {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "new_rate",
            "newRate",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            NewRate,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "newRate" | "new_rate" => Ok(GeneratedField::NewRate),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SetCommissionRate;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SetCommissionRate")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SetCommissionRate, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut new_rate__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::NewRate => {
                            if new_rate__.is_some() {
                                return Err(serde::de::Error::duplicate_field("newRate"));
                            }
                            new_rate__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SetCommissionRate {
                    new_rate: new_rate__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SetCommissionRate", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SetModelCommissionRate {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.model_id.is_some() {
            len += 1;
        }
        if self.new_rate.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SetModelCommissionRate", len)?;
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        if let Some(v) = self.new_rate.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("newRate", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SetModelCommissionRate {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "model_id",
            "modelId",
            "new_rate",
            "newRate",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ModelId,
            NewRate,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            "newRate" | "new_rate" => Ok(GeneratedField::NewRate),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SetModelCommissionRate;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SetModelCommissionRate")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SetModelCommissionRate, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut model_id__ = None;
                let mut new_rate__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::NewRate => {
                            if new_rate__.is_some() {
                                return Err(serde::de::Error::duplicate_field("newRate"));
                            }
                            new_rate__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SetModelCommissionRate {
                    model_id: model_id__,
                    new_rate: new_rate__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SetModelCommissionRate", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SignatureScheme {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Ed25519 => "ED25519",
            Self::Bls12381 => "BLS12381",
            Self::Multisig => "MULTISIG",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for SignatureScheme {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "ED25519",
            "BLS12381",
            "MULTISIG",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SignatureScheme;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "ED25519" => Ok(SignatureScheme::Ed25519),
                    "BLS12381" => Ok(SignatureScheme::Bls12381),
                    "MULTISIG" => Ok(SignatureScheme::Multisig),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for SimpleSignature {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.scheme.is_some() {
            len += 1;
        }
        if self.signature.is_some() {
            len += 1;
        }
        if self.public_key.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SimpleSignature", len)?;
        if let Some(v) = self.scheme.as_ref() {
            let v = SignatureScheme::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("scheme", &v)?;
        }
        if let Some(v) = self.signature.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("signature", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.public_key.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("publicKey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SimpleSignature {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "scheme",
            "signature",
            "public_key",
            "publicKey",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Scheme,
            Signature,
            PublicKey,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "scheme" => Ok(GeneratedField::Scheme),
                            "signature" => Ok(GeneratedField::Signature),
                            "publicKey" | "public_key" => Ok(GeneratedField::PublicKey),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SimpleSignature;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SimpleSignature")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SimpleSignature, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut scheme__ = None;
                let mut signature__ = None;
                let mut public_key__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Scheme => {
                            if scheme__.is_some() {
                                return Err(serde::de::Error::duplicate_field("scheme"));
                            }
                            scheme__ = map_.next_value::<::std::option::Option<SignatureScheme>>()?.map(|x| x as i32);
                        }
                        GeneratedField::Signature => {
                            if signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signature"));
                            }
                            signature__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PublicKey => {
                            if public_key__.is_some() {
                                return Err(serde::de::Error::duplicate_field("publicKey"));
                            }
                            public_key__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SimpleSignature {
                    scheme: scheme__,
                    signature: signature__,
                    public_key: public_key__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SimpleSignature", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SimulateTransactionRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.transaction.is_some() {
            len += 1;
        }
        if self.read_mask.is_some() {
            len += 1;
        }
        if self.checks.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SimulateTransactionRequest", len)?;
        if let Some(v) = self.transaction.as_ref() {
            struct_ser.serialize_field("transaction", v)?;
        }
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        if let Some(v) = self.checks.as_ref() {
            let v = simulate_transaction_request::TransactionChecks::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("checks", &v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SimulateTransactionRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "transaction",
            "read_mask",
            "readMask",
            "checks",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Transaction,
            ReadMask,
            Checks,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "transaction" => Ok(GeneratedField::Transaction),
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            "checks" => Ok(GeneratedField::Checks),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SimulateTransactionRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SimulateTransactionRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SimulateTransactionRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut transaction__ = None;
                let mut read_mask__ = None;
                let mut checks__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Transaction => {
                            if transaction__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transaction"));
                            }
                            transaction__ = map_.next_value()?;
                        }
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::Checks => {
                            if checks__.is_some() {
                                return Err(serde::de::Error::duplicate_field("checks"));
                            }
                            checks__ = map_.next_value::<::std::option::Option<simulate_transaction_request::TransactionChecks>>()?.map(|x| x as i32);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SimulateTransactionRequest {
                    transaction: transaction__,
                    read_mask: read_mask__,
                    checks: checks__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SimulateTransactionRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for simulate_transaction_request::TransactionChecks {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Enabled => "ENABLED",
            Self::Disabled => "DISABLED",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for simulate_transaction_request::TransactionChecks {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "ENABLED",
            "DISABLED",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = simulate_transaction_request::TransactionChecks;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "ENABLED" => Ok(simulate_transaction_request::TransactionChecks::Enabled),
                    "DISABLED" => Ok(simulate_transaction_request::TransactionChecks::Disabled),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for SimulateTransactionResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.transaction.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SimulateTransactionResponse", len)?;
        if let Some(v) = self.transaction.as_ref() {
            struct_ser.serialize_field("transaction", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SimulateTransactionResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "transaction",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Transaction,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "transaction" => Ok(GeneratedField::Transaction),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SimulateTransactionResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SimulateTransactionResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SimulateTransactionResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut transaction__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Transaction => {
                            if transaction__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transaction"));
                            }
                            transaction__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SimulateTransactionResponse {
                    transaction: transaction__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SimulateTransactionResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for StakingPool {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.id.is_some() {
            len += 1;
        }
        if self.activation_epoch.is_some() {
            len += 1;
        }
        if self.deactivation_epoch.is_some() {
            len += 1;
        }
        if self.soma_balance.is_some() {
            len += 1;
        }
        if self.rewards_pool.is_some() {
            len += 1;
        }
        if self.pool_token_balance.is_some() {
            len += 1;
        }
        if !self.exchange_rates.is_empty() {
            len += 1;
        }
        if self.pending_stake.is_some() {
            len += 1;
        }
        if self.pending_total_soma_withdraw.is_some() {
            len += 1;
        }
        if self.pending_pool_token_withdraw.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.StakingPool", len)?;
        if let Some(v) = self.id.as_ref() {
            struct_ser.serialize_field("id", v)?;
        }
        if let Some(v) = self.activation_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("activationEpoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.deactivation_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("deactivationEpoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.soma_balance.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("somaBalance", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.rewards_pool.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("rewardsPool", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.pool_token_balance.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("poolTokenBalance", ToString::to_string(&v).as_str())?;
        }
        if !self.exchange_rates.is_empty() {
            struct_ser.serialize_field("exchangeRates", &self.exchange_rates)?;
        }
        if let Some(v) = self.pending_stake.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("pendingStake", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.pending_total_soma_withdraw.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("pendingTotalSomaWithdraw", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.pending_pool_token_withdraw.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("pendingPoolTokenWithdraw", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for StakingPool {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "id",
            "activation_epoch",
            "activationEpoch",
            "deactivation_epoch",
            "deactivationEpoch",
            "soma_balance",
            "somaBalance",
            "rewards_pool",
            "rewardsPool",
            "pool_token_balance",
            "poolTokenBalance",
            "exchange_rates",
            "exchangeRates",
            "pending_stake",
            "pendingStake",
            "pending_total_soma_withdraw",
            "pendingTotalSomaWithdraw",
            "pending_pool_token_withdraw",
            "pendingPoolTokenWithdraw",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Id,
            ActivationEpoch,
            DeactivationEpoch,
            SomaBalance,
            RewardsPool,
            PoolTokenBalance,
            ExchangeRates,
            PendingStake,
            PendingTotalSomaWithdraw,
            PendingPoolTokenWithdraw,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "id" => Ok(GeneratedField::Id),
                            "activationEpoch" | "activation_epoch" => Ok(GeneratedField::ActivationEpoch),
                            "deactivationEpoch" | "deactivation_epoch" => Ok(GeneratedField::DeactivationEpoch),
                            "somaBalance" | "soma_balance" => Ok(GeneratedField::SomaBalance),
                            "rewardsPool" | "rewards_pool" => Ok(GeneratedField::RewardsPool),
                            "poolTokenBalance" | "pool_token_balance" => Ok(GeneratedField::PoolTokenBalance),
                            "exchangeRates" | "exchange_rates" => Ok(GeneratedField::ExchangeRates),
                            "pendingStake" | "pending_stake" => Ok(GeneratedField::PendingStake),
                            "pendingTotalSomaWithdraw" | "pending_total_soma_withdraw" => Ok(GeneratedField::PendingTotalSomaWithdraw),
                            "pendingPoolTokenWithdraw" | "pending_pool_token_withdraw" => Ok(GeneratedField::PendingPoolTokenWithdraw),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = StakingPool;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.StakingPool")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<StakingPool, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut id__ = None;
                let mut activation_epoch__ = None;
                let mut deactivation_epoch__ = None;
                let mut soma_balance__ = None;
                let mut rewards_pool__ = None;
                let mut pool_token_balance__ = None;
                let mut exchange_rates__ = None;
                let mut pending_stake__ = None;
                let mut pending_total_soma_withdraw__ = None;
                let mut pending_pool_token_withdraw__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Id => {
                            if id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("id"));
                            }
                            id__ = map_.next_value()?;
                        }
                        GeneratedField::ActivationEpoch => {
                            if activation_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("activationEpoch"));
                            }
                            activation_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::DeactivationEpoch => {
                            if deactivation_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("deactivationEpoch"));
                            }
                            deactivation_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::SomaBalance => {
                            if soma_balance__.is_some() {
                                return Err(serde::de::Error::duplicate_field("somaBalance"));
                            }
                            soma_balance__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::RewardsPool => {
                            if rewards_pool__.is_some() {
                                return Err(serde::de::Error::duplicate_field("rewardsPool"));
                            }
                            rewards_pool__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PoolTokenBalance => {
                            if pool_token_balance__.is_some() {
                                return Err(serde::de::Error::duplicate_field("poolTokenBalance"));
                            }
                            pool_token_balance__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ExchangeRates => {
                            if exchange_rates__.is_some() {
                                return Err(serde::de::Error::duplicate_field("exchangeRates"));
                            }
                            exchange_rates__ = Some(
                                map_.next_value::<std::collections::BTreeMap<crate::utils::_serde::NumberDeserialize<u64>, _>>()?
                                    .into_iter().map(|(k,v)| (k.0, v)).collect()
                            );
                        }
                        GeneratedField::PendingStake => {
                            if pending_stake__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pendingStake"));
                            }
                            pending_stake__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PendingTotalSomaWithdraw => {
                            if pending_total_soma_withdraw__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pendingTotalSomaWithdraw"));
                            }
                            pending_total_soma_withdraw__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PendingPoolTokenWithdraw => {
                            if pending_pool_token_withdraw__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pendingPoolTokenWithdraw"));
                            }
                            pending_pool_token_withdraw__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(StakingPool {
                    id: id__,
                    activation_epoch: activation_epoch__,
                    deactivation_epoch: deactivation_epoch__,
                    soma_balance: soma_balance__,
                    rewards_pool: rewards_pool__,
                    pool_token_balance: pool_token_balance__,
                    exchange_rates: exchange_rates__.unwrap_or_default(),
                    pending_stake: pending_stake__,
                    pending_total_soma_withdraw: pending_total_soma_withdraw__,
                    pending_pool_token_withdraw: pending_pool_token_withdraw__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.StakingPool", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Submission {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.miner.is_some() {
            len += 1;
        }
        if self.data_commitment.is_some() {
            len += 1;
        }
        if self.data_manifest.is_some() {
            len += 1;
        }
        if self.model_id.is_some() {
            len += 1;
        }
        if !self.embedding.is_empty() {
            len += 1;
        }
        if self.distance_score.is_some() {
            len += 1;
        }
        if self.bond_amount.is_some() {
            len += 1;
        }
        if self.submit_epoch.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Submission", len)?;
        if let Some(v) = self.miner.as_ref() {
            struct_ser.serialize_field("miner", v)?;
        }
        if let Some(v) = self.data_commitment.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("dataCommitment", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.data_manifest.as_ref() {
            struct_ser.serialize_field("dataManifest", v)?;
        }
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        if !self.embedding.is_empty() {
            struct_ser.serialize_field("embedding", &self.embedding)?;
        }
        if let Some(v) = self.distance_score.as_ref() {
            struct_ser.serialize_field("distanceScore", v)?;
        }
        if let Some(v) = self.bond_amount.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("bondAmount", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.submit_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("submitEpoch", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Submission {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "miner",
            "data_commitment",
            "dataCommitment",
            "data_manifest",
            "dataManifest",
            "model_id",
            "modelId",
            "embedding",
            "distance_score",
            "distanceScore",
            "bond_amount",
            "bondAmount",
            "submit_epoch",
            "submitEpoch",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Miner,
            DataCommitment,
            DataManifest,
            ModelId,
            Embedding,
            DistanceScore,
            BondAmount,
            SubmitEpoch,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "miner" => Ok(GeneratedField::Miner),
                            "dataCommitment" | "data_commitment" => Ok(GeneratedField::DataCommitment),
                            "dataManifest" | "data_manifest" => Ok(GeneratedField::DataManifest),
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            "embedding" => Ok(GeneratedField::Embedding),
                            "distanceScore" | "distance_score" => Ok(GeneratedField::DistanceScore),
                            "bondAmount" | "bond_amount" => Ok(GeneratedField::BondAmount),
                            "submitEpoch" | "submit_epoch" => Ok(GeneratedField::SubmitEpoch),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Submission;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Submission")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Submission, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut miner__ = None;
                let mut data_commitment__ = None;
                let mut data_manifest__ = None;
                let mut model_id__ = None;
                let mut embedding__ = None;
                let mut distance_score__ = None;
                let mut bond_amount__ = None;
                let mut submit_epoch__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Miner => {
                            if miner__.is_some() {
                                return Err(serde::de::Error::duplicate_field("miner"));
                            }
                            miner__ = map_.next_value()?;
                        }
                        GeneratedField::DataCommitment => {
                            if data_commitment__.is_some() {
                                return Err(serde::de::Error::duplicate_field("dataCommitment"));
                            }
                            data_commitment__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::DataManifest => {
                            if data_manifest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("dataManifest"));
                            }
                            data_manifest__ = map_.next_value()?;
                        }
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::Embedding => {
                            if embedding__.is_some() {
                                return Err(serde::de::Error::duplicate_field("embedding"));
                            }
                            embedding__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::DistanceScore => {
                            if distance_score__.is_some() {
                                return Err(serde::de::Error::duplicate_field("distanceScore"));
                            }
                            distance_score__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::BondAmount => {
                            if bond_amount__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bondAmount"));
                            }
                            bond_amount__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::SubmitEpoch => {
                            if submit_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("submitEpoch"));
                            }
                            submit_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Submission {
                    miner: miner__,
                    data_commitment: data_commitment__,
                    data_manifest: data_manifest__,
                    model_id: model_id__,
                    embedding: embedding__.unwrap_or_default(),
                    distance_score: distance_score__,
                    bond_amount: bond_amount__,
                    submit_epoch: submit_epoch__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Submission", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SubmissionManifest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.manifest.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SubmissionManifest", len)?;
        if let Some(v) = self.manifest.as_ref() {
            struct_ser.serialize_field("manifest", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SubmissionManifest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "manifest",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Manifest,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "manifest" => Ok(GeneratedField::Manifest),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SubmissionManifest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SubmissionManifest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SubmissionManifest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut manifest__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Manifest => {
                            if manifest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("manifest"));
                            }
                            manifest__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SubmissionManifest {
                    manifest: manifest__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SubmissionManifest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SubmitData {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.target_id.is_some() {
            len += 1;
        }
        if self.data_commitment.is_some() {
            len += 1;
        }
        if self.data_manifest.is_some() {
            len += 1;
        }
        if self.model_id.is_some() {
            len += 1;
        }
        if !self.embedding.is_empty() {
            len += 1;
        }
        if self.distance_score.is_some() {
            len += 1;
        }
        if self.bond_coin.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SubmitData", len)?;
        if let Some(v) = self.target_id.as_ref() {
            struct_ser.serialize_field("targetId", v)?;
        }
        if let Some(v) = self.data_commitment.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("dataCommitment", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.data_manifest.as_ref() {
            struct_ser.serialize_field("dataManifest", v)?;
        }
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        if !self.embedding.is_empty() {
            struct_ser.serialize_field("embedding", &self.embedding)?;
        }
        if let Some(v) = self.distance_score.as_ref() {
            struct_ser.serialize_field("distanceScore", v)?;
        }
        if let Some(v) = self.bond_coin.as_ref() {
            struct_ser.serialize_field("bondCoin", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SubmitData {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "target_id",
            "targetId",
            "data_commitment",
            "dataCommitment",
            "data_manifest",
            "dataManifest",
            "model_id",
            "modelId",
            "embedding",
            "distance_score",
            "distanceScore",
            "bond_coin",
            "bondCoin",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TargetId,
            DataCommitment,
            DataManifest,
            ModelId,
            Embedding,
            DistanceScore,
            BondCoin,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "targetId" | "target_id" => Ok(GeneratedField::TargetId),
                            "dataCommitment" | "data_commitment" => Ok(GeneratedField::DataCommitment),
                            "dataManifest" | "data_manifest" => Ok(GeneratedField::DataManifest),
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            "embedding" => Ok(GeneratedField::Embedding),
                            "distanceScore" | "distance_score" => Ok(GeneratedField::DistanceScore),
                            "bondCoin" | "bond_coin" => Ok(GeneratedField::BondCoin),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SubmitData;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SubmitData")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SubmitData, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut target_id__ = None;
                let mut data_commitment__ = None;
                let mut data_manifest__ = None;
                let mut model_id__ = None;
                let mut embedding__ = None;
                let mut distance_score__ = None;
                let mut bond_coin__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::TargetId => {
                            if target_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetId"));
                            }
                            target_id__ = map_.next_value()?;
                        }
                        GeneratedField::DataCommitment => {
                            if data_commitment__.is_some() {
                                return Err(serde::de::Error::duplicate_field("dataCommitment"));
                            }
                            data_commitment__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::DataManifest => {
                            if data_manifest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("dataManifest"));
                            }
                            data_manifest__ = map_.next_value()?;
                        }
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::Embedding => {
                            if embedding__.is_some() {
                                return Err(serde::de::Error::duplicate_field("embedding"));
                            }
                            embedding__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::DistanceScore => {
                            if distance_score__.is_some() {
                                return Err(serde::de::Error::duplicate_field("distanceScore"));
                            }
                            distance_score__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::BondCoin => {
                            if bond_coin__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bondCoin"));
                            }
                            bond_coin__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SubmitData {
                    target_id: target_id__,
                    data_commitment: data_commitment__,
                    data_manifest: data_manifest__,
                    model_id: model_id__,
                    embedding: embedding__.unwrap_or_default(),
                    distance_score: distance_score__,
                    bond_coin: bond_coin__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SubmitData", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SubscribeCheckpointsRequest {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.read_mask.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SubscribeCheckpointsRequest", len)?;
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SubscribeCheckpointsRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "read_mask",
            "readMask",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ReadMask,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "readMask" | "read_mask" => Ok(GeneratedField::ReadMask),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SubscribeCheckpointsRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SubscribeCheckpointsRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SubscribeCheckpointsRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut read_mask__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SubscribeCheckpointsRequest {
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SubscribeCheckpointsRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SubscribeCheckpointsResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.cursor.is_some() {
            len += 1;
        }
        if self.checkpoint.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SubscribeCheckpointsResponse", len)?;
        if let Some(v) = self.cursor.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("cursor", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.checkpoint.as_ref() {
            struct_ser.serialize_field("checkpoint", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SubscribeCheckpointsResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "cursor",
            "checkpoint",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Cursor,
            Checkpoint,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "cursor" => Ok(GeneratedField::Cursor),
                            "checkpoint" => Ok(GeneratedField::Checkpoint),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SubscribeCheckpointsResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SubscribeCheckpointsResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SubscribeCheckpointsResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut cursor__ = None;
                let mut checkpoint__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Cursor => {
                            if cursor__.is_some() {
                                return Err(serde::de::Error::duplicate_field("cursor"));
                            }
                            cursor__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Checkpoint => {
                            if checkpoint__.is_some() {
                                return Err(serde::de::Error::duplicate_field("checkpoint"));
                            }
                            checkpoint__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SubscribeCheckpointsResponse {
                    cursor: cursor__,
                    checkpoint: checkpoint__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SubscribeCheckpointsResponse", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SystemParameters {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.epoch_duration_ms.is_some() {
            len += 1;
        }
        if self.validator_reward_allocation_bps.is_some() {
            len += 1;
        }
        if self.model_min_stake.is_some() {
            len += 1;
        }
        if self.model_architecture_version.is_some() {
            len += 1;
        }
        if self.model_reveal_slash_rate_bps.is_some() {
            len += 1;
        }
        if self.model_tally_slash_rate_bps.is_some() {
            len += 1;
        }
        if self.target_epoch_fee_collection.is_some() {
            len += 1;
        }
        if self.base_fee.is_some() {
            len += 1;
        }
        if self.write_object_fee.is_some() {
            len += 1;
        }
        if self.value_fee_bps.is_some() {
            len += 1;
        }
        if self.min_value_fee_bps.is_some() {
            len += 1;
        }
        if self.max_value_fee_bps.is_some() {
            len += 1;
        }
        if self.fee_adjustment_rate_bps.is_some() {
            len += 1;
        }
        if self.target_models_per_target.is_some() {
            len += 1;
        }
        if self.target_embedding_dim.is_some() {
            len += 1;
        }
        if self.target_initial_distance_threshold.is_some() {
            len += 1;
        }
        if self.target_reward_allocation_bps.is_some() {
            len += 1;
        }
        if self.target_hit_rate_target_bps.is_some() {
            len += 1;
        }
        if self.target_hit_rate_ema_decay_bps.is_some() {
            len += 1;
        }
        if self.target_difficulty_adjustment_rate_bps.is_some() {
            len += 1;
        }
        if self.target_max_distance_threshold.is_some() {
            len += 1;
        }
        if self.target_min_distance_threshold.is_some() {
            len += 1;
        }
        if self.target_initial_targets_per_epoch.is_some() {
            len += 1;
        }
        if self.target_miner_reward_share_bps.is_some() {
            len += 1;
        }
        if self.target_model_reward_share_bps.is_some() {
            len += 1;
        }
        if self.target_claimer_incentive_bps.is_some() {
            len += 1;
        }
        if self.submission_bond_per_byte.is_some() {
            len += 1;
        }
        if self.challenger_bond_per_byte.is_some() {
            len += 1;
        }
        if self.max_submission_data_size.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SystemParameters", len)?;
        if let Some(v) = self.epoch_duration_ms.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epochDurationMs", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.validator_reward_allocation_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("validatorRewardAllocationBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.model_min_stake.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("modelMinStake", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.model_architecture_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("modelArchitectureVersion", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.model_reveal_slash_rate_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("modelRevealSlashRateBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.model_tally_slash_rate_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("modelTallySlashRateBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.target_epoch_fee_collection.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetEpochFeeCollection", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.base_fee.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("baseFee", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.write_object_fee.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("writeObjectFee", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.value_fee_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("valueFeeBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.min_value_fee_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("minValueFeeBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.max_value_fee_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("maxValueFeeBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.fee_adjustment_rate_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("feeAdjustmentRateBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.target_models_per_target.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetModelsPerTarget", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.target_embedding_dim.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetEmbeddingDim", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.target_initial_distance_threshold.as_ref() {
            struct_ser.serialize_field("targetInitialDistanceThreshold", v)?;
        }
        if let Some(v) = self.target_reward_allocation_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetRewardAllocationBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.target_hit_rate_target_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetHitRateTargetBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.target_hit_rate_ema_decay_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetHitRateEmaDecayBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.target_difficulty_adjustment_rate_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetDifficultyAdjustmentRateBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.target_max_distance_threshold.as_ref() {
            struct_ser.serialize_field("targetMaxDistanceThreshold", v)?;
        }
        if let Some(v) = self.target_min_distance_threshold.as_ref() {
            struct_ser.serialize_field("targetMinDistanceThreshold", v)?;
        }
        if let Some(v) = self.target_initial_targets_per_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetInitialTargetsPerEpoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.target_miner_reward_share_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetMinerRewardShareBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.target_model_reward_share_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetModelRewardShareBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.target_claimer_incentive_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetClaimerIncentiveBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.submission_bond_per_byte.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("submissionBondPerByte", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.challenger_bond_per_byte.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("challengerBondPerByte", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.max_submission_data_size.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("maxSubmissionDataSize", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SystemParameters {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epoch_duration_ms",
            "epochDurationMs",
            "validator_reward_allocation_bps",
            "validatorRewardAllocationBps",
            "model_min_stake",
            "modelMinStake",
            "model_architecture_version",
            "modelArchitectureVersion",
            "model_reveal_slash_rate_bps",
            "modelRevealSlashRateBps",
            "model_tally_slash_rate_bps",
            "modelTallySlashRateBps",
            "target_epoch_fee_collection",
            "targetEpochFeeCollection",
            "base_fee",
            "baseFee",
            "write_object_fee",
            "writeObjectFee",
            "value_fee_bps",
            "valueFeeBps",
            "min_value_fee_bps",
            "minValueFeeBps",
            "max_value_fee_bps",
            "maxValueFeeBps",
            "fee_adjustment_rate_bps",
            "feeAdjustmentRateBps",
            "target_models_per_target",
            "targetModelsPerTarget",
            "target_embedding_dim",
            "targetEmbeddingDim",
            "target_initial_distance_threshold",
            "targetInitialDistanceThreshold",
            "target_reward_allocation_bps",
            "targetRewardAllocationBps",
            "target_hit_rate_target_bps",
            "targetHitRateTargetBps",
            "target_hit_rate_ema_decay_bps",
            "targetHitRateEmaDecayBps",
            "target_difficulty_adjustment_rate_bps",
            "targetDifficultyAdjustmentRateBps",
            "target_max_distance_threshold",
            "targetMaxDistanceThreshold",
            "target_min_distance_threshold",
            "targetMinDistanceThreshold",
            "target_initial_targets_per_epoch",
            "targetInitialTargetsPerEpoch",
            "target_miner_reward_share_bps",
            "targetMinerRewardShareBps",
            "target_model_reward_share_bps",
            "targetModelRewardShareBps",
            "target_claimer_incentive_bps",
            "targetClaimerIncentiveBps",
            "submission_bond_per_byte",
            "submissionBondPerByte",
            "challenger_bond_per_byte",
            "challengerBondPerByte",
            "max_submission_data_size",
            "maxSubmissionDataSize",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            EpochDurationMs,
            ValidatorRewardAllocationBps,
            ModelMinStake,
            ModelArchitectureVersion,
            ModelRevealSlashRateBps,
            ModelTallySlashRateBps,
            TargetEpochFeeCollection,
            BaseFee,
            WriteObjectFee,
            ValueFeeBps,
            MinValueFeeBps,
            MaxValueFeeBps,
            FeeAdjustmentRateBps,
            TargetModelsPerTarget,
            TargetEmbeddingDim,
            TargetInitialDistanceThreshold,
            TargetRewardAllocationBps,
            TargetHitRateTargetBps,
            TargetHitRateEmaDecayBps,
            TargetDifficultyAdjustmentRateBps,
            TargetMaxDistanceThreshold,
            TargetMinDistanceThreshold,
            TargetInitialTargetsPerEpoch,
            TargetMinerRewardShareBps,
            TargetModelRewardShareBps,
            TargetClaimerIncentiveBps,
            SubmissionBondPerByte,
            ChallengerBondPerByte,
            MaxSubmissionDataSize,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "epochDurationMs" | "epoch_duration_ms" => Ok(GeneratedField::EpochDurationMs),
                            "validatorRewardAllocationBps" | "validator_reward_allocation_bps" => Ok(GeneratedField::ValidatorRewardAllocationBps),
                            "modelMinStake" | "model_min_stake" => Ok(GeneratedField::ModelMinStake),
                            "modelArchitectureVersion" | "model_architecture_version" => Ok(GeneratedField::ModelArchitectureVersion),
                            "modelRevealSlashRateBps" | "model_reveal_slash_rate_bps" => Ok(GeneratedField::ModelRevealSlashRateBps),
                            "modelTallySlashRateBps" | "model_tally_slash_rate_bps" => Ok(GeneratedField::ModelTallySlashRateBps),
                            "targetEpochFeeCollection" | "target_epoch_fee_collection" => Ok(GeneratedField::TargetEpochFeeCollection),
                            "baseFee" | "base_fee" => Ok(GeneratedField::BaseFee),
                            "writeObjectFee" | "write_object_fee" => Ok(GeneratedField::WriteObjectFee),
                            "valueFeeBps" | "value_fee_bps" => Ok(GeneratedField::ValueFeeBps),
                            "minValueFeeBps" | "min_value_fee_bps" => Ok(GeneratedField::MinValueFeeBps),
                            "maxValueFeeBps" | "max_value_fee_bps" => Ok(GeneratedField::MaxValueFeeBps),
                            "feeAdjustmentRateBps" | "fee_adjustment_rate_bps" => Ok(GeneratedField::FeeAdjustmentRateBps),
                            "targetModelsPerTarget" | "target_models_per_target" => Ok(GeneratedField::TargetModelsPerTarget),
                            "targetEmbeddingDim" | "target_embedding_dim" => Ok(GeneratedField::TargetEmbeddingDim),
                            "targetInitialDistanceThreshold" | "target_initial_distance_threshold" => Ok(GeneratedField::TargetInitialDistanceThreshold),
                            "targetRewardAllocationBps" | "target_reward_allocation_bps" => Ok(GeneratedField::TargetRewardAllocationBps),
                            "targetHitRateTargetBps" | "target_hit_rate_target_bps" => Ok(GeneratedField::TargetHitRateTargetBps),
                            "targetHitRateEmaDecayBps" | "target_hit_rate_ema_decay_bps" => Ok(GeneratedField::TargetHitRateEmaDecayBps),
                            "targetDifficultyAdjustmentRateBps" | "target_difficulty_adjustment_rate_bps" => Ok(GeneratedField::TargetDifficultyAdjustmentRateBps),
                            "targetMaxDistanceThreshold" | "target_max_distance_threshold" => Ok(GeneratedField::TargetMaxDistanceThreshold),
                            "targetMinDistanceThreshold" | "target_min_distance_threshold" => Ok(GeneratedField::TargetMinDistanceThreshold),
                            "targetInitialTargetsPerEpoch" | "target_initial_targets_per_epoch" => Ok(GeneratedField::TargetInitialTargetsPerEpoch),
                            "targetMinerRewardShareBps" | "target_miner_reward_share_bps" => Ok(GeneratedField::TargetMinerRewardShareBps),
                            "targetModelRewardShareBps" | "target_model_reward_share_bps" => Ok(GeneratedField::TargetModelRewardShareBps),
                            "targetClaimerIncentiveBps" | "target_claimer_incentive_bps" => Ok(GeneratedField::TargetClaimerIncentiveBps),
                            "submissionBondPerByte" | "submission_bond_per_byte" => Ok(GeneratedField::SubmissionBondPerByte),
                            "challengerBondPerByte" | "challenger_bond_per_byte" => Ok(GeneratedField::ChallengerBondPerByte),
                            "maxSubmissionDataSize" | "max_submission_data_size" => Ok(GeneratedField::MaxSubmissionDataSize),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SystemParameters;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SystemParameters")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SystemParameters, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epoch_duration_ms__ = None;
                let mut validator_reward_allocation_bps__ = None;
                let mut model_min_stake__ = None;
                let mut model_architecture_version__ = None;
                let mut model_reveal_slash_rate_bps__ = None;
                let mut model_tally_slash_rate_bps__ = None;
                let mut target_epoch_fee_collection__ = None;
                let mut base_fee__ = None;
                let mut write_object_fee__ = None;
                let mut value_fee_bps__ = None;
                let mut min_value_fee_bps__ = None;
                let mut max_value_fee_bps__ = None;
                let mut fee_adjustment_rate_bps__ = None;
                let mut target_models_per_target__ = None;
                let mut target_embedding_dim__ = None;
                let mut target_initial_distance_threshold__ = None;
                let mut target_reward_allocation_bps__ = None;
                let mut target_hit_rate_target_bps__ = None;
                let mut target_hit_rate_ema_decay_bps__ = None;
                let mut target_difficulty_adjustment_rate_bps__ = None;
                let mut target_max_distance_threshold__ = None;
                let mut target_min_distance_threshold__ = None;
                let mut target_initial_targets_per_epoch__ = None;
                let mut target_miner_reward_share_bps__ = None;
                let mut target_model_reward_share_bps__ = None;
                let mut target_claimer_incentive_bps__ = None;
                let mut submission_bond_per_byte__ = None;
                let mut challenger_bond_per_byte__ = None;
                let mut max_submission_data_size__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::EpochDurationMs => {
                            if epoch_duration_ms__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epochDurationMs"));
                            }
                            epoch_duration_ms__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ValidatorRewardAllocationBps => {
                            if validator_reward_allocation_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("validatorRewardAllocationBps"));
                            }
                            validator_reward_allocation_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ModelMinStake => {
                            if model_min_stake__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelMinStake"));
                            }
                            model_min_stake__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ModelArchitectureVersion => {
                            if model_architecture_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelArchitectureVersion"));
                            }
                            model_architecture_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ModelRevealSlashRateBps => {
                            if model_reveal_slash_rate_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelRevealSlashRateBps"));
                            }
                            model_reveal_slash_rate_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ModelTallySlashRateBps => {
                            if model_tally_slash_rate_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelTallySlashRateBps"));
                            }
                            model_tally_slash_rate_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetEpochFeeCollection => {
                            if target_epoch_fee_collection__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetEpochFeeCollection"));
                            }
                            target_epoch_fee_collection__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::BaseFee => {
                            if base_fee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("baseFee"));
                            }
                            base_fee__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::WriteObjectFee => {
                            if write_object_fee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("writeObjectFee"));
                            }
                            write_object_fee__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ValueFeeBps => {
                            if value_fee_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("valueFeeBps"));
                            }
                            value_fee_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::MinValueFeeBps => {
                            if min_value_fee_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("minValueFeeBps"));
                            }
                            min_value_fee_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::MaxValueFeeBps => {
                            if max_value_fee_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("maxValueFeeBps"));
                            }
                            max_value_fee_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::FeeAdjustmentRateBps => {
                            if fee_adjustment_rate_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("feeAdjustmentRateBps"));
                            }
                            fee_adjustment_rate_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetModelsPerTarget => {
                            if target_models_per_target__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetModelsPerTarget"));
                            }
                            target_models_per_target__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetEmbeddingDim => {
                            if target_embedding_dim__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetEmbeddingDim"));
                            }
                            target_embedding_dim__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetInitialDistanceThreshold => {
                            if target_initial_distance_threshold__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetInitialDistanceThreshold"));
                            }
                            target_initial_distance_threshold__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetRewardAllocationBps => {
                            if target_reward_allocation_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetRewardAllocationBps"));
                            }
                            target_reward_allocation_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetHitRateTargetBps => {
                            if target_hit_rate_target_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetHitRateTargetBps"));
                            }
                            target_hit_rate_target_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetHitRateEmaDecayBps => {
                            if target_hit_rate_ema_decay_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetHitRateEmaDecayBps"));
                            }
                            target_hit_rate_ema_decay_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetDifficultyAdjustmentRateBps => {
                            if target_difficulty_adjustment_rate_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetDifficultyAdjustmentRateBps"));
                            }
                            target_difficulty_adjustment_rate_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetMaxDistanceThreshold => {
                            if target_max_distance_threshold__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetMaxDistanceThreshold"));
                            }
                            target_max_distance_threshold__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetMinDistanceThreshold => {
                            if target_min_distance_threshold__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetMinDistanceThreshold"));
                            }
                            target_min_distance_threshold__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetInitialTargetsPerEpoch => {
                            if target_initial_targets_per_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetInitialTargetsPerEpoch"));
                            }
                            target_initial_targets_per_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetMinerRewardShareBps => {
                            if target_miner_reward_share_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetMinerRewardShareBps"));
                            }
                            target_miner_reward_share_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetModelRewardShareBps => {
                            if target_model_reward_share_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetModelRewardShareBps"));
                            }
                            target_model_reward_share_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetClaimerIncentiveBps => {
                            if target_claimer_incentive_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetClaimerIncentiveBps"));
                            }
                            target_claimer_incentive_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::SubmissionBondPerByte => {
                            if submission_bond_per_byte__.is_some() {
                                return Err(serde::de::Error::duplicate_field("submissionBondPerByte"));
                            }
                            submission_bond_per_byte__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ChallengerBondPerByte => {
                            if challenger_bond_per_byte__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challengerBondPerByte"));
                            }
                            challenger_bond_per_byte__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::MaxSubmissionDataSize => {
                            if max_submission_data_size__.is_some() {
                                return Err(serde::de::Error::duplicate_field("maxSubmissionDataSize"));
                            }
                            max_submission_data_size__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SystemParameters {
                    epoch_duration_ms: epoch_duration_ms__,
                    validator_reward_allocation_bps: validator_reward_allocation_bps__,
                    model_min_stake: model_min_stake__,
                    model_architecture_version: model_architecture_version__,
                    model_reveal_slash_rate_bps: model_reveal_slash_rate_bps__,
                    model_tally_slash_rate_bps: model_tally_slash_rate_bps__,
                    target_epoch_fee_collection: target_epoch_fee_collection__,
                    base_fee: base_fee__,
                    write_object_fee: write_object_fee__,
                    value_fee_bps: value_fee_bps__,
                    min_value_fee_bps: min_value_fee_bps__,
                    max_value_fee_bps: max_value_fee_bps__,
                    fee_adjustment_rate_bps: fee_adjustment_rate_bps__,
                    target_models_per_target: target_models_per_target__,
                    target_embedding_dim: target_embedding_dim__,
                    target_initial_distance_threshold: target_initial_distance_threshold__,
                    target_reward_allocation_bps: target_reward_allocation_bps__,
                    target_hit_rate_target_bps: target_hit_rate_target_bps__,
                    target_hit_rate_ema_decay_bps: target_hit_rate_ema_decay_bps__,
                    target_difficulty_adjustment_rate_bps: target_difficulty_adjustment_rate_bps__,
                    target_max_distance_threshold: target_max_distance_threshold__,
                    target_min_distance_threshold: target_min_distance_threshold__,
                    target_initial_targets_per_epoch: target_initial_targets_per_epoch__,
                    target_miner_reward_share_bps: target_miner_reward_share_bps__,
                    target_model_reward_share_bps: target_model_reward_share_bps__,
                    target_claimer_incentive_bps: target_claimer_incentive_bps__,
                    submission_bond_per_byte: submission_bond_per_byte__,
                    challenger_bond_per_byte: challenger_bond_per_byte__,
                    max_submission_data_size: max_submission_data_size__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SystemParameters", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SystemState {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.epoch.is_some() {
            len += 1;
        }
        if self.epoch_start_timestamp_ms.is_some() {
            len += 1;
        }
        if self.protocol_version.is_some() {
            len += 1;
        }
        if self.parameters.is_some() {
            len += 1;
        }
        if self.validators.is_some() {
            len += 1;
        }
        if !self.validator_report_records.is_empty() {
            len += 1;
        }
        if self.emission_pool.is_some() {
            len += 1;
        }
        if self.target_state.is_some() {
            len += 1;
        }
        if self.model_registry.is_some() {
            len += 1;
        }
        if !self.submission_report_records.is_empty() {
            len += 1;
        }
        if self.safe_mode.is_some() {
            len += 1;
        }
        if self.safe_mode_accumulated_fees.is_some() {
            len += 1;
        }
        if self.safe_mode_accumulated_emissions.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SystemState", len)?;
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.epoch_start_timestamp_ms.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epochStartTimestampMs", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.protocol_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("protocolVersion", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.parameters.as_ref() {
            struct_ser.serialize_field("parameters", v)?;
        }
        if let Some(v) = self.validators.as_ref() {
            struct_ser.serialize_field("validators", v)?;
        }
        if !self.validator_report_records.is_empty() {
            struct_ser.serialize_field("validatorReportRecords", &self.validator_report_records)?;
        }
        if let Some(v) = self.emission_pool.as_ref() {
            struct_ser.serialize_field("emissionPool", v)?;
        }
        if let Some(v) = self.target_state.as_ref() {
            struct_ser.serialize_field("targetState", v)?;
        }
        if let Some(v) = self.model_registry.as_ref() {
            struct_ser.serialize_field("modelRegistry", v)?;
        }
        if !self.submission_report_records.is_empty() {
            struct_ser.serialize_field("submissionReportRecords", &self.submission_report_records)?;
        }
        if let Some(v) = self.safe_mode.as_ref() {
            struct_ser.serialize_field("safeMode", v)?;
        }
        if let Some(v) = self.safe_mode_accumulated_fees.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("safeModeAccumulatedFees", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.safe_mode_accumulated_emissions.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("safeModeAccumulatedEmissions", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SystemState {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epoch",
            "epoch_start_timestamp_ms",
            "epochStartTimestampMs",
            "protocol_version",
            "protocolVersion",
            "parameters",
            "validators",
            "validator_report_records",
            "validatorReportRecords",
            "emission_pool",
            "emissionPool",
            "target_state",
            "targetState",
            "model_registry",
            "modelRegistry",
            "submission_report_records",
            "submissionReportRecords",
            "safe_mode",
            "safeMode",
            "safe_mode_accumulated_fees",
            "safeModeAccumulatedFees",
            "safe_mode_accumulated_emissions",
            "safeModeAccumulatedEmissions",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            EpochStartTimestampMs,
            ProtocolVersion,
            Parameters,
            Validators,
            ValidatorReportRecords,
            EmissionPool,
            TargetState,
            ModelRegistry,
            SubmissionReportRecords,
            SafeMode,
            SafeModeAccumulatedFees,
            SafeModeAccumulatedEmissions,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "epoch" => Ok(GeneratedField::Epoch),
                            "epochStartTimestampMs" | "epoch_start_timestamp_ms" => Ok(GeneratedField::EpochStartTimestampMs),
                            "protocolVersion" | "protocol_version" => Ok(GeneratedField::ProtocolVersion),
                            "parameters" => Ok(GeneratedField::Parameters),
                            "validators" => Ok(GeneratedField::Validators),
                            "validatorReportRecords" | "validator_report_records" => Ok(GeneratedField::ValidatorReportRecords),
                            "emissionPool" | "emission_pool" => Ok(GeneratedField::EmissionPool),
                            "targetState" | "target_state" => Ok(GeneratedField::TargetState),
                            "modelRegistry" | "model_registry" => Ok(GeneratedField::ModelRegistry),
                            "submissionReportRecords" | "submission_report_records" => Ok(GeneratedField::SubmissionReportRecords),
                            "safeMode" | "safe_mode" => Ok(GeneratedField::SafeMode),
                            "safeModeAccumulatedFees" | "safe_mode_accumulated_fees" => Ok(GeneratedField::SafeModeAccumulatedFees),
                            "safeModeAccumulatedEmissions" | "safe_mode_accumulated_emissions" => Ok(GeneratedField::SafeModeAccumulatedEmissions),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SystemState;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SystemState")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SystemState, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epoch__ = None;
                let mut epoch_start_timestamp_ms__ = None;
                let mut protocol_version__ = None;
                let mut parameters__ = None;
                let mut validators__ = None;
                let mut validator_report_records__ = None;
                let mut emission_pool__ = None;
                let mut target_state__ = None;
                let mut model_registry__ = None;
                let mut submission_report_records__ = None;
                let mut safe_mode__ = None;
                let mut safe_mode_accumulated_fees__ = None;
                let mut safe_mode_accumulated_emissions__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::EpochStartTimestampMs => {
                            if epoch_start_timestamp_ms__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epochStartTimestampMs"));
                            }
                            epoch_start_timestamp_ms__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ProtocolVersion => {
                            if protocol_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("protocolVersion"));
                            }
                            protocol_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Parameters => {
                            if parameters__.is_some() {
                                return Err(serde::de::Error::duplicate_field("parameters"));
                            }
                            parameters__ = map_.next_value()?;
                        }
                        GeneratedField::Validators => {
                            if validators__.is_some() {
                                return Err(serde::de::Error::duplicate_field("validators"));
                            }
                            validators__ = map_.next_value()?;
                        }
                        GeneratedField::ValidatorReportRecords => {
                            if validator_report_records__.is_some() {
                                return Err(serde::de::Error::duplicate_field("validatorReportRecords"));
                            }
                            validator_report_records__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::EmissionPool => {
                            if emission_pool__.is_some() {
                                return Err(serde::de::Error::duplicate_field("emissionPool"));
                            }
                            emission_pool__ = map_.next_value()?;
                        }
                        GeneratedField::TargetState => {
                            if target_state__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetState"));
                            }
                            target_state__ = map_.next_value()?;
                        }
                        GeneratedField::ModelRegistry => {
                            if model_registry__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelRegistry"));
                            }
                            model_registry__ = map_.next_value()?;
                        }
                        GeneratedField::SubmissionReportRecords => {
                            if submission_report_records__.is_some() {
                                return Err(serde::de::Error::duplicate_field("submissionReportRecords"));
                            }
                            submission_report_records__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::SafeMode => {
                            if safe_mode__.is_some() {
                                return Err(serde::de::Error::duplicate_field("safeMode"));
                            }
                            safe_mode__ = map_.next_value()?;
                        }
                        GeneratedField::SafeModeAccumulatedFees => {
                            if safe_mode_accumulated_fees__.is_some() {
                                return Err(serde::de::Error::duplicate_field("safeModeAccumulatedFees"));
                            }
                            safe_mode_accumulated_fees__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::SafeModeAccumulatedEmissions => {
                            if safe_mode_accumulated_emissions__.is_some() {
                                return Err(serde::de::Error::duplicate_field("safeModeAccumulatedEmissions"));
                            }
                            safe_mode_accumulated_emissions__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SystemState {
                    epoch: epoch__,
                    epoch_start_timestamp_ms: epoch_start_timestamp_ms__,
                    protocol_version: protocol_version__,
                    parameters: parameters__,
                    validators: validators__,
                    validator_report_records: validator_report_records__.unwrap_or_default(),
                    emission_pool: emission_pool__,
                    target_state: target_state__,
                    model_registry: model_registry__,
                    submission_report_records: submission_report_records__.unwrap_or_default(),
                    safe_mode: safe_mode__,
                    safe_mode_accumulated_fees: safe_mode_accumulated_fees__,
                    safe_mode_accumulated_emissions: safe_mode_accumulated_emissions__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SystemState", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Target {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.id.is_some() {
            len += 1;
        }
        if !self.embedding.is_empty() {
            len += 1;
        }
        if !self.model_ids.is_empty() {
            len += 1;
        }
        if self.distance_threshold.is_some() {
            len += 1;
        }
        if self.reward_pool.is_some() {
            len += 1;
        }
        if self.generation_epoch.is_some() {
            len += 1;
        }
        if self.status.is_some() {
            len += 1;
        }
        if self.fill_epoch.is_some() {
            len += 1;
        }
        if self.miner.is_some() {
            len += 1;
        }
        if self.winning_model_id.is_some() {
            len += 1;
        }
        if self.winning_model_owner.is_some() {
            len += 1;
        }
        if self.bond_amount.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Target", len)?;
        if let Some(v) = self.id.as_ref() {
            struct_ser.serialize_field("id", v)?;
        }
        if !self.embedding.is_empty() {
            struct_ser.serialize_field("embedding", &self.embedding)?;
        }
        if !self.model_ids.is_empty() {
            struct_ser.serialize_field("modelIds", &self.model_ids)?;
        }
        if let Some(v) = self.distance_threshold.as_ref() {
            struct_ser.serialize_field("distanceThreshold", v)?;
        }
        if let Some(v) = self.reward_pool.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("rewardPool", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.generation_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("generationEpoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.status.as_ref() {
            struct_ser.serialize_field("status", v)?;
        }
        if let Some(v) = self.fill_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("fillEpoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.miner.as_ref() {
            struct_ser.serialize_field("miner", v)?;
        }
        if let Some(v) = self.winning_model_id.as_ref() {
            struct_ser.serialize_field("winningModelId", v)?;
        }
        if let Some(v) = self.winning_model_owner.as_ref() {
            struct_ser.serialize_field("winningModelOwner", v)?;
        }
        if let Some(v) = self.bond_amount.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("bondAmount", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Target {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "id",
            "embedding",
            "model_ids",
            "modelIds",
            "distance_threshold",
            "distanceThreshold",
            "reward_pool",
            "rewardPool",
            "generation_epoch",
            "generationEpoch",
            "status",
            "fill_epoch",
            "fillEpoch",
            "miner",
            "winning_model_id",
            "winningModelId",
            "winning_model_owner",
            "winningModelOwner",
            "bond_amount",
            "bondAmount",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Id,
            Embedding,
            ModelIds,
            DistanceThreshold,
            RewardPool,
            GenerationEpoch,
            Status,
            FillEpoch,
            Miner,
            WinningModelId,
            WinningModelOwner,
            BondAmount,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "id" => Ok(GeneratedField::Id),
                            "embedding" => Ok(GeneratedField::Embedding),
                            "modelIds" | "model_ids" => Ok(GeneratedField::ModelIds),
                            "distanceThreshold" | "distance_threshold" => Ok(GeneratedField::DistanceThreshold),
                            "rewardPool" | "reward_pool" => Ok(GeneratedField::RewardPool),
                            "generationEpoch" | "generation_epoch" => Ok(GeneratedField::GenerationEpoch),
                            "status" => Ok(GeneratedField::Status),
                            "fillEpoch" | "fill_epoch" => Ok(GeneratedField::FillEpoch),
                            "miner" => Ok(GeneratedField::Miner),
                            "winningModelId" | "winning_model_id" => Ok(GeneratedField::WinningModelId),
                            "winningModelOwner" | "winning_model_owner" => Ok(GeneratedField::WinningModelOwner),
                            "bondAmount" | "bond_amount" => Ok(GeneratedField::BondAmount),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Target;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Target")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Target, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut id__ = None;
                let mut embedding__ = None;
                let mut model_ids__ = None;
                let mut distance_threshold__ = None;
                let mut reward_pool__ = None;
                let mut generation_epoch__ = None;
                let mut status__ = None;
                let mut fill_epoch__ = None;
                let mut miner__ = None;
                let mut winning_model_id__ = None;
                let mut winning_model_owner__ = None;
                let mut bond_amount__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Id => {
                            if id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("id"));
                            }
                            id__ = map_.next_value()?;
                        }
                        GeneratedField::Embedding => {
                            if embedding__.is_some() {
                                return Err(serde::de::Error::duplicate_field("embedding"));
                            }
                            embedding__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::ModelIds => {
                            if model_ids__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelIds"));
                            }
                            model_ids__ = Some(map_.next_value()?);
                        }
                        GeneratedField::DistanceThreshold => {
                            if distance_threshold__.is_some() {
                                return Err(serde::de::Error::duplicate_field("distanceThreshold"));
                            }
                            distance_threshold__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::RewardPool => {
                            if reward_pool__.is_some() {
                                return Err(serde::de::Error::duplicate_field("rewardPool"));
                            }
                            reward_pool__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::GenerationEpoch => {
                            if generation_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("generationEpoch"));
                            }
                            generation_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Status => {
                            if status__.is_some() {
                                return Err(serde::de::Error::duplicate_field("status"));
                            }
                            status__ = map_.next_value()?;
                        }
                        GeneratedField::FillEpoch => {
                            if fill_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("fillEpoch"));
                            }
                            fill_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Miner => {
                            if miner__.is_some() {
                                return Err(serde::de::Error::duplicate_field("miner"));
                            }
                            miner__ = map_.next_value()?;
                        }
                        GeneratedField::WinningModelId => {
                            if winning_model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("winningModelId"));
                            }
                            winning_model_id__ = map_.next_value()?;
                        }
                        GeneratedField::WinningModelOwner => {
                            if winning_model_owner__.is_some() {
                                return Err(serde::de::Error::duplicate_field("winningModelOwner"));
                            }
                            winning_model_owner__ = map_.next_value()?;
                        }
                        GeneratedField::BondAmount => {
                            if bond_amount__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bondAmount"));
                            }
                            bond_amount__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Target {
                    id: id__,
                    embedding: embedding__.unwrap_or_default(),
                    model_ids: model_ids__.unwrap_or_default(),
                    distance_threshold: distance_threshold__,
                    reward_pool: reward_pool__,
                    generation_epoch: generation_epoch__,
                    status: status__,
                    fill_epoch: fill_epoch__,
                    miner: miner__,
                    winning_model_id: winning_model_id__,
                    winning_model_owner: winning_model_owner__,
                    bond_amount: bond_amount__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Target", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TargetState {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.distance_threshold.is_some() {
            len += 1;
        }
        if self.targets_generated_this_epoch.is_some() {
            len += 1;
        }
        if self.hits_this_epoch.is_some() {
            len += 1;
        }
        if self.hit_rate_ema_bps.is_some() {
            len += 1;
        }
        if self.reward_per_target.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TargetState", len)?;
        if let Some(v) = self.distance_threshold.as_ref() {
            struct_ser.serialize_field("distanceThreshold", v)?;
        }
        if let Some(v) = self.targets_generated_this_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("targetsGeneratedThisEpoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.hits_this_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("hitsThisEpoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.hit_rate_ema_bps.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("hitRateEmaBps", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.reward_per_target.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("rewardPerTarget", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TargetState {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "distance_threshold",
            "distanceThreshold",
            "targets_generated_this_epoch",
            "targetsGeneratedThisEpoch",
            "hits_this_epoch",
            "hitsThisEpoch",
            "hit_rate_ema_bps",
            "hitRateEmaBps",
            "reward_per_target",
            "rewardPerTarget",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            DistanceThreshold,
            TargetsGeneratedThisEpoch,
            HitsThisEpoch,
            HitRateEmaBps,
            RewardPerTarget,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "distanceThreshold" | "distance_threshold" => Ok(GeneratedField::DistanceThreshold),
                            "targetsGeneratedThisEpoch" | "targets_generated_this_epoch" => Ok(GeneratedField::TargetsGeneratedThisEpoch),
                            "hitsThisEpoch" | "hits_this_epoch" => Ok(GeneratedField::HitsThisEpoch),
                            "hitRateEmaBps" | "hit_rate_ema_bps" => Ok(GeneratedField::HitRateEmaBps),
                            "rewardPerTarget" | "reward_per_target" => Ok(GeneratedField::RewardPerTarget),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = TargetState;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TargetState")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TargetState, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut distance_threshold__ = None;
                let mut targets_generated_this_epoch__ = None;
                let mut hits_this_epoch__ = None;
                let mut hit_rate_ema_bps__ = None;
                let mut reward_per_target__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::DistanceThreshold => {
                            if distance_threshold__.is_some() {
                                return Err(serde::de::Error::duplicate_field("distanceThreshold"));
                            }
                            distance_threshold__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TargetsGeneratedThisEpoch => {
                            if targets_generated_this_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetsGeneratedThisEpoch"));
                            }
                            targets_generated_this_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::HitsThisEpoch => {
                            if hits_this_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("hitsThisEpoch"));
                            }
                            hits_this_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::HitRateEmaBps => {
                            if hit_rate_ema_bps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("hitRateEmaBps"));
                            }
                            hit_rate_ema_bps__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::RewardPerTarget => {
                            if reward_per_target__.is_some() {
                                return Err(serde::de::Error::duplicate_field("rewardPerTarget"));
                            }
                            reward_per_target__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TargetState {
                    distance_threshold: distance_threshold__,
                    targets_generated_this_epoch: targets_generated_this_epoch__,
                    hits_this_epoch: hits_this_epoch__,
                    hit_rate_ema_bps: hit_rate_ema_bps__,
                    reward_per_target: reward_per_target__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TargetState", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Transaction {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.digest.is_some() {
            len += 1;
        }
        if self.kind.is_some() {
            len += 1;
        }
        if self.sender.is_some() {
            len += 1;
        }
        if !self.gas_payment.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Transaction", len)?;
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.kind.as_ref() {
            struct_ser.serialize_field("kind", v)?;
        }
        if let Some(v) = self.sender.as_ref() {
            struct_ser.serialize_field("sender", v)?;
        }
        if !self.gas_payment.is_empty() {
            struct_ser.serialize_field("gasPayment", &self.gas_payment)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Transaction {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "digest",
            "kind",
            "sender",
            "gas_payment",
            "gasPayment",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Digest,
            Kind,
            Sender,
            GasPayment,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "digest" => Ok(GeneratedField::Digest),
                            "kind" => Ok(GeneratedField::Kind),
                            "sender" => Ok(GeneratedField::Sender),
                            "gasPayment" | "gas_payment" => Ok(GeneratedField::GasPayment),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Transaction;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Transaction")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Transaction, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut digest__ = None;
                let mut kind__ = None;
                let mut sender__ = None;
                let mut gas_payment__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value()?;
                        }
                        GeneratedField::Sender => {
                            if sender__.is_some() {
                                return Err(serde::de::Error::duplicate_field("sender"));
                            }
                            sender__ = map_.next_value()?;
                        }
                        GeneratedField::GasPayment => {
                            if gas_payment__.is_some() {
                                return Err(serde::de::Error::duplicate_field("gasPayment"));
                            }
                            gas_payment__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Transaction {
                    digest: digest__,
                    kind: kind__,
                    sender: sender__,
                    gas_payment: gas_payment__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Transaction", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TransactionEffects {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.status.is_some() {
            len += 1;
        }
        if self.epoch.is_some() {
            len += 1;
        }
        if self.fee.is_some() {
            len += 1;
        }
        if self.transaction_digest.is_some() {
            len += 1;
        }
        if self.gas_object_index.is_some() {
            len += 1;
        }
        if !self.dependencies.is_empty() {
            len += 1;
        }
        if self.lamport_version.is_some() {
            len += 1;
        }
        if !self.changed_objects.is_empty() {
            len += 1;
        }
        if !self.unchanged_shared_objects.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TransactionEffects", len)?;
        if let Some(v) = self.status.as_ref() {
            struct_ser.serialize_field("status", v)?;
        }
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.fee.as_ref() {
            struct_ser.serialize_field("fee", v)?;
        }
        if let Some(v) = self.transaction_digest.as_ref() {
            struct_ser.serialize_field("transactionDigest", v)?;
        }
        if let Some(v) = self.gas_object_index.as_ref() {
            struct_ser.serialize_field("gasObjectIndex", v)?;
        }
        if !self.dependencies.is_empty() {
            struct_ser.serialize_field("dependencies", &self.dependencies)?;
        }
        if let Some(v) = self.lamport_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("lamportVersion", ToString::to_string(&v).as_str())?;
        }
        if !self.changed_objects.is_empty() {
            struct_ser.serialize_field("changedObjects", &self.changed_objects)?;
        }
        if !self.unchanged_shared_objects.is_empty() {
            struct_ser.serialize_field("unchangedSharedObjects", &self.unchanged_shared_objects)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TransactionEffects {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "status",
            "epoch",
            "fee",
            "transaction_digest",
            "transactionDigest",
            "gas_object_index",
            "gasObjectIndex",
            "dependencies",
            "lamport_version",
            "lamportVersion",
            "changed_objects",
            "changedObjects",
            "unchanged_shared_objects",
            "unchangedSharedObjects",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Status,
            Epoch,
            Fee,
            TransactionDigest,
            GasObjectIndex,
            Dependencies,
            LamportVersion,
            ChangedObjects,
            UnchangedSharedObjects,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "status" => Ok(GeneratedField::Status),
                            "epoch" => Ok(GeneratedField::Epoch),
                            "fee" => Ok(GeneratedField::Fee),
                            "transactionDigest" | "transaction_digest" => Ok(GeneratedField::TransactionDigest),
                            "gasObjectIndex" | "gas_object_index" => Ok(GeneratedField::GasObjectIndex),
                            "dependencies" => Ok(GeneratedField::Dependencies),
                            "lamportVersion" | "lamport_version" => Ok(GeneratedField::LamportVersion),
                            "changedObjects" | "changed_objects" => Ok(GeneratedField::ChangedObjects),
                            "unchangedSharedObjects" | "unchanged_shared_objects" => Ok(GeneratedField::UnchangedSharedObjects),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = TransactionEffects;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TransactionEffects")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TransactionEffects, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut status__ = None;
                let mut epoch__ = None;
                let mut fee__ = None;
                let mut transaction_digest__ = None;
                let mut gas_object_index__ = None;
                let mut dependencies__ = None;
                let mut lamport_version__ = None;
                let mut changed_objects__ = None;
                let mut unchanged_shared_objects__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Status => {
                            if status__.is_some() {
                                return Err(serde::de::Error::duplicate_field("status"));
                            }
                            status__ = map_.next_value()?;
                        }
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Fee => {
                            if fee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("fee"));
                            }
                            fee__ = map_.next_value()?;
                        }
                        GeneratedField::TransactionDigest => {
                            if transaction_digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transactionDigest"));
                            }
                            transaction_digest__ = map_.next_value()?;
                        }
                        GeneratedField::GasObjectIndex => {
                            if gas_object_index__.is_some() {
                                return Err(serde::de::Error::duplicate_field("gasObjectIndex"));
                            }
                            gas_object_index__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Dependencies => {
                            if dependencies__.is_some() {
                                return Err(serde::de::Error::duplicate_field("dependencies"));
                            }
                            dependencies__ = Some(map_.next_value()?);
                        }
                        GeneratedField::LamportVersion => {
                            if lamport_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("lamportVersion"));
                            }
                            lamport_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ChangedObjects => {
                            if changed_objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("changedObjects"));
                            }
                            changed_objects__ = Some(map_.next_value()?);
                        }
                        GeneratedField::UnchangedSharedObjects => {
                            if unchanged_shared_objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("unchangedSharedObjects"));
                            }
                            unchanged_shared_objects__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TransactionEffects {
                    status: status__,
                    epoch: epoch__,
                    fee: fee__,
                    transaction_digest: transaction_digest__,
                    gas_object_index: gas_object_index__,
                    dependencies: dependencies__.unwrap_or_default(),
                    lamport_version: lamport_version__,
                    changed_objects: changed_objects__.unwrap_or_default(),
                    unchanged_shared_objects: unchanged_shared_objects__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TransactionEffects", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TransactionFee {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.base_fee.is_some() {
            len += 1;
        }
        if self.operation_fee.is_some() {
            len += 1;
        }
        if self.value_fee.is_some() {
            len += 1;
        }
        if self.total_fee.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TransactionFee", len)?;
        if let Some(v) = self.base_fee.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("baseFee", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.operation_fee.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("operationFee", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.value_fee.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("valueFee", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.total_fee.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("totalFee", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TransactionFee {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "base_fee",
            "baseFee",
            "operation_fee",
            "operationFee",
            "value_fee",
            "valueFee",
            "total_fee",
            "totalFee",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            BaseFee,
            OperationFee,
            ValueFee,
            TotalFee,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "baseFee" | "base_fee" => Ok(GeneratedField::BaseFee),
                            "operationFee" | "operation_fee" => Ok(GeneratedField::OperationFee),
                            "valueFee" | "value_fee" => Ok(GeneratedField::ValueFee),
                            "totalFee" | "total_fee" => Ok(GeneratedField::TotalFee),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = TransactionFee;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TransactionFee")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TransactionFee, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut base_fee__ = None;
                let mut operation_fee__ = None;
                let mut value_fee__ = None;
                let mut total_fee__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::BaseFee => {
                            if base_fee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("baseFee"));
                            }
                            base_fee__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::OperationFee => {
                            if operation_fee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("operationFee"));
                            }
                            operation_fee__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ValueFee => {
                            if value_fee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("valueFee"));
                            }
                            value_fee__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TotalFee => {
                            if total_fee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("totalFee"));
                            }
                            total_fee__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TransactionFee {
                    base_fee: base_fee__,
                    operation_fee: operation_fee__,
                    value_fee: value_fee__,
                    total_fee: total_fee__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TransactionFee", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TransactionKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.kind.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TransactionKind", len)?;
        if let Some(v) = self.kind.as_ref() {
            match v {
                transaction_kind::Kind::Genesis(v) => {
                    struct_ser.serialize_field("genesis", v)?;
                }
                transaction_kind::Kind::ConsensusCommitPrologue(v) => {
                    struct_ser.serialize_field("consensusCommitPrologue", v)?;
                }
                transaction_kind::Kind::ChangeEpoch(v) => {
                    struct_ser.serialize_field("changeEpoch", v)?;
                }
                transaction_kind::Kind::AddValidator(v) => {
                    struct_ser.serialize_field("addValidator", v)?;
                }
                transaction_kind::Kind::RemoveValidator(v) => {
                    struct_ser.serialize_field("removeValidator", v)?;
                }
                transaction_kind::Kind::ReportValidator(v) => {
                    struct_ser.serialize_field("reportValidator", v)?;
                }
                transaction_kind::Kind::UndoReportValidator(v) => {
                    struct_ser.serialize_field("undoReportValidator", v)?;
                }
                transaction_kind::Kind::UpdateValidatorMetadata(v) => {
                    struct_ser.serialize_field("updateValidatorMetadata", v)?;
                }
                transaction_kind::Kind::SetCommissionRate(v) => {
                    struct_ser.serialize_field("setCommissionRate", v)?;
                }
                transaction_kind::Kind::TransferCoin(v) => {
                    struct_ser.serialize_field("transferCoin", v)?;
                }
                transaction_kind::Kind::PayCoins(v) => {
                    struct_ser.serialize_field("payCoins", v)?;
                }
                transaction_kind::Kind::TransferObjects(v) => {
                    struct_ser.serialize_field("transferObjects", v)?;
                }
                transaction_kind::Kind::AddStake(v) => {
                    struct_ser.serialize_field("addStake", v)?;
                }
                transaction_kind::Kind::WithdrawStake(v) => {
                    struct_ser.serialize_field("withdrawStake", v)?;
                }
                transaction_kind::Kind::CommitModel(v) => {
                    struct_ser.serialize_field("commitModel", v)?;
                }
                transaction_kind::Kind::RevealModel(v) => {
                    struct_ser.serialize_field("revealModel", v)?;
                }
                transaction_kind::Kind::CommitModelUpdate(v) => {
                    struct_ser.serialize_field("commitModelUpdate", v)?;
                }
                transaction_kind::Kind::RevealModelUpdate(v) => {
                    struct_ser.serialize_field("revealModelUpdate", v)?;
                }
                transaction_kind::Kind::AddStakeToModel(v) => {
                    struct_ser.serialize_field("addStakeToModel", v)?;
                }
                transaction_kind::Kind::SetModelCommissionRate(v) => {
                    struct_ser.serialize_field("setModelCommissionRate", v)?;
                }
                transaction_kind::Kind::DeactivateModel(v) => {
                    struct_ser.serialize_field("deactivateModel", v)?;
                }
                transaction_kind::Kind::ReportModel(v) => {
                    struct_ser.serialize_field("reportModel", v)?;
                }
                transaction_kind::Kind::UndoReportModel(v) => {
                    struct_ser.serialize_field("undoReportModel", v)?;
                }
                transaction_kind::Kind::SubmitData(v) => {
                    struct_ser.serialize_field("submitData", v)?;
                }
                transaction_kind::Kind::ClaimRewards(v) => {
                    struct_ser.serialize_field("claimRewards", v)?;
                }
                transaction_kind::Kind::ReportSubmission(v) => {
                    struct_ser.serialize_field("reportSubmission", v)?;
                }
                transaction_kind::Kind::UndoReportSubmission(v) => {
                    struct_ser.serialize_field("undoReportSubmission", v)?;
                }
                transaction_kind::Kind::InitiateChallenge(v) => {
                    struct_ser.serialize_field("initiateChallenge", v)?;
                }
                transaction_kind::Kind::ReportChallenge(v) => {
                    struct_ser.serialize_field("reportChallenge", v)?;
                }
                transaction_kind::Kind::UndoReportChallenge(v) => {
                    struct_ser.serialize_field("undoReportChallenge", v)?;
                }
                transaction_kind::Kind::ClaimChallengeBond(v) => {
                    struct_ser.serialize_field("claimChallengeBond", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TransactionKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "genesis",
            "consensus_commit_prologue",
            "consensusCommitPrologue",
            "change_epoch",
            "changeEpoch",
            "add_validator",
            "addValidator",
            "remove_validator",
            "removeValidator",
            "report_validator",
            "reportValidator",
            "undo_report_validator",
            "undoReportValidator",
            "update_validator_metadata",
            "updateValidatorMetadata",
            "set_commission_rate",
            "setCommissionRate",
            "transfer_coin",
            "transferCoin",
            "pay_coins",
            "payCoins",
            "transfer_objects",
            "transferObjects",
            "add_stake",
            "addStake",
            "withdraw_stake",
            "withdrawStake",
            "commit_model",
            "commitModel",
            "reveal_model",
            "revealModel",
            "commit_model_update",
            "commitModelUpdate",
            "reveal_model_update",
            "revealModelUpdate",
            "add_stake_to_model",
            "addStakeToModel",
            "set_model_commission_rate",
            "setModelCommissionRate",
            "deactivate_model",
            "deactivateModel",
            "report_model",
            "reportModel",
            "undo_report_model",
            "undoReportModel",
            "submit_data",
            "submitData",
            "claim_rewards",
            "claimRewards",
            "report_submission",
            "reportSubmission",
            "undo_report_submission",
            "undoReportSubmission",
            "initiate_challenge",
            "initiateChallenge",
            "report_challenge",
            "reportChallenge",
            "undo_report_challenge",
            "undoReportChallenge",
            "claim_challenge_bond",
            "claimChallengeBond",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Genesis,
            ConsensusCommitPrologue,
            ChangeEpoch,
            AddValidator,
            RemoveValidator,
            ReportValidator,
            UndoReportValidator,
            UpdateValidatorMetadata,
            SetCommissionRate,
            TransferCoin,
            PayCoins,
            TransferObjects,
            AddStake,
            WithdrawStake,
            CommitModel,
            RevealModel,
            CommitModelUpdate,
            RevealModelUpdate,
            AddStakeToModel,
            SetModelCommissionRate,
            DeactivateModel,
            ReportModel,
            UndoReportModel,
            SubmitData,
            ClaimRewards,
            ReportSubmission,
            UndoReportSubmission,
            InitiateChallenge,
            ReportChallenge,
            UndoReportChallenge,
            ClaimChallengeBond,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "genesis" => Ok(GeneratedField::Genesis),
                            "consensusCommitPrologue" | "consensus_commit_prologue" => Ok(GeneratedField::ConsensusCommitPrologue),
                            "changeEpoch" | "change_epoch" => Ok(GeneratedField::ChangeEpoch),
                            "addValidator" | "add_validator" => Ok(GeneratedField::AddValidator),
                            "removeValidator" | "remove_validator" => Ok(GeneratedField::RemoveValidator),
                            "reportValidator" | "report_validator" => Ok(GeneratedField::ReportValidator),
                            "undoReportValidator" | "undo_report_validator" => Ok(GeneratedField::UndoReportValidator),
                            "updateValidatorMetadata" | "update_validator_metadata" => Ok(GeneratedField::UpdateValidatorMetadata),
                            "setCommissionRate" | "set_commission_rate" => Ok(GeneratedField::SetCommissionRate),
                            "transferCoin" | "transfer_coin" => Ok(GeneratedField::TransferCoin),
                            "payCoins" | "pay_coins" => Ok(GeneratedField::PayCoins),
                            "transferObjects" | "transfer_objects" => Ok(GeneratedField::TransferObjects),
                            "addStake" | "add_stake" => Ok(GeneratedField::AddStake),
                            "withdrawStake" | "withdraw_stake" => Ok(GeneratedField::WithdrawStake),
                            "commitModel" | "commit_model" => Ok(GeneratedField::CommitModel),
                            "revealModel" | "reveal_model" => Ok(GeneratedField::RevealModel),
                            "commitModelUpdate" | "commit_model_update" => Ok(GeneratedField::CommitModelUpdate),
                            "revealModelUpdate" | "reveal_model_update" => Ok(GeneratedField::RevealModelUpdate),
                            "addStakeToModel" | "add_stake_to_model" => Ok(GeneratedField::AddStakeToModel),
                            "setModelCommissionRate" | "set_model_commission_rate" => Ok(GeneratedField::SetModelCommissionRate),
                            "deactivateModel" | "deactivate_model" => Ok(GeneratedField::DeactivateModel),
                            "reportModel" | "report_model" => Ok(GeneratedField::ReportModel),
                            "undoReportModel" | "undo_report_model" => Ok(GeneratedField::UndoReportModel),
                            "submitData" | "submit_data" => Ok(GeneratedField::SubmitData),
                            "claimRewards" | "claim_rewards" => Ok(GeneratedField::ClaimRewards),
                            "reportSubmission" | "report_submission" => Ok(GeneratedField::ReportSubmission),
                            "undoReportSubmission" | "undo_report_submission" => Ok(GeneratedField::UndoReportSubmission),
                            "initiateChallenge" | "initiate_challenge" => Ok(GeneratedField::InitiateChallenge),
                            "reportChallenge" | "report_challenge" => Ok(GeneratedField::ReportChallenge),
                            "undoReportChallenge" | "undo_report_challenge" => Ok(GeneratedField::UndoReportChallenge),
                            "claimChallengeBond" | "claim_challenge_bond" => Ok(GeneratedField::ClaimChallengeBond),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = TransactionKind;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TransactionKind")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TransactionKind, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Genesis => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("genesis"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::Genesis)
;
                        }
                        GeneratedField::ConsensusCommitPrologue => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("consensusCommitPrologue"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ConsensusCommitPrologue)
;
                        }
                        GeneratedField::ChangeEpoch => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("changeEpoch"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ChangeEpoch)
;
                        }
                        GeneratedField::AddValidator => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("addValidator"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::AddValidator)
;
                        }
                        GeneratedField::RemoveValidator => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("removeValidator"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::RemoveValidator)
;
                        }
                        GeneratedField::ReportValidator => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reportValidator"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ReportValidator)
;
                        }
                        GeneratedField::UndoReportValidator => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("undoReportValidator"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::UndoReportValidator)
;
                        }
                        GeneratedField::UpdateValidatorMetadata => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("updateValidatorMetadata"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::UpdateValidatorMetadata)
;
                        }
                        GeneratedField::SetCommissionRate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("setCommissionRate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::SetCommissionRate)
;
                        }
                        GeneratedField::TransferCoin => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transferCoin"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::TransferCoin)
;
                        }
                        GeneratedField::PayCoins => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("payCoins"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::PayCoins)
;
                        }
                        GeneratedField::TransferObjects => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transferObjects"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::TransferObjects)
;
                        }
                        GeneratedField::AddStake => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("addStake"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::AddStake)
;
                        }
                        GeneratedField::WithdrawStake => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("withdrawStake"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::WithdrawStake)
;
                        }
                        GeneratedField::CommitModel => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commitModel"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::CommitModel)
;
                        }
                        GeneratedField::RevealModel => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("revealModel"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::RevealModel)
;
                        }
                        GeneratedField::CommitModelUpdate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commitModelUpdate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::CommitModelUpdate)
;
                        }
                        GeneratedField::RevealModelUpdate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("revealModelUpdate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::RevealModelUpdate)
;
                        }
                        GeneratedField::AddStakeToModel => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("addStakeToModel"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::AddStakeToModel)
;
                        }
                        GeneratedField::SetModelCommissionRate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("setModelCommissionRate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::SetModelCommissionRate)
;
                        }
                        GeneratedField::DeactivateModel => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("deactivateModel"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::DeactivateModel)
;
                        }
                        GeneratedField::ReportModel => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reportModel"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ReportModel)
;
                        }
                        GeneratedField::UndoReportModel => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("undoReportModel"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::UndoReportModel)
;
                        }
                        GeneratedField::SubmitData => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("submitData"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::SubmitData)
;
                        }
                        GeneratedField::ClaimRewards => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("claimRewards"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ClaimRewards)
;
                        }
                        GeneratedField::ReportSubmission => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reportSubmission"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ReportSubmission)
;
                        }
                        GeneratedField::UndoReportSubmission => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("undoReportSubmission"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::UndoReportSubmission)
;
                        }
                        GeneratedField::InitiateChallenge => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("initiateChallenge"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::InitiateChallenge)
;
                        }
                        GeneratedField::ReportChallenge => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reportChallenge"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ReportChallenge)
;
                        }
                        GeneratedField::UndoReportChallenge => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("undoReportChallenge"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::UndoReportChallenge)
;
                        }
                        GeneratedField::ClaimChallengeBond => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("claimChallengeBond"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ClaimChallengeBond)
;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TransactionKind {
                    kind: kind__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TransactionKind", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TransferCoin {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.coin.is_some() {
            len += 1;
        }
        if self.amount.is_some() {
            len += 1;
        }
        if self.recipient.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TransferCoin", len)?;
        if let Some(v) = self.coin.as_ref() {
            struct_ser.serialize_field("coin", v)?;
        }
        if let Some(v) = self.amount.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("amount", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.recipient.as_ref() {
            struct_ser.serialize_field("recipient", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TransferCoin {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "coin",
            "amount",
            "recipient",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Coin,
            Amount,
            Recipient,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "coin" => Ok(GeneratedField::Coin),
                            "amount" => Ok(GeneratedField::Amount),
                            "recipient" => Ok(GeneratedField::Recipient),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = TransferCoin;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TransferCoin")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TransferCoin, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut coin__ = None;
                let mut amount__ = None;
                let mut recipient__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Coin => {
                            if coin__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coin"));
                            }
                            coin__ = map_.next_value()?;
                        }
                        GeneratedField::Amount => {
                            if amount__.is_some() {
                                return Err(serde::de::Error::duplicate_field("amount"));
                            }
                            amount__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Recipient => {
                            if recipient__.is_some() {
                                return Err(serde::de::Error::duplicate_field("recipient"));
                            }
                            recipient__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TransferCoin {
                    coin: coin__,
                    amount: amount__,
                    recipient: recipient__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TransferCoin", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TransferObjects {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.objects.is_empty() {
            len += 1;
        }
        if self.recipient.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TransferObjects", len)?;
        if !self.objects.is_empty() {
            struct_ser.serialize_field("objects", &self.objects)?;
        }
        if let Some(v) = self.recipient.as_ref() {
            struct_ser.serialize_field("recipient", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TransferObjects {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "objects",
            "recipient",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Objects,
            Recipient,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "objects" => Ok(GeneratedField::Objects),
                            "recipient" => Ok(GeneratedField::Recipient),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = TransferObjects;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TransferObjects")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TransferObjects, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut objects__ = None;
                let mut recipient__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Objects => {
                            if objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objects"));
                            }
                            objects__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Recipient => {
                            if recipient__.is_some() {
                                return Err(serde::de::Error::duplicate_field("recipient"));
                            }
                            recipient__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TransferObjects {
                    objects: objects__.unwrap_or_default(),
                    recipient: recipient__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TransferObjects", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for UnchangedSharedObject {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.kind.is_some() {
            len += 1;
        }
        if self.object_id.is_some() {
            len += 1;
        }
        if self.version.is_some() {
            len += 1;
        }
        if self.digest.is_some() {
            len += 1;
        }
        if self.object_type.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.UnchangedSharedObject", len)?;
        if let Some(v) = self.kind.as_ref() {
            let v = unchanged_shared_object::UnchangedSharedObjectKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.object_id.as_ref() {
            struct_ser.serialize_field("objectId", v)?;
        }
        if let Some(v) = self.version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("version", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.object_type.as_ref() {
            struct_ser.serialize_field("objectType", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for UnchangedSharedObject {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "object_id",
            "objectId",
            "version",
            "digest",
            "object_type",
            "objectType",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            ObjectId,
            Version,
            Digest,
            ObjectType,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "kind" => Ok(GeneratedField::Kind),
                            "objectId" | "object_id" => Ok(GeneratedField::ObjectId),
                            "version" => Ok(GeneratedField::Version),
                            "digest" => Ok(GeneratedField::Digest),
                            "objectType" | "object_type" => Ok(GeneratedField::ObjectType),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = UnchangedSharedObject;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.UnchangedSharedObject")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<UnchangedSharedObject, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut object_id__ = None;
                let mut version__ = None;
                let mut digest__ = None;
                let mut object_type__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<unchanged_shared_object::UnchangedSharedObjectKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::ObjectId => {
                            if object_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectId"));
                            }
                            object_id__ = map_.next_value()?;
                        }
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::ObjectType => {
                            if object_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectType"));
                            }
                            object_type__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(UnchangedSharedObject {
                    kind: kind__,
                    object_id: object_id__,
                    version: version__,
                    digest: digest__,
                    object_type: object_type__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.UnchangedSharedObject", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for unchanged_shared_object::UnchangedSharedObjectKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "UNCHANGED_SHARED_OBJECT_KIND_UNKNOWN",
            Self::ReadOnlyRoot => "READ_ONLY_ROOT",
            Self::MutatedDeleted => "MUTATED_DELETED",
            Self::ReadDeleted => "READ_DELETED",
            Self::Canceled => "CANCELED",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for unchanged_shared_object::UnchangedSharedObjectKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "UNCHANGED_SHARED_OBJECT_KIND_UNKNOWN",
            "READ_ONLY_ROOT",
            "MUTATED_DELETED",
            "READ_DELETED",
            "CANCELED",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = unchanged_shared_object::UnchangedSharedObjectKind;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "UNCHANGED_SHARED_OBJECT_KIND_UNKNOWN" => Ok(unchanged_shared_object::UnchangedSharedObjectKind::Unknown),
                    "READ_ONLY_ROOT" => Ok(unchanged_shared_object::UnchangedSharedObjectKind::ReadOnlyRoot),
                    "MUTATED_DELETED" => Ok(unchanged_shared_object::UnchangedSharedObjectKind::MutatedDeleted),
                    "READ_DELETED" => Ok(unchanged_shared_object::UnchangedSharedObjectKind::ReadDeleted),
                    "CANCELED" => Ok(unchanged_shared_object::UnchangedSharedObjectKind::Canceled),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for UndoReportChallenge {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.challenge_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.UndoReportChallenge", len)?;
        if let Some(v) = self.challenge_id.as_ref() {
            struct_ser.serialize_field("challengeId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for UndoReportChallenge {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "challenge_id",
            "challengeId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ChallengeId,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "challengeId" | "challenge_id" => Ok(GeneratedField::ChallengeId),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = UndoReportChallenge;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.UndoReportChallenge")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<UndoReportChallenge, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut challenge_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ChallengeId => {
                            if challenge_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("challengeId"));
                            }
                            challenge_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(UndoReportChallenge {
                    challenge_id: challenge_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.UndoReportChallenge", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for UndoReportModel {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.model_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.UndoReportModel", len)?;
        if let Some(v) = self.model_id.as_ref() {
            struct_ser.serialize_field("modelId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for UndoReportModel {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "model_id",
            "modelId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ModelId,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "modelId" | "model_id" => Ok(GeneratedField::ModelId),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = UndoReportModel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.UndoReportModel")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<UndoReportModel, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut model_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ModelId => {
                            if model_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modelId"));
                            }
                            model_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(UndoReportModel {
                    model_id: model_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.UndoReportModel", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for UndoReportSubmission {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.target_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.UndoReportSubmission", len)?;
        if let Some(v) = self.target_id.as_ref() {
            struct_ser.serialize_field("targetId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for UndoReportSubmission {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "target_id",
            "targetId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TargetId,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "targetId" | "target_id" => Ok(GeneratedField::TargetId),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = UndoReportSubmission;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.UndoReportSubmission")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<UndoReportSubmission, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut target_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::TargetId => {
                            if target_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("targetId"));
                            }
                            target_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(UndoReportSubmission {
                    target_id: target_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.UndoReportSubmission", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for UndoReportValidator {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.reportee.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.UndoReportValidator", len)?;
        if let Some(v) = self.reportee.as_ref() {
            struct_ser.serialize_field("reportee", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for UndoReportValidator {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "reportee",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Reportee,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "reportee" => Ok(GeneratedField::Reportee),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = UndoReportValidator;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.UndoReportValidator")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<UndoReportValidator, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut reportee__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Reportee => {
                            if reportee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reportee"));
                            }
                            reportee__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(UndoReportValidator {
                    reportee: reportee__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.UndoReportValidator", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for UpdateValidatorMetadata {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.next_epoch_network_address.is_some() {
            len += 1;
        }
        if self.next_epoch_p2p_address.is_some() {
            len += 1;
        }
        if self.next_epoch_primary_address.is_some() {
            len += 1;
        }
        if self.next_epoch_proxy_address.is_some() {
            len += 1;
        }
        if self.next_epoch_protocol_pubkey.is_some() {
            len += 1;
        }
        if self.next_epoch_worker_pubkey.is_some() {
            len += 1;
        }
        if self.next_epoch_network_pubkey.is_some() {
            len += 1;
        }
        if self.next_epoch_proof_of_possession.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.UpdateValidatorMetadata", len)?;
        if let Some(v) = self.next_epoch_network_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochNetworkAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_p2p_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochP2pAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_primary_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochPrimaryAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_proxy_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochProxyAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_protocol_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochProtocolPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_worker_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochWorkerPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_network_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochNetworkPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_proof_of_possession.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochProofOfPossession", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for UpdateValidatorMetadata {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "next_epoch_network_address",
            "nextEpochNetworkAddress",
            "next_epoch_p2p_address",
            "nextEpochP2pAddress",
            "next_epoch_primary_address",
            "nextEpochPrimaryAddress",
            "next_epoch_proxy_address",
            "nextEpochProxyAddress",
            "next_epoch_protocol_pubkey",
            "nextEpochProtocolPubkey",
            "next_epoch_worker_pubkey",
            "nextEpochWorkerPubkey",
            "next_epoch_network_pubkey",
            "nextEpochNetworkPubkey",
            "next_epoch_proof_of_possession",
            "nextEpochProofOfPossession",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            NextEpochNetworkAddress,
            NextEpochP2pAddress,
            NextEpochPrimaryAddress,
            NextEpochProxyAddress,
            NextEpochProtocolPubkey,
            NextEpochWorkerPubkey,
            NextEpochNetworkPubkey,
            NextEpochProofOfPossession,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "nextEpochNetworkAddress" | "next_epoch_network_address" => Ok(GeneratedField::NextEpochNetworkAddress),
                            "nextEpochP2pAddress" | "next_epoch_p2p_address" => Ok(GeneratedField::NextEpochP2pAddress),
                            "nextEpochPrimaryAddress" | "next_epoch_primary_address" => Ok(GeneratedField::NextEpochPrimaryAddress),
                            "nextEpochProxyAddress" | "next_epoch_proxy_address" => Ok(GeneratedField::NextEpochProxyAddress),
                            "nextEpochProtocolPubkey" | "next_epoch_protocol_pubkey" => Ok(GeneratedField::NextEpochProtocolPubkey),
                            "nextEpochWorkerPubkey" | "next_epoch_worker_pubkey" => Ok(GeneratedField::NextEpochWorkerPubkey),
                            "nextEpochNetworkPubkey" | "next_epoch_network_pubkey" => Ok(GeneratedField::NextEpochNetworkPubkey),
                            "nextEpochProofOfPossession" | "next_epoch_proof_of_possession" => Ok(GeneratedField::NextEpochProofOfPossession),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = UpdateValidatorMetadata;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.UpdateValidatorMetadata")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<UpdateValidatorMetadata, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut next_epoch_network_address__ = None;
                let mut next_epoch_p2p_address__ = None;
                let mut next_epoch_primary_address__ = None;
                let mut next_epoch_proxy_address__ = None;
                let mut next_epoch_protocol_pubkey__ = None;
                let mut next_epoch_worker_pubkey__ = None;
                let mut next_epoch_network_pubkey__ = None;
                let mut next_epoch_proof_of_possession__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::NextEpochNetworkAddress => {
                            if next_epoch_network_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochNetworkAddress"));
                            }
                            next_epoch_network_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochP2pAddress => {
                            if next_epoch_p2p_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochP2pAddress"));
                            }
                            next_epoch_p2p_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochPrimaryAddress => {
                            if next_epoch_primary_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochPrimaryAddress"));
                            }
                            next_epoch_primary_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochProxyAddress => {
                            if next_epoch_proxy_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochProxyAddress"));
                            }
                            next_epoch_proxy_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochProtocolPubkey => {
                            if next_epoch_protocol_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochProtocolPubkey"));
                            }
                            next_epoch_protocol_pubkey__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochWorkerPubkey => {
                            if next_epoch_worker_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochWorkerPubkey"));
                            }
                            next_epoch_worker_pubkey__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochNetworkPubkey => {
                            if next_epoch_network_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochNetworkPubkey"));
                            }
                            next_epoch_network_pubkey__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochProofOfPossession => {
                            if next_epoch_proof_of_possession__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochProofOfPossession"));
                            }
                            next_epoch_proof_of_possession__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(UpdateValidatorMetadata {
                    next_epoch_network_address: next_epoch_network_address__,
                    next_epoch_p2p_address: next_epoch_p2p_address__,
                    next_epoch_primary_address: next_epoch_primary_address__,
                    next_epoch_proxy_address: next_epoch_proxy_address__,
                    next_epoch_protocol_pubkey: next_epoch_protocol_pubkey__,
                    next_epoch_worker_pubkey: next_epoch_worker_pubkey__,
                    next_epoch_network_pubkey: next_epoch_network_pubkey__,
                    next_epoch_proof_of_possession: next_epoch_proof_of_possession__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.UpdateValidatorMetadata", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for UserSignature {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.scheme.is_some() {
            len += 1;
        }
        if self.signature.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.UserSignature", len)?;
        if let Some(v) = self.scheme.as_ref() {
            let v = SignatureScheme::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("scheme", &v)?;
        }
        if let Some(v) = self.signature.as_ref() {
            match v {
                user_signature::Signature::Simple(v) => {
                    struct_ser.serialize_field("simple", v)?;
                }
                user_signature::Signature::Multisig(v) => {
                    struct_ser.serialize_field("multisig", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for UserSignature {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "scheme",
            "simple",
            "multisig",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Scheme,
            Simple,
            Multisig,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "scheme" => Ok(GeneratedField::Scheme),
                            "simple" => Ok(GeneratedField::Simple),
                            "multisig" => Ok(GeneratedField::Multisig),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = UserSignature;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.UserSignature")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<UserSignature, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut scheme__ = None;
                let mut signature__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Scheme => {
                            if scheme__.is_some() {
                                return Err(serde::de::Error::duplicate_field("scheme"));
                            }
                            scheme__ = map_.next_value::<::std::option::Option<SignatureScheme>>()?.map(|x| x as i32);
                        }
                        GeneratedField::Simple => {
                            if signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("simple"));
                            }
                            signature__ = map_.next_value::<::std::option::Option<_>>()?.map(user_signature::Signature::Simple)
;
                        }
                        GeneratedField::Multisig => {
                            if signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("multisig"));
                            }
                            signature__ = map_.next_value::<::std::option::Option<_>>()?.map(user_signature::Signature::Multisig)
;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(UserSignature {
                    scheme: scheme__,
                    signature: signature__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.UserSignature", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Validator {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.soma_address.is_some() {
            len += 1;
        }
        if self.protocol_pubkey.is_some() {
            len += 1;
        }
        if self.network_pubkey.is_some() {
            len += 1;
        }
        if self.worker_pubkey.is_some() {
            len += 1;
        }
        if self.net_address.is_some() {
            len += 1;
        }
        if self.p2p_address.is_some() {
            len += 1;
        }
        if self.primary_address.is_some() {
            len += 1;
        }
        if self.proxy_address.is_some() {
            len += 1;
        }
        if self.voting_power.is_some() {
            len += 1;
        }
        if self.commission_rate.is_some() {
            len += 1;
        }
        if self.next_epoch_stake.is_some() {
            len += 1;
        }
        if self.next_epoch_commission_rate.is_some() {
            len += 1;
        }
        if self.staking_pool.is_some() {
            len += 1;
        }
        if self.next_epoch_protocol_pubkey.is_some() {
            len += 1;
        }
        if self.next_epoch_network_pubkey.is_some() {
            len += 1;
        }
        if self.next_epoch_worker_pubkey.is_some() {
            len += 1;
        }
        if self.next_epoch_net_address.is_some() {
            len += 1;
        }
        if self.next_epoch_p2p_address.is_some() {
            len += 1;
        }
        if self.next_epoch_primary_address.is_some() {
            len += 1;
        }
        if self.next_epoch_proxy_address.is_some() {
            len += 1;
        }
        if self.proof_of_possession.is_some() {
            len += 1;
        }
        if self.next_epoch_proof_of_possession.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Validator", len)?;
        if let Some(v) = self.soma_address.as_ref() {
            struct_ser.serialize_field("somaAddress", v)?;
        }
        if let Some(v) = self.protocol_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("protocolPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.network_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("networkPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.worker_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("workerPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.net_address.as_ref() {
            struct_ser.serialize_field("netAddress", v)?;
        }
        if let Some(v) = self.p2p_address.as_ref() {
            struct_ser.serialize_field("p2pAddress", v)?;
        }
        if let Some(v) = self.primary_address.as_ref() {
            struct_ser.serialize_field("primaryAddress", v)?;
        }
        if let Some(v) = self.proxy_address.as_ref() {
            struct_ser.serialize_field("proxyAddress", v)?;
        }
        if let Some(v) = self.voting_power.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("votingPower", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.commission_rate.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("commissionRate", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_stake.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochStake", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_commission_rate.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochCommissionRate", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.staking_pool.as_ref() {
            struct_ser.serialize_field("stakingPool", v)?;
        }
        if let Some(v) = self.next_epoch_protocol_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochProtocolPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_network_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochNetworkPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_worker_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochWorkerPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_net_address.as_ref() {
            struct_ser.serialize_field("nextEpochNetAddress", v)?;
        }
        if let Some(v) = self.next_epoch_p2p_address.as_ref() {
            struct_ser.serialize_field("nextEpochP2pAddress", v)?;
        }
        if let Some(v) = self.next_epoch_primary_address.as_ref() {
            struct_ser.serialize_field("nextEpochPrimaryAddress", v)?;
        }
        if let Some(v) = self.next_epoch_proxy_address.as_ref() {
            struct_ser.serialize_field("nextEpochProxyAddress", v)?;
        }
        if let Some(v) = self.proof_of_possession.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("proofOfPossession", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_proof_of_possession.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochProofOfPossession", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Validator {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "soma_address",
            "somaAddress",
            "protocol_pubkey",
            "protocolPubkey",
            "network_pubkey",
            "networkPubkey",
            "worker_pubkey",
            "workerPubkey",
            "net_address",
            "netAddress",
            "p2p_address",
            "p2pAddress",
            "primary_address",
            "primaryAddress",
            "proxy_address",
            "proxyAddress",
            "voting_power",
            "votingPower",
            "commission_rate",
            "commissionRate",
            "next_epoch_stake",
            "nextEpochStake",
            "next_epoch_commission_rate",
            "nextEpochCommissionRate",
            "staking_pool",
            "stakingPool",
            "next_epoch_protocol_pubkey",
            "nextEpochProtocolPubkey",
            "next_epoch_network_pubkey",
            "nextEpochNetworkPubkey",
            "next_epoch_worker_pubkey",
            "nextEpochWorkerPubkey",
            "next_epoch_net_address",
            "nextEpochNetAddress",
            "next_epoch_p2p_address",
            "nextEpochP2pAddress",
            "next_epoch_primary_address",
            "nextEpochPrimaryAddress",
            "next_epoch_proxy_address",
            "nextEpochProxyAddress",
            "proof_of_possession",
            "proofOfPossession",
            "next_epoch_proof_of_possession",
            "nextEpochProofOfPossession",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            SomaAddress,
            ProtocolPubkey,
            NetworkPubkey,
            WorkerPubkey,
            NetAddress,
            P2pAddress,
            PrimaryAddress,
            ProxyAddress,
            VotingPower,
            CommissionRate,
            NextEpochStake,
            NextEpochCommissionRate,
            StakingPool,
            NextEpochProtocolPubkey,
            NextEpochNetworkPubkey,
            NextEpochWorkerPubkey,
            NextEpochNetAddress,
            NextEpochP2pAddress,
            NextEpochPrimaryAddress,
            NextEpochProxyAddress,
            ProofOfPossession,
            NextEpochProofOfPossession,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "somaAddress" | "soma_address" => Ok(GeneratedField::SomaAddress),
                            "protocolPubkey" | "protocol_pubkey" => Ok(GeneratedField::ProtocolPubkey),
                            "networkPubkey" | "network_pubkey" => Ok(GeneratedField::NetworkPubkey),
                            "workerPubkey" | "worker_pubkey" => Ok(GeneratedField::WorkerPubkey),
                            "netAddress" | "net_address" => Ok(GeneratedField::NetAddress),
                            "p2pAddress" | "p2p_address" => Ok(GeneratedField::P2pAddress),
                            "primaryAddress" | "primary_address" => Ok(GeneratedField::PrimaryAddress),
                            "proxyAddress" | "proxy_address" => Ok(GeneratedField::ProxyAddress),
                            "votingPower" | "voting_power" => Ok(GeneratedField::VotingPower),
                            "commissionRate" | "commission_rate" => Ok(GeneratedField::CommissionRate),
                            "nextEpochStake" | "next_epoch_stake" => Ok(GeneratedField::NextEpochStake),
                            "nextEpochCommissionRate" | "next_epoch_commission_rate" => Ok(GeneratedField::NextEpochCommissionRate),
                            "stakingPool" | "staking_pool" => Ok(GeneratedField::StakingPool),
                            "nextEpochProtocolPubkey" | "next_epoch_protocol_pubkey" => Ok(GeneratedField::NextEpochProtocolPubkey),
                            "nextEpochNetworkPubkey" | "next_epoch_network_pubkey" => Ok(GeneratedField::NextEpochNetworkPubkey),
                            "nextEpochWorkerPubkey" | "next_epoch_worker_pubkey" => Ok(GeneratedField::NextEpochWorkerPubkey),
                            "nextEpochNetAddress" | "next_epoch_net_address" => Ok(GeneratedField::NextEpochNetAddress),
                            "nextEpochP2pAddress" | "next_epoch_p2p_address" => Ok(GeneratedField::NextEpochP2pAddress),
                            "nextEpochPrimaryAddress" | "next_epoch_primary_address" => Ok(GeneratedField::NextEpochPrimaryAddress),
                            "nextEpochProxyAddress" | "next_epoch_proxy_address" => Ok(GeneratedField::NextEpochProxyAddress),
                            "proofOfPossession" | "proof_of_possession" => Ok(GeneratedField::ProofOfPossession),
                            "nextEpochProofOfPossession" | "next_epoch_proof_of_possession" => Ok(GeneratedField::NextEpochProofOfPossession),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Validator;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Validator")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Validator, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut soma_address__ = None;
                let mut protocol_pubkey__ = None;
                let mut network_pubkey__ = None;
                let mut worker_pubkey__ = None;
                let mut net_address__ = None;
                let mut p2p_address__ = None;
                let mut primary_address__ = None;
                let mut proxy_address__ = None;
                let mut voting_power__ = None;
                let mut commission_rate__ = None;
                let mut next_epoch_stake__ = None;
                let mut next_epoch_commission_rate__ = None;
                let mut staking_pool__ = None;
                let mut next_epoch_protocol_pubkey__ = None;
                let mut next_epoch_network_pubkey__ = None;
                let mut next_epoch_worker_pubkey__ = None;
                let mut next_epoch_net_address__ = None;
                let mut next_epoch_p2p_address__ = None;
                let mut next_epoch_primary_address__ = None;
                let mut next_epoch_proxy_address__ = None;
                let mut proof_of_possession__ = None;
                let mut next_epoch_proof_of_possession__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::SomaAddress => {
                            if soma_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("somaAddress"));
                            }
                            soma_address__ = map_.next_value()?;
                        }
                        GeneratedField::ProtocolPubkey => {
                            if protocol_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("protocolPubkey"));
                            }
                            protocol_pubkey__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NetworkPubkey => {
                            if network_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("networkPubkey"));
                            }
                            network_pubkey__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::WorkerPubkey => {
                            if worker_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("workerPubkey"));
                            }
                            worker_pubkey__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NetAddress => {
                            if net_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("netAddress"));
                            }
                            net_address__ = map_.next_value()?;
                        }
                        GeneratedField::P2pAddress => {
                            if p2p_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("p2pAddress"));
                            }
                            p2p_address__ = map_.next_value()?;
                        }
                        GeneratedField::PrimaryAddress => {
                            if primary_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("primaryAddress"));
                            }
                            primary_address__ = map_.next_value()?;
                        }
                        GeneratedField::ProxyAddress => {
                            if proxy_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("proxyAddress"));
                            }
                            proxy_address__ = map_.next_value()?;
                        }
                        GeneratedField::VotingPower => {
                            if voting_power__.is_some() {
                                return Err(serde::de::Error::duplicate_field("votingPower"));
                            }
                            voting_power__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::CommissionRate => {
                            if commission_rate__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commissionRate"));
                            }
                            commission_rate__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochStake => {
                            if next_epoch_stake__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochStake"));
                            }
                            next_epoch_stake__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochCommissionRate => {
                            if next_epoch_commission_rate__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochCommissionRate"));
                            }
                            next_epoch_commission_rate__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::StakingPool => {
                            if staking_pool__.is_some() {
                                return Err(serde::de::Error::duplicate_field("stakingPool"));
                            }
                            staking_pool__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochProtocolPubkey => {
                            if next_epoch_protocol_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochProtocolPubkey"));
                            }
                            next_epoch_protocol_pubkey__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochNetworkPubkey => {
                            if next_epoch_network_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochNetworkPubkey"));
                            }
                            next_epoch_network_pubkey__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochWorkerPubkey => {
                            if next_epoch_worker_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochWorkerPubkey"));
                            }
                            next_epoch_worker_pubkey__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochNetAddress => {
                            if next_epoch_net_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochNetAddress"));
                            }
                            next_epoch_net_address__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochP2pAddress => {
                            if next_epoch_p2p_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochP2pAddress"));
                            }
                            next_epoch_p2p_address__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochPrimaryAddress => {
                            if next_epoch_primary_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochPrimaryAddress"));
                            }
                            next_epoch_primary_address__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochProxyAddress => {
                            if next_epoch_proxy_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochProxyAddress"));
                            }
                            next_epoch_proxy_address__ = map_.next_value()?;
                        }
                        GeneratedField::ProofOfPossession => {
                            if proof_of_possession__.is_some() {
                                return Err(serde::de::Error::duplicate_field("proofOfPossession"));
                            }
                            proof_of_possession__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochProofOfPossession => {
                            if next_epoch_proof_of_possession__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochProofOfPossession"));
                            }
                            next_epoch_proof_of_possession__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Validator {
                    soma_address: soma_address__,
                    protocol_pubkey: protocol_pubkey__,
                    network_pubkey: network_pubkey__,
                    worker_pubkey: worker_pubkey__,
                    net_address: net_address__,
                    p2p_address: p2p_address__,
                    primary_address: primary_address__,
                    proxy_address: proxy_address__,
                    voting_power: voting_power__,
                    commission_rate: commission_rate__,
                    next_epoch_stake: next_epoch_stake__,
                    next_epoch_commission_rate: next_epoch_commission_rate__,
                    staking_pool: staking_pool__,
                    next_epoch_protocol_pubkey: next_epoch_protocol_pubkey__,
                    next_epoch_network_pubkey: next_epoch_network_pubkey__,
                    next_epoch_worker_pubkey: next_epoch_worker_pubkey__,
                    next_epoch_net_address: next_epoch_net_address__,
                    next_epoch_p2p_address: next_epoch_p2p_address__,
                    next_epoch_primary_address: next_epoch_primary_address__,
                    next_epoch_proxy_address: next_epoch_proxy_address__,
                    proof_of_possession: proof_of_possession__,
                    next_epoch_proof_of_possession: next_epoch_proof_of_possession__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Validator", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ValidatorAggregatedSignature {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.epoch.is_some() {
            len += 1;
        }
        if self.signature.is_some() {
            len += 1;
        }
        if !self.bitmap.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ValidatorAggregatedSignature", len)?;
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.signature.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("signature", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if !self.bitmap.is_empty() {
            struct_ser.serialize_field("bitmap", &self.bitmap)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ValidatorAggregatedSignature {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epoch",
            "signature",
            "bitmap",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            Signature,
            Bitmap,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "epoch" => Ok(GeneratedField::Epoch),
                            "signature" => Ok(GeneratedField::Signature),
                            "bitmap" => Ok(GeneratedField::Bitmap),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ValidatorAggregatedSignature;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ValidatorAggregatedSignature")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ValidatorAggregatedSignature, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epoch__ = None;
                let mut signature__ = None;
                let mut bitmap__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Signature => {
                            if signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signature"));
                            }
                            signature__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Bitmap => {
                            if bitmap__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bitmap"));
                            }
                            bitmap__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ValidatorAggregatedSignature {
                    epoch: epoch__,
                    signature: signature__,
                    bitmap: bitmap__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ValidatorAggregatedSignature", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ValidatorCommittee {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.epoch.is_some() {
            len += 1;
        }
        if !self.members.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ValidatorCommittee", len)?;
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if !self.members.is_empty() {
            struct_ser.serialize_field("members", &self.members)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ValidatorCommittee {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epoch",
            "members",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            Members,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "epoch" => Ok(GeneratedField::Epoch),
                            "members" => Ok(GeneratedField::Members),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ValidatorCommittee;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ValidatorCommittee")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ValidatorCommittee, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epoch__ = None;
                let mut members__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Members => {
                            if members__.is_some() {
                                return Err(serde::de::Error::duplicate_field("members"));
                            }
                            members__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ValidatorCommittee {
                    epoch: epoch__,
                    members: members__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ValidatorCommittee", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ValidatorCommitteeMember {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.authority_key.is_some() {
            len += 1;
        }
        if self.weight.is_some() {
            len += 1;
        }
        if self.network_metadata.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ValidatorCommitteeMember", len)?;
        if let Some(v) = self.authority_key.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("authorityKey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.weight.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("weight", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.network_metadata.as_ref() {
            struct_ser.serialize_field("networkMetadata", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ValidatorCommitteeMember {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "authority_key",
            "authorityKey",
            "weight",
            "network_metadata",
            "networkMetadata",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            AuthorityKey,
            Weight,
            NetworkMetadata,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "authorityKey" | "authority_key" => Ok(GeneratedField::AuthorityKey),
                            "weight" => Ok(GeneratedField::Weight),
                            "networkMetadata" | "network_metadata" => Ok(GeneratedField::NetworkMetadata),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ValidatorCommitteeMember;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ValidatorCommitteeMember")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ValidatorCommitteeMember, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut authority_key__ = None;
                let mut weight__ = None;
                let mut network_metadata__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::AuthorityKey => {
                            if authority_key__.is_some() {
                                return Err(serde::de::Error::duplicate_field("authorityKey"));
                            }
                            authority_key__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Weight => {
                            if weight__.is_some() {
                                return Err(serde::de::Error::duplicate_field("weight"));
                            }
                            weight__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NetworkMetadata => {
                            if network_metadata__.is_some() {
                                return Err(serde::de::Error::duplicate_field("networkMetadata"));
                            }
                            network_metadata__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ValidatorCommitteeMember {
                    authority_key: authority_key__,
                    weight: weight__,
                    network_metadata: network_metadata__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ValidatorCommitteeMember", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ValidatorNetworkMetadata {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.consensus_address.is_some() {
            len += 1;
        }
        if self.hostname.is_some() {
            len += 1;
        }
        if self.protocol_key.is_some() {
            len += 1;
        }
        if self.network_key.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ValidatorNetworkMetadata", len)?;
        if let Some(v) = self.consensus_address.as_ref() {
            struct_ser.serialize_field("consensusAddress", v)?;
        }
        if let Some(v) = self.hostname.as_ref() {
            struct_ser.serialize_field("hostname", v)?;
        }
        if let Some(v) = self.protocol_key.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("protocolKey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.network_key.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("networkKey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ValidatorNetworkMetadata {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "consensus_address",
            "consensusAddress",
            "hostname",
            "protocol_key",
            "protocolKey",
            "network_key",
            "networkKey",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ConsensusAddress,
            Hostname,
            ProtocolKey,
            NetworkKey,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "consensusAddress" | "consensus_address" => Ok(GeneratedField::ConsensusAddress),
                            "hostname" => Ok(GeneratedField::Hostname),
                            "protocolKey" | "protocol_key" => Ok(GeneratedField::ProtocolKey),
                            "networkKey" | "network_key" => Ok(GeneratedField::NetworkKey),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ValidatorNetworkMetadata;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ValidatorNetworkMetadata")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ValidatorNetworkMetadata, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut consensus_address__ = None;
                let mut hostname__ = None;
                let mut protocol_key__ = None;
                let mut network_key__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ConsensusAddress => {
                            if consensus_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("consensusAddress"));
                            }
                            consensus_address__ = map_.next_value()?;
                        }
                        GeneratedField::Hostname => {
                            if hostname__.is_some() {
                                return Err(serde::de::Error::duplicate_field("hostname"));
                            }
                            hostname__ = map_.next_value()?;
                        }
                        GeneratedField::ProtocolKey => {
                            if protocol_key__.is_some() {
                                return Err(serde::de::Error::duplicate_field("protocolKey"));
                            }
                            protocol_key__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NetworkKey => {
                            if network_key__.is_some() {
                                return Err(serde::de::Error::duplicate_field("networkKey"));
                            }
                            network_key__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ValidatorNetworkMetadata {
                    consensus_address: consensus_address__,
                    hostname: hostname__,
                    protocol_key: protocol_key__,
                    network_key: network_key__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ValidatorNetworkMetadata", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ValidatorSet {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.total_stake.is_some() {
            len += 1;
        }
        if !self.validators.is_empty() {
            len += 1;
        }
        if !self.pending_validators.is_empty() {
            len += 1;
        }
        if !self.pending_removals.is_empty() {
            len += 1;
        }
        if !self.staking_pool_mappings.is_empty() {
            len += 1;
        }
        if !self.inactive_validators.is_empty() {
            len += 1;
        }
        if !self.at_risk_validators.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ValidatorSet", len)?;
        if let Some(v) = self.total_stake.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("totalStake", ToString::to_string(&v).as_str())?;
        }
        if !self.validators.is_empty() {
            struct_ser.serialize_field("validators", &self.validators)?;
        }
        if !self.pending_validators.is_empty() {
            struct_ser.serialize_field("pendingValidators", &self.pending_validators)?;
        }
        if !self.pending_removals.is_empty() {
            struct_ser.serialize_field("pendingRemovals", &self.pending_removals)?;
        }
        if !self.staking_pool_mappings.is_empty() {
            struct_ser.serialize_field("stakingPoolMappings", &self.staking_pool_mappings)?;
        }
        if !self.inactive_validators.is_empty() {
            struct_ser.serialize_field("inactiveValidators", &self.inactive_validators)?;
        }
        if !self.at_risk_validators.is_empty() {
            let v: std::collections::BTreeMap<_, _> = self.at_risk_validators.iter()
                .map(|(k, v)| (k, v.to_string())).collect();
            struct_ser.serialize_field("atRiskValidators", &v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ValidatorSet {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "total_stake",
            "totalStake",
            "validators",
            "pending_validators",
            "pendingValidators",
            "pending_removals",
            "pendingRemovals",
            "staking_pool_mappings",
            "stakingPoolMappings",
            "inactive_validators",
            "inactiveValidators",
            "at_risk_validators",
            "atRiskValidators",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TotalStake,
            Validators,
            PendingValidators,
            PendingRemovals,
            StakingPoolMappings,
            InactiveValidators,
            AtRiskValidators,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "totalStake" | "total_stake" => Ok(GeneratedField::TotalStake),
                            "validators" => Ok(GeneratedField::Validators),
                            "pendingValidators" | "pending_validators" => Ok(GeneratedField::PendingValidators),
                            "pendingRemovals" | "pending_removals" => Ok(GeneratedField::PendingRemovals),
                            "stakingPoolMappings" | "staking_pool_mappings" => Ok(GeneratedField::StakingPoolMappings),
                            "inactiveValidators" | "inactive_validators" => Ok(GeneratedField::InactiveValidators),
                            "atRiskValidators" | "at_risk_validators" => Ok(GeneratedField::AtRiskValidators),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ValidatorSet;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ValidatorSet")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ValidatorSet, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut total_stake__ = None;
                let mut validators__ = None;
                let mut pending_validators__ = None;
                let mut pending_removals__ = None;
                let mut staking_pool_mappings__ = None;
                let mut inactive_validators__ = None;
                let mut at_risk_validators__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::TotalStake => {
                            if total_stake__.is_some() {
                                return Err(serde::de::Error::duplicate_field("totalStake"));
                            }
                            total_stake__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Validators => {
                            if validators__.is_some() {
                                return Err(serde::de::Error::duplicate_field("validators"));
                            }
                            validators__ = Some(map_.next_value()?);
                        }
                        GeneratedField::PendingValidators => {
                            if pending_validators__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pendingValidators"));
                            }
                            pending_validators__ = Some(map_.next_value()?);
                        }
                        GeneratedField::PendingRemovals => {
                            if pending_removals__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pendingRemovals"));
                            }
                            pending_removals__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::StakingPoolMappings => {
                            if staking_pool_mappings__.is_some() {
                                return Err(serde::de::Error::duplicate_field("stakingPoolMappings"));
                            }
                            staking_pool_mappings__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::InactiveValidators => {
                            if inactive_validators__.is_some() {
                                return Err(serde::de::Error::duplicate_field("inactiveValidators"));
                            }
                            inactive_validators__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::AtRiskValidators => {
                            if at_risk_validators__.is_some() {
                                return Err(serde::de::Error::duplicate_field("atRiskValidators"));
                            }
                            at_risk_validators__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, crate::utils::_serde::NumberDeserialize<u64>>>()?
                                    .into_iter().map(|(k,v)| (k, v.0)).collect()
                            );
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ValidatorSet {
                    total_stake: total_stake__,
                    validators: validators__.unwrap_or_default(),
                    pending_validators: pending_validators__.unwrap_or_default(),
                    pending_removals: pending_removals__.unwrap_or_default(),
                    staking_pool_mappings: staking_pool_mappings__.unwrap_or_default(),
                    inactive_validators: inactive_validators__.unwrap_or_default(),
                    at_risk_validators: at_risk_validators__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ValidatorSet", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for WithdrawStake {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.staked_soma.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.WithdrawStake", len)?;
        if let Some(v) = self.staked_soma.as_ref() {
            struct_ser.serialize_field("stakedSoma", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for WithdrawStake {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "staked_soma",
            "stakedSoma",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            StakedSoma,
            __SkipField__,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "stakedSoma" | "staked_soma" => Ok(GeneratedField::StakedSoma),
                            _ => Ok(GeneratedField::__SkipField__),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unit_arg)]
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = WithdrawStake;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.WithdrawStake")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<WithdrawStake, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut staked_soma__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::StakedSoma => {
                            if staked_soma__.is_some() {
                                return Err(serde::de::Error::duplicate_field("stakedSoma"));
                            }
                            staked_soma__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(WithdrawStake {
                    staked_soma: staked_soma__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.WithdrawStake", FIELDS, GeneratedVisitor)
    }
}
