impl serde::Serialize for AddEncoder {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.encoder_pubkey_bytes.is_some() {
            len += 1;
        }
        if self.network_pubkey_bytes.is_some() {
            len += 1;
        }
        if self.internal_network_address.is_some() {
            len += 1;
        }
        if self.external_network_address.is_some() {
            len += 1;
        }
        if self.object_server_address.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.AddEncoder", len)?;
        if let Some(v) = self.encoder_pubkey_bytes.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("encoderPubkeyBytes", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.network_pubkey_bytes.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("networkPubkeyBytes", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.internal_network_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("internalNetworkAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.external_network_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("externalNetworkAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.object_server_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("objectServerAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for AddEncoder {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "encoder_pubkey_bytes",
            "encoderPubkeyBytes",
            "network_pubkey_bytes",
            "networkPubkeyBytes",
            "internal_network_address",
            "internalNetworkAddress",
            "external_network_address",
            "externalNetworkAddress",
            "object_server_address",
            "objectServerAddress",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            EncoderPubkeyBytes,
            NetworkPubkeyBytes,
            InternalNetworkAddress,
            ExternalNetworkAddress,
            ObjectServerAddress,
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
                            "encoderPubkeyBytes" | "encoder_pubkey_bytes" => Ok(GeneratedField::EncoderPubkeyBytes),
                            "networkPubkeyBytes" | "network_pubkey_bytes" => Ok(GeneratedField::NetworkPubkeyBytes),
                            "internalNetworkAddress" | "internal_network_address" => Ok(GeneratedField::InternalNetworkAddress),
                            "externalNetworkAddress" | "external_network_address" => Ok(GeneratedField::ExternalNetworkAddress),
                            "objectServerAddress" | "object_server_address" => Ok(GeneratedField::ObjectServerAddress),
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
            type Value = AddEncoder;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.AddEncoder")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<AddEncoder, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut encoder_pubkey_bytes__ = None;
                let mut network_pubkey_bytes__ = None;
                let mut internal_network_address__ = None;
                let mut external_network_address__ = None;
                let mut object_server_address__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::EncoderPubkeyBytes => {
                            if encoder_pubkey_bytes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("encoderPubkeyBytes"));
                            }
                            encoder_pubkey_bytes__ = 
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
                        GeneratedField::InternalNetworkAddress => {
                            if internal_network_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("internalNetworkAddress"));
                            }
                            internal_network_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ExternalNetworkAddress => {
                            if external_network_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("externalNetworkAddress"));
                            }
                            external_network_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ObjectServerAddress => {
                            if object_server_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectServerAddress"));
                            }
                            object_server_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(AddEncoder {
                    encoder_pubkey_bytes: encoder_pubkey_bytes__,
                    network_pubkey_bytes: network_pubkey_bytes__,
                    internal_network_address: internal_network_address__,
                    external_network_address: external_network_address__,
                    object_server_address: object_server_address__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.AddEncoder", FIELDS, GeneratedVisitor)
    }
}
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
impl serde::Serialize for AddStakeToEncoder {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.encoder_address.is_some() {
            len += 1;
        }
        if self.coin_ref.is_some() {
            len += 1;
        }
        if self.amount.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.AddStakeToEncoder", len)?;
        if let Some(v) = self.encoder_address.as_ref() {
            struct_ser.serialize_field("encoderAddress", v)?;
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
impl<'de> serde::Deserialize<'de> for AddStakeToEncoder {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "encoder_address",
            "encoderAddress",
            "coin_ref",
            "coinRef",
            "amount",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            EncoderAddress,
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
                            "encoderAddress" | "encoder_address" => Ok(GeneratedField::EncoderAddress),
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
            type Value = AddStakeToEncoder;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.AddStakeToEncoder")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<AddStakeToEncoder, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut encoder_address__ = None;
                let mut coin_ref__ = None;
                let mut amount__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::EncoderAddress => {
                            if encoder_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("encoderAddress"));
                            }
                            encoder_address__ = map_.next_value()?;
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
                Ok(AddStakeToEncoder {
                    encoder_address: encoder_address__,
                    coin_ref: coin_ref__,
                    amount: amount__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.AddStakeToEncoder", FIELDS, GeneratedVisitor)
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
        if self.encoder_validator_address.is_some() {
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
        if let Some(v) = self.encoder_validator_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("encoderValidatorAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
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
            "encoder_validator_address",
            "encoderValidatorAddress",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            PubkeyBytes,
            NetworkPubkeyBytes,
            WorkerPubkeyBytes,
            NetAddress,
            P2pAddress,
            PrimaryAddress,
            EncoderValidatorAddress,
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
                            "encoderValidatorAddress" | "encoder_validator_address" => Ok(GeneratedField::EncoderValidatorAddress),
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
                let mut encoder_validator_address__ = None;
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
                        GeneratedField::EncoderValidatorAddress => {
                            if encoder_validator_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("encoderValidatorAddress"));
                            }
                            encoder_validator_address__ = 
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
                    encoder_validator_address: encoder_validator_address__,
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
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ChangeEpoch", len)?;
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.epoch_start_timestamp.as_ref() {
            struct_ser.serialize_field("epochStartTimestamp", &crate::utils::_serde::TimestampSerializer(v))?;
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
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            EpochStartTimestamp,
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
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ChangeEpoch {
                    epoch: epoch__,
                    epoch_start_timestamp: epoch_start_timestamp__,
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
impl serde::Serialize for ClaimEscrow {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.shard_input_ref.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ClaimEscrow", len)?;
        if let Some(v) = self.shard_input_ref.as_ref() {
            struct_ser.serialize_field("shardInputRef", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ClaimEscrow {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "shard_input_ref",
            "shardInputRef",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ShardInputRef,
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
                            "shardInputRef" | "shard_input_ref" => Ok(GeneratedField::ShardInputRef),
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
            type Value = ClaimEscrow;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ClaimEscrow")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ClaimEscrow, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut shard_input_ref__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ShardInputRef => {
                            if shard_input_ref__.is_some() {
                                return Err(serde::de::Error::duplicate_field("shardInputRef"));
                            }
                            shard_input_ref__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ClaimEscrow {
                    shard_input_ref: shard_input_ref__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ClaimEscrow", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Commit {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.index.is_some() {
            len += 1;
        }
        if self.digest.is_some() {
            len += 1;
        }
        if self.timestamp_ms.is_some() {
            len += 1;
        }
        if !self.transactions.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Commit", len)?;
        if let Some(v) = self.index.as_ref() {
            struct_ser.serialize_field("index", v)?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.timestamp_ms.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("timestampMs", ToString::to_string(&v).as_str())?;
        }
        if !self.transactions.is_empty() {
            struct_ser.serialize_field("transactions", &self.transactions)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Commit {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "index",
            "digest",
            "timestamp_ms",
            "timestampMs",
            "transactions",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Index,
            Digest,
            TimestampMs,
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
                            "index" => Ok(GeneratedField::Index),
                            "digest" => Ok(GeneratedField::Digest),
                            "timestampMs" | "timestamp_ms" => Ok(GeneratedField::TimestampMs),
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
            type Value = Commit;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Commit")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Commit, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut index__ = None;
                let mut digest__ = None;
                let mut timestamp_ms__ = None;
                let mut transactions__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Index => {
                            if index__.is_some() {
                                return Err(serde::de::Error::duplicate_field("index"));
                            }
                            index__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::TimestampMs => {
                            if timestamp_ms__.is_some() {
                                return Err(serde::de::Error::duplicate_field("timestampMs"));
                            }
                            timestamp_ms__ = 
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
                Ok(Commit {
                    index: index__,
                    digest: digest__,
                    timestamp_ms: timestamp_ms__,
                    transactions: transactions__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Commit", FIELDS, GeneratedVisitor)
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
        if self.commit_timestamp.is_some() {
            len += 1;
        }
        if self.consensus_commit_digest.is_some() {
            len += 1;
        }
        if self.sub_dag_index.is_some() {
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
        if let Some(v) = self.commit_timestamp.as_ref() {
            struct_ser.serialize_field("commitTimestamp", &crate::utils::_serde::TimestampSerializer(v))?;
        }
        if let Some(v) = self.consensus_commit_digest.as_ref() {
            struct_ser.serialize_field("consensusCommitDigest", v)?;
        }
        if let Some(v) = self.sub_dag_index.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("subDagIndex", ToString::to_string(&v).as_str())?;
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
            "commit_timestamp",
            "commitTimestamp",
            "consensus_commit_digest",
            "consensusCommitDigest",
            "sub_dag_index",
            "subDagIndex",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            Round,
            CommitTimestamp,
            ConsensusCommitDigest,
            SubDagIndex,
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
                            "commitTimestamp" | "commit_timestamp" => Ok(GeneratedField::CommitTimestamp),
                            "consensusCommitDigest" | "consensus_commit_digest" => Ok(GeneratedField::ConsensusCommitDigest),
                            "subDagIndex" | "sub_dag_index" => Ok(GeneratedField::SubDagIndex),
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
                let mut commit_timestamp__ = None;
                let mut consensus_commit_digest__ = None;
                let mut sub_dag_index__ = None;
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
                        GeneratedField::SubDagIndex => {
                            if sub_dag_index__.is_some() {
                                return Err(serde::de::Error::duplicate_field("subDagIndex"));
                            }
                            sub_dag_index__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ConsensusCommitPrologue {
                    epoch: epoch__,
                    round: round__,
                    commit_timestamp: commit_timestamp__,
                    consensus_commit_digest: consensus_commit_digest__,
                    sub_dag_index: sub_dag_index__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ConsensusCommitPrologue", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for EmbedData {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.metadata.is_some() {
            len += 1;
        }
        if self.coin_ref.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.EmbedData", len)?;
        if let Some(v) = self.metadata.as_ref() {
            struct_ser.serialize_field("metadata", v)?;
        }
        if let Some(v) = self.coin_ref.as_ref() {
            struct_ser.serialize_field("coinRef", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for EmbedData {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "metadata",
            "coin_ref",
            "coinRef",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Metadata,
            CoinRef,
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
                            "metadata" => Ok(GeneratedField::Metadata),
                            "coinRef" | "coin_ref" => Ok(GeneratedField::CoinRef),
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
            type Value = EmbedData;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.EmbedData")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<EmbedData, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut metadata__ = None;
                let mut coin_ref__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Metadata => {
                            if metadata__.is_some() {
                                return Err(serde::de::Error::duplicate_field("metadata"));
                            }
                            metadata__ = map_.next_value()?;
                        }
                        GeneratedField::CoinRef => {
                            if coin_ref__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coinRef"));
                            }
                            coin_ref__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(EmbedData {
                    metadata: metadata__,
                    coin_ref: coin_ref__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.EmbedData", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Encoder {
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
        if self.encoder_pubkey.is_some() {
            len += 1;
        }
        if self.network_pubkey.is_some() {
            len += 1;
        }
        if self.internal_network_address.is_some() {
            len += 1;
        }
        if self.external_network_address.is_some() {
            len += 1;
        }
        if self.object_server_address.is_some() {
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
        if self.byte_price.is_some() {
            len += 1;
        }
        if self.next_epoch_byte_price.is_some() {
            len += 1;
        }
        if self.staking_pool.is_some() {
            len += 1;
        }
        if self.next_epoch_network_pubkey.is_some() {
            len += 1;
        }
        if self.next_epoch_internal_network_address.is_some() {
            len += 1;
        }
        if self.next_epoch_external_network_address.is_some() {
            len += 1;
        }
        if self.next_epoch_object_server_address.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Encoder", len)?;
        if let Some(v) = self.soma_address.as_ref() {
            struct_ser.serialize_field("somaAddress", v)?;
        }
        if let Some(v) = self.encoder_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("encoderPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.network_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("networkPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.internal_network_address.as_ref() {
            struct_ser.serialize_field("internalNetworkAddress", v)?;
        }
        if let Some(v) = self.external_network_address.as_ref() {
            struct_ser.serialize_field("externalNetworkAddress", v)?;
        }
        if let Some(v) = self.object_server_address.as_ref() {
            struct_ser.serialize_field("objectServerAddress", v)?;
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
        if let Some(v) = self.byte_price.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("bytePrice", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_byte_price.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochBytePrice", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.staking_pool.as_ref() {
            struct_ser.serialize_field("stakingPool", v)?;
        }
        if let Some(v) = self.next_epoch_network_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochNetworkPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_internal_network_address.as_ref() {
            struct_ser.serialize_field("nextEpochInternalNetworkAddress", v)?;
        }
        if let Some(v) = self.next_epoch_external_network_address.as_ref() {
            struct_ser.serialize_field("nextEpochExternalNetworkAddress", v)?;
        }
        if let Some(v) = self.next_epoch_object_server_address.as_ref() {
            struct_ser.serialize_field("nextEpochObjectServerAddress", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Encoder {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "soma_address",
            "somaAddress",
            "encoder_pubkey",
            "encoderPubkey",
            "network_pubkey",
            "networkPubkey",
            "internal_network_address",
            "internalNetworkAddress",
            "external_network_address",
            "externalNetworkAddress",
            "object_server_address",
            "objectServerAddress",
            "voting_power",
            "votingPower",
            "commission_rate",
            "commissionRate",
            "next_epoch_stake",
            "nextEpochStake",
            "next_epoch_commission_rate",
            "nextEpochCommissionRate",
            "byte_price",
            "bytePrice",
            "next_epoch_byte_price",
            "nextEpochBytePrice",
            "staking_pool",
            "stakingPool",
            "next_epoch_network_pubkey",
            "nextEpochNetworkPubkey",
            "next_epoch_internal_network_address",
            "nextEpochInternalNetworkAddress",
            "next_epoch_external_network_address",
            "nextEpochExternalNetworkAddress",
            "next_epoch_object_server_address",
            "nextEpochObjectServerAddress",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            SomaAddress,
            EncoderPubkey,
            NetworkPubkey,
            InternalNetworkAddress,
            ExternalNetworkAddress,
            ObjectServerAddress,
            VotingPower,
            CommissionRate,
            NextEpochStake,
            NextEpochCommissionRate,
            BytePrice,
            NextEpochBytePrice,
            StakingPool,
            NextEpochNetworkPubkey,
            NextEpochInternalNetworkAddress,
            NextEpochExternalNetworkAddress,
            NextEpochObjectServerAddress,
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
                            "encoderPubkey" | "encoder_pubkey" => Ok(GeneratedField::EncoderPubkey),
                            "networkPubkey" | "network_pubkey" => Ok(GeneratedField::NetworkPubkey),
                            "internalNetworkAddress" | "internal_network_address" => Ok(GeneratedField::InternalNetworkAddress),
                            "externalNetworkAddress" | "external_network_address" => Ok(GeneratedField::ExternalNetworkAddress),
                            "objectServerAddress" | "object_server_address" => Ok(GeneratedField::ObjectServerAddress),
                            "votingPower" | "voting_power" => Ok(GeneratedField::VotingPower),
                            "commissionRate" | "commission_rate" => Ok(GeneratedField::CommissionRate),
                            "nextEpochStake" | "next_epoch_stake" => Ok(GeneratedField::NextEpochStake),
                            "nextEpochCommissionRate" | "next_epoch_commission_rate" => Ok(GeneratedField::NextEpochCommissionRate),
                            "bytePrice" | "byte_price" => Ok(GeneratedField::BytePrice),
                            "nextEpochBytePrice" | "next_epoch_byte_price" => Ok(GeneratedField::NextEpochBytePrice),
                            "stakingPool" | "staking_pool" => Ok(GeneratedField::StakingPool),
                            "nextEpochNetworkPubkey" | "next_epoch_network_pubkey" => Ok(GeneratedField::NextEpochNetworkPubkey),
                            "nextEpochInternalNetworkAddress" | "next_epoch_internal_network_address" => Ok(GeneratedField::NextEpochInternalNetworkAddress),
                            "nextEpochExternalNetworkAddress" | "next_epoch_external_network_address" => Ok(GeneratedField::NextEpochExternalNetworkAddress),
                            "nextEpochObjectServerAddress" | "next_epoch_object_server_address" => Ok(GeneratedField::NextEpochObjectServerAddress),
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
            type Value = Encoder;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Encoder")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Encoder, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut soma_address__ = None;
                let mut encoder_pubkey__ = None;
                let mut network_pubkey__ = None;
                let mut internal_network_address__ = None;
                let mut external_network_address__ = None;
                let mut object_server_address__ = None;
                let mut voting_power__ = None;
                let mut commission_rate__ = None;
                let mut next_epoch_stake__ = None;
                let mut next_epoch_commission_rate__ = None;
                let mut byte_price__ = None;
                let mut next_epoch_byte_price__ = None;
                let mut staking_pool__ = None;
                let mut next_epoch_network_pubkey__ = None;
                let mut next_epoch_internal_network_address__ = None;
                let mut next_epoch_external_network_address__ = None;
                let mut next_epoch_object_server_address__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::SomaAddress => {
                            if soma_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("somaAddress"));
                            }
                            soma_address__ = map_.next_value()?;
                        }
                        GeneratedField::EncoderPubkey => {
                            if encoder_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("encoderPubkey"));
                            }
                            encoder_pubkey__ = 
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
                        GeneratedField::InternalNetworkAddress => {
                            if internal_network_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("internalNetworkAddress"));
                            }
                            internal_network_address__ = map_.next_value()?;
                        }
                        GeneratedField::ExternalNetworkAddress => {
                            if external_network_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("externalNetworkAddress"));
                            }
                            external_network_address__ = map_.next_value()?;
                        }
                        GeneratedField::ObjectServerAddress => {
                            if object_server_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectServerAddress"));
                            }
                            object_server_address__ = map_.next_value()?;
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
                        GeneratedField::BytePrice => {
                            if byte_price__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bytePrice"));
                            }
                            byte_price__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochBytePrice => {
                            if next_epoch_byte_price__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochBytePrice"));
                            }
                            next_epoch_byte_price__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::StakingPool => {
                            if staking_pool__.is_some() {
                                return Err(serde::de::Error::duplicate_field("stakingPool"));
                            }
                            staking_pool__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochNetworkPubkey => {
                            if next_epoch_network_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochNetworkPubkey"));
                            }
                            next_epoch_network_pubkey__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochInternalNetworkAddress => {
                            if next_epoch_internal_network_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochInternalNetworkAddress"));
                            }
                            next_epoch_internal_network_address__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochExternalNetworkAddress => {
                            if next_epoch_external_network_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochExternalNetworkAddress"));
                            }
                            next_epoch_external_network_address__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochObjectServerAddress => {
                            if next_epoch_object_server_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochObjectServerAddress"));
                            }
                            next_epoch_object_server_address__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Encoder {
                    soma_address: soma_address__,
                    encoder_pubkey: encoder_pubkey__,
                    network_pubkey: network_pubkey__,
                    internal_network_address: internal_network_address__,
                    external_network_address: external_network_address__,
                    object_server_address: object_server_address__,
                    voting_power: voting_power__,
                    commission_rate: commission_rate__,
                    next_epoch_stake: next_epoch_stake__,
                    next_epoch_commission_rate: next_epoch_commission_rate__,
                    byte_price: byte_price__,
                    next_epoch_byte_price: next_epoch_byte_price__,
                    staking_pool: staking_pool__,
                    next_epoch_network_pubkey: next_epoch_network_pubkey__,
                    next_epoch_internal_network_address: next_epoch_internal_network_address__,
                    next_epoch_external_network_address: next_epoch_external_network_address__,
                    next_epoch_object_server_address: next_epoch_object_server_address__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Encoder", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for EncoderSet {
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
        if !self.active_encoders.is_empty() {
            len += 1;
        }
        if !self.pending_active_encoders.is_empty() {
            len += 1;
        }
        if !self.pending_removals.is_empty() {
            len += 1;
        }
        if !self.staking_pool_mappings.is_empty() {
            len += 1;
        }
        if !self.inactive_encoders.is_empty() {
            len += 1;
        }
        if !self.at_risk_encoders.is_empty() {
            len += 1;
        }
        if self.reference_byte_price.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.EncoderSet", len)?;
        if let Some(v) = self.total_stake.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("totalStake", ToString::to_string(&v).as_str())?;
        }
        if !self.active_encoders.is_empty() {
            struct_ser.serialize_field("activeEncoders", &self.active_encoders)?;
        }
        if !self.pending_active_encoders.is_empty() {
            struct_ser.serialize_field("pendingActiveEncoders", &self.pending_active_encoders)?;
        }
        if !self.pending_removals.is_empty() {
            struct_ser.serialize_field("pendingRemovals", &self.pending_removals)?;
        }
        if !self.staking_pool_mappings.is_empty() {
            struct_ser.serialize_field("stakingPoolMappings", &self.staking_pool_mappings)?;
        }
        if !self.inactive_encoders.is_empty() {
            struct_ser.serialize_field("inactiveEncoders", &self.inactive_encoders)?;
        }
        if !self.at_risk_encoders.is_empty() {
            let v: std::collections::BTreeMap<_, _> = self.at_risk_encoders.iter()
                .map(|(k, v)| (k, v.to_string())).collect();
            struct_ser.serialize_field("atRiskEncoders", &v)?;
        }
        if let Some(v) = self.reference_byte_price.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("referenceBytePrice", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for EncoderSet {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "total_stake",
            "totalStake",
            "active_encoders",
            "activeEncoders",
            "pending_active_encoders",
            "pendingActiveEncoders",
            "pending_removals",
            "pendingRemovals",
            "staking_pool_mappings",
            "stakingPoolMappings",
            "inactive_encoders",
            "inactiveEncoders",
            "at_risk_encoders",
            "atRiskEncoders",
            "reference_byte_price",
            "referenceBytePrice",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TotalStake,
            ActiveEncoders,
            PendingActiveEncoders,
            PendingRemovals,
            StakingPoolMappings,
            InactiveEncoders,
            AtRiskEncoders,
            ReferenceBytePrice,
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
                            "activeEncoders" | "active_encoders" => Ok(GeneratedField::ActiveEncoders),
                            "pendingActiveEncoders" | "pending_active_encoders" => Ok(GeneratedField::PendingActiveEncoders),
                            "pendingRemovals" | "pending_removals" => Ok(GeneratedField::PendingRemovals),
                            "stakingPoolMappings" | "staking_pool_mappings" => Ok(GeneratedField::StakingPoolMappings),
                            "inactiveEncoders" | "inactive_encoders" => Ok(GeneratedField::InactiveEncoders),
                            "atRiskEncoders" | "at_risk_encoders" => Ok(GeneratedField::AtRiskEncoders),
                            "referenceBytePrice" | "reference_byte_price" => Ok(GeneratedField::ReferenceBytePrice),
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
            type Value = EncoderSet;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.EncoderSet")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<EncoderSet, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut total_stake__ = None;
                let mut active_encoders__ = None;
                let mut pending_active_encoders__ = None;
                let mut pending_removals__ = None;
                let mut staking_pool_mappings__ = None;
                let mut inactive_encoders__ = None;
                let mut at_risk_encoders__ = None;
                let mut reference_byte_price__ = None;
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
                        GeneratedField::ActiveEncoders => {
                            if active_encoders__.is_some() {
                                return Err(serde::de::Error::duplicate_field("activeEncoders"));
                            }
                            active_encoders__ = Some(map_.next_value()?);
                        }
                        GeneratedField::PendingActiveEncoders => {
                            if pending_active_encoders__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pendingActiveEncoders"));
                            }
                            pending_active_encoders__ = Some(map_.next_value()?);
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
                        GeneratedField::InactiveEncoders => {
                            if inactive_encoders__.is_some() {
                                return Err(serde::de::Error::duplicate_field("inactiveEncoders"));
                            }
                            inactive_encoders__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::AtRiskEncoders => {
                            if at_risk_encoders__.is_some() {
                                return Err(serde::de::Error::duplicate_field("atRiskEncoders"));
                            }
                            at_risk_encoders__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, crate::utils::_serde::NumberDeserialize<u64>>>()?
                                    .into_iter().map(|(k,v)| (k, v.0)).collect()
                            );
                        }
                        GeneratedField::ReferenceBytePrice => {
                            if reference_byte_price__.is_some() {
                                return Err(serde::de::Error::duplicate_field("referenceBytePrice"));
                            }
                            reference_byte_price__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(EncoderSet {
                    total_stake: total_stake__,
                    active_encoders: active_encoders__.unwrap_or_default(),
                    pending_active_encoders: pending_active_encoders__.unwrap_or_default(),
                    pending_removals: pending_removals__.unwrap_or_default(),
                    staking_pool_mappings: staking_pool_mappings__.unwrap_or_default(),
                    inactive_encoders: inactive_encoders__.unwrap_or_default(),
                    at_risk_encoders: at_risk_encoders__.unwrap_or_default(),
                    reference_byte_price: reference_byte_price__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.EncoderSet", FIELDS, GeneratedVisitor)
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
        if self.start.is_some() {
            len += 1;
        }
        if self.end.is_some() {
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
        if let Some(v) = self.start.as_ref() {
            struct_ser.serialize_field("start", &crate::utils::_serde::TimestampSerializer(v))?;
        }
        if let Some(v) = self.end.as_ref() {
            struct_ser.serialize_field("end", &crate::utils::_serde::TimestampSerializer(v))?;
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
            "start",
            "end",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            Committee,
            SystemState,
            Start,
            End,
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
                            "start" => Ok(GeneratedField::Start),
                            "end" => Ok(GeneratedField::End),
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
                let mut start__ = None;
                let mut end__ = None;
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
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Epoch {
                    epoch: epoch__,
                    committee: committee__,
                    system_state: system_state__,
                    start: start__,
                    end: end__,
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
        if self.finality.is_some() {
            len += 1;
        }
        if self.transaction.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ExecuteTransactionResponse", len)?;
        if let Some(v) = self.finality.as_ref() {
            struct_ser.serialize_field("finality", v)?;
        }
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
            "finality",
            "transaction",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Finality,
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
                            "finality" => Ok(GeneratedField::Finality),
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
                let mut finality__ = None;
                let mut transaction__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Finality => {
                            if finality__.is_some() {
                                return Err(serde::de::Error::duplicate_field("finality"));
                            }
                            finality__ = map_.next_value()?;
                        }
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
                    finality: finality__,
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
        if self.commit.is_some() {
            len += 1;
        }
        if self.timestamp.is_some() {
            len += 1;
        }
        if !self.balance_changes.is_empty() {
            len += 1;
        }
        if !self.input_objects.is_empty() {
            len += 1;
        }
        if !self.output_objects.is_empty() {
            len += 1;
        }
        if self.shard.is_some() {
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
        if let Some(v) = self.commit.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("commit", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.timestamp.as_ref() {
            struct_ser.serialize_field("timestamp", &crate::utils::_serde::TimestampSerializer(v))?;
        }
        if !self.balance_changes.is_empty() {
            struct_ser.serialize_field("balanceChanges", &self.balance_changes)?;
        }
        if !self.input_objects.is_empty() {
            struct_ser.serialize_field("inputObjects", &self.input_objects)?;
        }
        if !self.output_objects.is_empty() {
            struct_ser.serialize_field("outputObjects", &self.output_objects)?;
        }
        if let Some(v) = self.shard.as_ref() {
            struct_ser.serialize_field("shard", v)?;
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
            "commit",
            "timestamp",
            "balance_changes",
            "balanceChanges",
            "input_objects",
            "inputObjects",
            "output_objects",
            "outputObjects",
            "shard",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Digest,
            Transaction,
            Signatures,
            Effects,
            Commit,
            Timestamp,
            BalanceChanges,
            InputObjects,
            OutputObjects,
            Shard,
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
                            "commit" => Ok(GeneratedField::Commit),
                            "timestamp" => Ok(GeneratedField::Timestamp),
                            "balanceChanges" | "balance_changes" => Ok(GeneratedField::BalanceChanges),
                            "inputObjects" | "input_objects" => Ok(GeneratedField::InputObjects),
                            "outputObjects" | "output_objects" => Ok(GeneratedField::OutputObjects),
                            "shard" => Ok(GeneratedField::Shard),
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
                let mut commit__ = None;
                let mut timestamp__ = None;
                let mut balance_changes__ = None;
                let mut input_objects__ = None;
                let mut output_objects__ = None;
                let mut shard__ = None;
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
                        GeneratedField::Commit => {
                            if commit__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commit"));
                            }
                            commit__ = 
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
                        GeneratedField::InputObjects => {
                            if input_objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("inputObjects"));
                            }
                            input_objects__ = Some(map_.next_value()?);
                        }
                        GeneratedField::OutputObjects => {
                            if output_objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("outputObjects"));
                            }
                            output_objects__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Shard => {
                            if shard__.is_some() {
                                return Err(serde::de::Error::duplicate_field("shard"));
                            }
                            shard__ = map_.next_value()?;
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
                    commit: commit__,
                    timestamp: timestamp__,
                    balance_changes: balance_changes__.unwrap_or_default(),
                    input_objects: input_objects__.unwrap_or_default(),
                    output_objects: output_objects__.unwrap_or_default(),
                    shard: shard__,
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
            Self::DuplicateEncoder => "DUPLICATE_ENCODER",
            Self::NotAnEncoder => "NOT_AN_ENCODER",
            Self::EncoderAlreadyRemoved => "ENCODER_ALREADY_REMOVED",
            Self::InsufficientCoinBalance => "INSUFFICIENT_COIN_BALANCE",
            Self::CoinBalanceOverflow => "COIN_BALANCE_OVERFLOW",
            Self::ValidatorNotFound => "VALIDATOR_NOT_FOUND",
            Self::EncoderNotFound => "ENCODER_NOT_FOUND",
            Self::StakingPoolNotFound => "STAKING_POOL_NOT_FOUND",
            Self::CannotReportOneself => "CANNOT_REPORT_ONESELF",
            Self::ReportRecordNotFound => "REPORT_RECORD_NOT_FOUND",
            Self::OtherError => "OTHER_ERROR",
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
            "DUPLICATE_ENCODER",
            "NOT_AN_ENCODER",
            "ENCODER_ALREADY_REMOVED",
            "INSUFFICIENT_COIN_BALANCE",
            "COIN_BALANCE_OVERFLOW",
            "VALIDATOR_NOT_FOUND",
            "ENCODER_NOT_FOUND",
            "STAKING_POOL_NOT_FOUND",
            "CANNOT_REPORT_ONESELF",
            "REPORT_RECORD_NOT_FOUND",
            "OTHER_ERROR",
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
                    "DUPLICATE_ENCODER" => Ok(execution_error::ExecutionErrorKind::DuplicateEncoder),
                    "NOT_AN_ENCODER" => Ok(execution_error::ExecutionErrorKind::NotAnEncoder),
                    "ENCODER_ALREADY_REMOVED" => Ok(execution_error::ExecutionErrorKind::EncoderAlreadyRemoved),
                    "INSUFFICIENT_COIN_BALANCE" => Ok(execution_error::ExecutionErrorKind::InsufficientCoinBalance),
                    "COIN_BALANCE_OVERFLOW" => Ok(execution_error::ExecutionErrorKind::CoinBalanceOverflow),
                    "VALIDATOR_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::ValidatorNotFound),
                    "ENCODER_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::EncoderNotFound),
                    "STAKING_POOL_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::StakingPoolNotFound),
                    "CANNOT_REPORT_ONESELF" => Ok(execution_error::ExecutionErrorKind::CannotReportOneself),
                    "REPORT_RECORD_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::ReportRecordNotFound),
                    "OTHER_ERROR" => Ok(execution_error::ExecutionErrorKind::OtherError),
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
impl serde::Serialize for GetCommitRequest {
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
        if self.commit_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetCommitRequest", len)?;
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        if let Some(v) = self.commit_id.as_ref() {
            match v {
                get_commit_request::CommitId::Index(v) => {
                    struct_ser.serialize_field("index", v)?;
                }
                get_commit_request::CommitId::Digest(v) => {
                    struct_ser.serialize_field("digest", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetCommitRequest {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "read_mask",
            "readMask",
            "index",
            "digest",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ReadMask,
            Index,
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
                            "index" => Ok(GeneratedField::Index),
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
            type Value = GetCommitRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetCommitRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetCommitRequest, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut read_mask__ = None;
                let mut commit_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ReadMask => {
                            if read_mask__.is_some() {
                                return Err(serde::de::Error::duplicate_field("readMask"));
                            }
                            read_mask__ = map_.next_value::<::std::option::Option<crate::utils::_serde::FieldMaskDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::Index => {
                            if commit_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("index"));
                            }
                            commit_id__ = map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| get_commit_request::CommitId::Index(x.0));
                        }
                        GeneratedField::Digest => {
                            if commit_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            commit_id__ = map_.next_value::<::std::option::Option<_>>()?.map(get_commit_request::CommitId::Digest);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetCommitRequest {
                    read_mask: read_mask__,
                    commit_id: commit_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetCommitRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GetCommitResponse {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.commit.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GetCommitResponse", len)?;
        if let Some(v) = self.commit.as_ref() {
            struct_ser.serialize_field("commit", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GetCommitResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "commit",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Commit,
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
                            "commit" => Ok(GeneratedField::Commit),
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
            type Value = GetCommitResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GetCommitResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GetCommitResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut commit__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Commit => {
                            if commit__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commit"));
                            }
                            commit__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GetCommitResponse {
                    commit: commit__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GetCommitResponse", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for PendingRemoval {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.is_consensus.is_some() {
            len += 1;
        }
        if self.index.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.PendingRemoval", len)?;
        if let Some(v) = self.is_consensus.as_ref() {
            struct_ser.serialize_field("isConsensus", v)?;
        }
        if let Some(v) = self.index.as_ref() {
            struct_ser.serialize_field("index", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for PendingRemoval {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "is_consensus",
            "isConsensus",
            "index",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            IsConsensus,
            Index,
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
                            "isConsensus" | "is_consensus" => Ok(GeneratedField::IsConsensus),
                            "index" => Ok(GeneratedField::Index),
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
            type Value = PendingRemoval;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.PendingRemoval")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<PendingRemoval, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut is_consensus__ = None;
                let mut index__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::IsConsensus => {
                            if is_consensus__.is_some() {
                                return Err(serde::de::Error::duplicate_field("isConsensus"));
                            }
                            is_consensus__ = map_.next_value()?;
                        }
                        GeneratedField::Index => {
                            if index__.is_some() {
                                return Err(serde::de::Error::duplicate_field("index"));
                            }
                            index__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(PendingRemoval {
                    is_consensus: is_consensus__,
                    index: index__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.PendingRemoval", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for RemoveEncoder {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.encoder_pubkey_bytes.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.RemoveEncoder", len)?;
        if let Some(v) = self.encoder_pubkey_bytes.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("encoderPubkeyBytes", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for RemoveEncoder {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "encoder_pubkey_bytes",
            "encoderPubkeyBytes",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            EncoderPubkeyBytes,
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
                            "encoderPubkeyBytes" | "encoder_pubkey_bytes" => Ok(GeneratedField::EncoderPubkeyBytes),
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
            type Value = RemoveEncoder;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.RemoveEncoder")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<RemoveEncoder, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut encoder_pubkey_bytes__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::EncoderPubkeyBytes => {
                            if encoder_pubkey_bytes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("encoderPubkeyBytes"));
                            }
                            encoder_pubkey_bytes__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(RemoveEncoder {
                    encoder_pubkey_bytes: encoder_pubkey_bytes__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.RemoveEncoder", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for ReportEncoder {
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
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ReportEncoder", len)?;
        if let Some(v) = self.reportee.as_ref() {
            struct_ser.serialize_field("reportee", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ReportEncoder {
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
            type Value = ReportEncoder;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ReportEncoder")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ReportEncoder, V::Error>
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
                Ok(ReportEncoder {
                    reportee: reportee__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ReportEncoder", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for ReportWinner {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.shard_input_ref.is_some() {
            len += 1;
        }
        if self.signed_report.is_some() {
            len += 1;
        }
        if self.encoder_aggregate_signature.is_some() {
            len += 1;
        }
        if !self.signers.is_empty() {
            len += 1;
        }
        if self.shard_auth_token.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ReportWinner", len)?;
        if let Some(v) = self.shard_input_ref.as_ref() {
            struct_ser.serialize_field("shardInputRef", v)?;
        }
        if let Some(v) = self.signed_report.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("signedReport", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.encoder_aggregate_signature.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("encoderAggregateSignature", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if !self.signers.is_empty() {
            struct_ser.serialize_field("signers", &self.signers)?;
        }
        if let Some(v) = self.shard_auth_token.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("shardAuthToken", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ReportWinner {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "shard_input_ref",
            "shardInputRef",
            "signed_report",
            "signedReport",
            "encoder_aggregate_signature",
            "encoderAggregateSignature",
            "signers",
            "shard_auth_token",
            "shardAuthToken",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ShardInputRef,
            SignedReport,
            EncoderAggregateSignature,
            Signers,
            ShardAuthToken,
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
                            "shardInputRef" | "shard_input_ref" => Ok(GeneratedField::ShardInputRef),
                            "signedReport" | "signed_report" => Ok(GeneratedField::SignedReport),
                            "encoderAggregateSignature" | "encoder_aggregate_signature" => Ok(GeneratedField::EncoderAggregateSignature),
                            "signers" => Ok(GeneratedField::Signers),
                            "shardAuthToken" | "shard_auth_token" => Ok(GeneratedField::ShardAuthToken),
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
            type Value = ReportWinner;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ReportWinner")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ReportWinner, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut shard_input_ref__ = None;
                let mut signed_report__ = None;
                let mut encoder_aggregate_signature__ = None;
                let mut signers__ = None;
                let mut shard_auth_token__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ShardInputRef => {
                            if shard_input_ref__.is_some() {
                                return Err(serde::de::Error::duplicate_field("shardInputRef"));
                            }
                            shard_input_ref__ = map_.next_value()?;
                        }
                        GeneratedField::SignedReport => {
                            if signed_report__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signedReport"));
                            }
                            signed_report__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::EncoderAggregateSignature => {
                            if encoder_aggregate_signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("encoderAggregateSignature"));
                            }
                            encoder_aggregate_signature__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Signers => {
                            if signers__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signers"));
                            }
                            signers__ = Some(map_.next_value()?);
                        }
                        GeneratedField::ShardAuthToken => {
                            if shard_auth_token__.is_some() {
                                return Err(serde::de::Error::duplicate_field("shardAuthToken"));
                            }
                            shard_auth_token__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ReportWinner {
                    shard_input_ref: shard_input_ref__,
                    signed_report: signed_report__,
                    encoder_aggregate_signature: encoder_aggregate_signature__,
                    signers: signers__.unwrap_or_default(),
                    shard_auth_token: shard_auth_token__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ReportWinner", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for SetEncoderBytePrice {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.new_price.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SetEncoderBytePrice", len)?;
        if let Some(v) = self.new_price.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("newPrice", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SetEncoderBytePrice {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "new_price",
            "newPrice",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            NewPrice,
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
                            "newPrice" | "new_price" => Ok(GeneratedField::NewPrice),
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
            type Value = SetEncoderBytePrice;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SetEncoderBytePrice")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SetEncoderBytePrice, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut new_price__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::NewPrice => {
                            if new_price__.is_some() {
                                return Err(serde::de::Error::duplicate_field("newPrice"));
                            }
                            new_price__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SetEncoderBytePrice {
                    new_price: new_price__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SetEncoderBytePrice", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SetEncoderCommissionRate {
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
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SetEncoderCommissionRate", len)?;
        if let Some(v) = self.new_rate.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("newRate", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SetEncoderCommissionRate {
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
            type Value = SetEncoderCommissionRate;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SetEncoderCommissionRate")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SetEncoderCommissionRate, V::Error>
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
                Ok(SetEncoderCommissionRate {
                    new_rate: new_rate__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SetEncoderCommissionRate", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Shard {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.quorum_threshold.is_some() {
            len += 1;
        }
        if !self.encoders.is_empty() {
            len += 1;
        }
        if self.seed.is_some() {
            len += 1;
        }
        if self.epoch.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Shard", len)?;
        if let Some(v) = self.quorum_threshold.as_ref() {
            struct_ser.serialize_field("quorumThreshold", v)?;
        }
        if !self.encoders.is_empty() {
            struct_ser.serialize_field("encoders", &self.encoders)?;
        }
        if let Some(v) = self.seed.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("seed", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Shard {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "quorum_threshold",
            "quorumThreshold",
            "encoders",
            "seed",
            "epoch",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            QuorumThreshold,
            Encoders,
            Seed,
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
                            "quorumThreshold" | "quorum_threshold" => Ok(GeneratedField::QuorumThreshold),
                            "encoders" => Ok(GeneratedField::Encoders),
                            "seed" => Ok(GeneratedField::Seed),
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
            type Value = Shard;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Shard")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Shard, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut quorum_threshold__ = None;
                let mut encoders__ = None;
                let mut seed__ = None;
                let mut epoch__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::QuorumThreshold => {
                            if quorum_threshold__.is_some() {
                                return Err(serde::de::Error::duplicate_field("quorumThreshold"));
                            }
                            quorum_threshold__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Encoders => {
                            if encoders__.is_some() {
                                return Err(serde::de::Error::duplicate_field("encoders"));
                            }
                            encoders__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Seed => {
                            if seed__.is_some() {
                                return Err(serde::de::Error::duplicate_field("seed"));
                            }
                            seed__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Epoch => {
                            if epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epoch"));
                            }
                            epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Shard {
                    quorum_threshold: quorum_threshold__,
                    encoders: encoders__.unwrap_or_default(),
                    seed: seed__,
                    epoch: epoch__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Shard", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ShardResult {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.metadata.is_some() {
            len += 1;
        }
        if self.amount.is_some() {
            len += 1;
        }
        if self.report.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ShardResult", len)?;
        if let Some(v) = self.metadata.as_ref() {
            struct_ser.serialize_field("metadata", v)?;
        }
        if let Some(v) = self.amount.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("amount", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.report.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("report", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ShardResult {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "metadata",
            "amount",
            "report",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Metadata,
            Amount,
            Report,
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
                            "metadata" => Ok(GeneratedField::Metadata),
                            "amount" => Ok(GeneratedField::Amount),
                            "report" => Ok(GeneratedField::Report),
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
            type Value = ShardResult;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ShardResult")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ShardResult, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut metadata__ = None;
                let mut amount__ = None;
                let mut report__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Metadata => {
                            if metadata__.is_some() {
                                return Err(serde::de::Error::duplicate_field("metadata"));
                            }
                            metadata__ = map_.next_value()?;
                        }
                        GeneratedField::Amount => {
                            if amount__.is_some() {
                                return Err(serde::de::Error::duplicate_field("amount"));
                            }
                            amount__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Report => {
                            if report__.is_some() {
                                return Err(serde::de::Error::duplicate_field("report"));
                            }
                            report__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ShardResult {
                    metadata: metadata__,
                    amount: amount__,
                    report: report__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ShardResult", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for StakeSubsidy {
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
        if self.distribution_counter.is_some() {
            len += 1;
        }
        if self.current_distribution_amount.is_some() {
            len += 1;
        }
        if self.period_length.is_some() {
            len += 1;
        }
        if self.decrease_rate.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.StakeSubsidy", len)?;
        if let Some(v) = self.balance.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("balance", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.distribution_counter.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("distributionCounter", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.current_distribution_amount.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("currentDistributionAmount", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.period_length.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("periodLength", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.decrease_rate.as_ref() {
            struct_ser.serialize_field("decreaseRate", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for StakeSubsidy {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "balance",
            "distribution_counter",
            "distributionCounter",
            "current_distribution_amount",
            "currentDistributionAmount",
            "period_length",
            "periodLength",
            "decrease_rate",
            "decreaseRate",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Balance,
            DistributionCounter,
            CurrentDistributionAmount,
            PeriodLength,
            DecreaseRate,
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
                            "distributionCounter" | "distribution_counter" => Ok(GeneratedField::DistributionCounter),
                            "currentDistributionAmount" | "current_distribution_amount" => Ok(GeneratedField::CurrentDistributionAmount),
                            "periodLength" | "period_length" => Ok(GeneratedField::PeriodLength),
                            "decreaseRate" | "decrease_rate" => Ok(GeneratedField::DecreaseRate),
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
            type Value = StakeSubsidy;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.StakeSubsidy")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<StakeSubsidy, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut balance__ = None;
                let mut distribution_counter__ = None;
                let mut current_distribution_amount__ = None;
                let mut period_length__ = None;
                let mut decrease_rate__ = None;
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
                        GeneratedField::DistributionCounter => {
                            if distribution_counter__.is_some() {
                                return Err(serde::de::Error::duplicate_field("distributionCounter"));
                            }
                            distribution_counter__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::CurrentDistributionAmount => {
                            if current_distribution_amount__.is_some() {
                                return Err(serde::de::Error::duplicate_field("currentDistributionAmount"));
                            }
                            current_distribution_amount__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::PeriodLength => {
                            if period_length__.is_some() {
                                return Err(serde::de::Error::duplicate_field("periodLength"));
                            }
                            period_length__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::DecreaseRate => {
                            if decrease_rate__.is_some() {
                                return Err(serde::de::Error::duplicate_field("decreaseRate"));
                            }
                            decrease_rate__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(StakeSubsidy {
                    balance: balance__,
                    distribution_counter: distribution_counter__,
                    current_distribution_amount: current_distribution_amount__,
                    period_length: period_length__,
                    decrease_rate: decrease_rate__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.StakeSubsidy", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for SubscribeCommitsRequest {
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
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SubscribeCommitsRequest", len)?;
        if let Some(v) = self.read_mask.as_ref() {
            struct_ser.serialize_field("readMask", &crate::utils::_serde::FieldMaskSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SubscribeCommitsRequest {
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
            type Value = SubscribeCommitsRequest;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SubscribeCommitsRequest")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SubscribeCommitsRequest, V::Error>
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
                Ok(SubscribeCommitsRequest {
                    read_mask: read_mask__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SubscribeCommitsRequest", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SubscribeCommitsResponse {
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
        if self.commit.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SubscribeCommitsResponse", len)?;
        if let Some(v) = self.cursor.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("cursor", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.commit.as_ref() {
            struct_ser.serialize_field("commit", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SubscribeCommitsResponse {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "cursor",
            "commit",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Cursor,
            Commit,
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
                            "commit" => Ok(GeneratedField::Commit),
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
            type Value = SubscribeCommitsResponse;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SubscribeCommitsResponse")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SubscribeCommitsResponse, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut cursor__ = None;
                let mut commit__ = None;
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
                        GeneratedField::Commit => {
                            if commit__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commit"));
                            }
                            commit__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SubscribeCommitsResponse {
                    cursor: cursor__,
                    commit: commit__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SubscribeCommitsResponse", FIELDS, GeneratedVisitor)
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
        if self.vdf_iterations.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SystemParameters", len)?;
        if let Some(v) = self.epoch_duration_ms.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epochDurationMs", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.vdf_iterations.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("vdfIterations", ToString::to_string(&v).as_str())?;
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
            "vdf_iterations",
            "vdfIterations",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            EpochDurationMs,
            VdfIterations,
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
                            "vdfIterations" | "vdf_iterations" => Ok(GeneratedField::VdfIterations),
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
                let mut vdf_iterations__ = None;
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
                        GeneratedField::VdfIterations => {
                            if vdf_iterations__.is_some() {
                                return Err(serde::de::Error::duplicate_field("vdfIterations"));
                            }
                            vdf_iterations__ = 
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
                    vdf_iterations: vdf_iterations__,
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
        if self.parameters.is_some() {
            len += 1;
        }
        if self.validators.is_some() {
            len += 1;
        }
        if self.encoders.is_some() {
            len += 1;
        }
        if !self.validator_report_records.is_empty() {
            len += 1;
        }
        if !self.encoder_report_records.is_empty() {
            len += 1;
        }
        if self.stake_subsidy.is_some() {
            len += 1;
        }
        if !self.shard_results.is_empty() {
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
        if let Some(v) = self.parameters.as_ref() {
            struct_ser.serialize_field("parameters", v)?;
        }
        if let Some(v) = self.validators.as_ref() {
            struct_ser.serialize_field("validators", v)?;
        }
        if let Some(v) = self.encoders.as_ref() {
            struct_ser.serialize_field("encoders", v)?;
        }
        if !self.validator_report_records.is_empty() {
            struct_ser.serialize_field("validatorReportRecords", &self.validator_report_records)?;
        }
        if !self.encoder_report_records.is_empty() {
            struct_ser.serialize_field("encoderReportRecords", &self.encoder_report_records)?;
        }
        if let Some(v) = self.stake_subsidy.as_ref() {
            struct_ser.serialize_field("stakeSubsidy", v)?;
        }
        if !self.shard_results.is_empty() {
            struct_ser.serialize_field("shardResults", &self.shard_results)?;
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
            "parameters",
            "validators",
            "encoders",
            "validator_report_records",
            "validatorReportRecords",
            "encoder_report_records",
            "encoderReportRecords",
            "stake_subsidy",
            "stakeSubsidy",
            "shard_results",
            "shardResults",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            EpochStartTimestampMs,
            Parameters,
            Validators,
            Encoders,
            ValidatorReportRecords,
            EncoderReportRecords,
            StakeSubsidy,
            ShardResults,
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
                            "parameters" => Ok(GeneratedField::Parameters),
                            "validators" => Ok(GeneratedField::Validators),
                            "encoders" => Ok(GeneratedField::Encoders),
                            "validatorReportRecords" | "validator_report_records" => Ok(GeneratedField::ValidatorReportRecords),
                            "encoderReportRecords" | "encoder_report_records" => Ok(GeneratedField::EncoderReportRecords),
                            "stakeSubsidy" | "stake_subsidy" => Ok(GeneratedField::StakeSubsidy),
                            "shardResults" | "shard_results" => Ok(GeneratedField::ShardResults),
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
                let mut parameters__ = None;
                let mut validators__ = None;
                let mut encoders__ = None;
                let mut validator_report_records__ = None;
                let mut encoder_report_records__ = None;
                let mut stake_subsidy__ = None;
                let mut shard_results__ = None;
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
                        GeneratedField::Encoders => {
                            if encoders__.is_some() {
                                return Err(serde::de::Error::duplicate_field("encoders"));
                            }
                            encoders__ = map_.next_value()?;
                        }
                        GeneratedField::ValidatorReportRecords => {
                            if validator_report_records__.is_some() {
                                return Err(serde::de::Error::duplicate_field("validatorReportRecords"));
                            }
                            validator_report_records__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::EncoderReportRecords => {
                            if encoder_report_records__.is_some() {
                                return Err(serde::de::Error::duplicate_field("encoderReportRecords"));
                            }
                            encoder_report_records__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::StakeSubsidy => {
                            if stake_subsidy__.is_some() {
                                return Err(serde::de::Error::duplicate_field("stakeSubsidy"));
                            }
                            stake_subsidy__ = map_.next_value()?;
                        }
                        GeneratedField::ShardResults => {
                            if shard_results__.is_some() {
                                return Err(serde::de::Error::duplicate_field("shardResults"));
                            }
                            shard_results__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SystemState {
                    epoch: epoch__,
                    epoch_start_timestamp_ms: epoch_start_timestamp_ms__,
                    parameters: parameters__,
                    validators: validators__,
                    encoders: encoders__,
                    validator_report_records: validator_report_records__.unwrap_or_default(),
                    encoder_report_records: encoder_report_records__.unwrap_or_default(),
                    stake_subsidy: stake_subsidy__,
                    shard_results: shard_results__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SystemState", FIELDS, GeneratedVisitor)
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
        if self.gas_object_ref.is_some() {
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
        if let Some(v) = self.gas_object_ref.as_ref() {
            struct_ser.serialize_field("gasObjectRef", v)?;
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
            "gas_object_ref",
            "gasObjectRef",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            BaseFee,
            OperationFee,
            ValueFee,
            TotalFee,
            GasObjectRef,
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
                            "gasObjectRef" | "gas_object_ref" => Ok(GeneratedField::GasObjectRef),
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
                let mut gas_object_ref__ = None;
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
                        GeneratedField::GasObjectRef => {
                            if gas_object_ref__.is_some() {
                                return Err(serde::de::Error::duplicate_field("gasObjectRef"));
                            }
                            gas_object_ref__ = map_.next_value()?;
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
                    gas_object_ref: gas_object_ref__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TransactionFee", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TransactionFinality {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.finality.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TransactionFinality", len)?;
        if let Some(v) = self.finality.as_ref() {
            match v {
                transaction_finality::Finality::Certified(v) => {
                    struct_ser.serialize_field("certified", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TransactionFinality {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "certified",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Certified,
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
                            "certified" => Ok(GeneratedField::Certified),
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
            type Value = TransactionFinality;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TransactionFinality")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TransactionFinality, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut finality__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Certified => {
                            if finality__.is_some() {
                                return Err(serde::de::Error::duplicate_field("certified"));
                            }
                            finality__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_finality::Finality::Certified)
;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TransactionFinality {
                    finality: finality__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TransactionFinality", FIELDS, GeneratedVisitor)
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
                transaction_kind::Kind::AddEncoder(v) => {
                    struct_ser.serialize_field("addEncoder", v)?;
                }
                transaction_kind::Kind::RemoveEncoder(v) => {
                    struct_ser.serialize_field("removeEncoder", v)?;
                }
                transaction_kind::Kind::ReportEncoder(v) => {
                    struct_ser.serialize_field("reportEncoder", v)?;
                }
                transaction_kind::Kind::UndoReportEncoder(v) => {
                    struct_ser.serialize_field("undoReportEncoder", v)?;
                }
                transaction_kind::Kind::UpdateEncoderMetadata(v) => {
                    struct_ser.serialize_field("updateEncoderMetadata", v)?;
                }
                transaction_kind::Kind::SetEncoderCommissionRate(v) => {
                    struct_ser.serialize_field("setEncoderCommissionRate", v)?;
                }
                transaction_kind::Kind::SetEncoderBytePrice(v) => {
                    struct_ser.serialize_field("setEncoderBytePrice", v)?;
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
                transaction_kind::Kind::AddStakeToEncoder(v) => {
                    struct_ser.serialize_field("addStakeToEncoder", v)?;
                }
                transaction_kind::Kind::WithdrawStake(v) => {
                    struct_ser.serialize_field("withdrawStake", v)?;
                }
                transaction_kind::Kind::EmbedData(v) => {
                    struct_ser.serialize_field("embedData", v)?;
                }
                transaction_kind::Kind::ClaimEscrow(v) => {
                    struct_ser.serialize_field("claimEscrow", v)?;
                }
                transaction_kind::Kind::ReportWinner(v) => {
                    struct_ser.serialize_field("reportWinner", v)?;
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
            "add_encoder",
            "addEncoder",
            "remove_encoder",
            "removeEncoder",
            "report_encoder",
            "reportEncoder",
            "undo_report_encoder",
            "undoReportEncoder",
            "update_encoder_metadata",
            "updateEncoderMetadata",
            "set_encoder_commission_rate",
            "setEncoderCommissionRate",
            "set_encoder_byte_price",
            "setEncoderBytePrice",
            "transfer_coin",
            "transferCoin",
            "pay_coins",
            "payCoins",
            "transfer_objects",
            "transferObjects",
            "add_stake",
            "addStake",
            "add_stake_to_encoder",
            "addStakeToEncoder",
            "withdraw_stake",
            "withdrawStake",
            "embed_data",
            "embedData",
            "claim_escrow",
            "claimEscrow",
            "report_winner",
            "reportWinner",
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
            AddEncoder,
            RemoveEncoder,
            ReportEncoder,
            UndoReportEncoder,
            UpdateEncoderMetadata,
            SetEncoderCommissionRate,
            SetEncoderBytePrice,
            TransferCoin,
            PayCoins,
            TransferObjects,
            AddStake,
            AddStakeToEncoder,
            WithdrawStake,
            EmbedData,
            ClaimEscrow,
            ReportWinner,
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
                            "addEncoder" | "add_encoder" => Ok(GeneratedField::AddEncoder),
                            "removeEncoder" | "remove_encoder" => Ok(GeneratedField::RemoveEncoder),
                            "reportEncoder" | "report_encoder" => Ok(GeneratedField::ReportEncoder),
                            "undoReportEncoder" | "undo_report_encoder" => Ok(GeneratedField::UndoReportEncoder),
                            "updateEncoderMetadata" | "update_encoder_metadata" => Ok(GeneratedField::UpdateEncoderMetadata),
                            "setEncoderCommissionRate" | "set_encoder_commission_rate" => Ok(GeneratedField::SetEncoderCommissionRate),
                            "setEncoderBytePrice" | "set_encoder_byte_price" => Ok(GeneratedField::SetEncoderBytePrice),
                            "transferCoin" | "transfer_coin" => Ok(GeneratedField::TransferCoin),
                            "payCoins" | "pay_coins" => Ok(GeneratedField::PayCoins),
                            "transferObjects" | "transfer_objects" => Ok(GeneratedField::TransferObjects),
                            "addStake" | "add_stake" => Ok(GeneratedField::AddStake),
                            "addStakeToEncoder" | "add_stake_to_encoder" => Ok(GeneratedField::AddStakeToEncoder),
                            "withdrawStake" | "withdraw_stake" => Ok(GeneratedField::WithdrawStake),
                            "embedData" | "embed_data" => Ok(GeneratedField::EmbedData),
                            "claimEscrow" | "claim_escrow" => Ok(GeneratedField::ClaimEscrow),
                            "reportWinner" | "report_winner" => Ok(GeneratedField::ReportWinner),
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
                        GeneratedField::AddEncoder => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("addEncoder"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::AddEncoder)
;
                        }
                        GeneratedField::RemoveEncoder => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("removeEncoder"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::RemoveEncoder)
;
                        }
                        GeneratedField::ReportEncoder => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reportEncoder"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ReportEncoder)
;
                        }
                        GeneratedField::UndoReportEncoder => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("undoReportEncoder"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::UndoReportEncoder)
;
                        }
                        GeneratedField::UpdateEncoderMetadata => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("updateEncoderMetadata"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::UpdateEncoderMetadata)
;
                        }
                        GeneratedField::SetEncoderCommissionRate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("setEncoderCommissionRate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::SetEncoderCommissionRate)
;
                        }
                        GeneratedField::SetEncoderBytePrice => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("setEncoderBytePrice"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::SetEncoderBytePrice)
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
                        GeneratedField::AddStakeToEncoder => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("addStakeToEncoder"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::AddStakeToEncoder)
;
                        }
                        GeneratedField::WithdrawStake => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("withdrawStake"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::WithdrawStake)
;
                        }
                        GeneratedField::EmbedData => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("embedData"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::EmbedData)
;
                        }
                        GeneratedField::ClaimEscrow => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("claimEscrow"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ClaimEscrow)
;
                        }
                        GeneratedField::ReportWinner => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reportWinner"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ReportWinner)
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
impl serde::Serialize for UndoReportEncoder {
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
        let mut struct_ser = serializer.serialize_struct("soma.rpc.UndoReportEncoder", len)?;
        if let Some(v) = self.reportee.as_ref() {
            struct_ser.serialize_field("reportee", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for UndoReportEncoder {
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
            type Value = UndoReportEncoder;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.UndoReportEncoder")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<UndoReportEncoder, V::Error>
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
                Ok(UndoReportEncoder {
                    reportee: reportee__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.UndoReportEncoder", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for UpdateEncoderMetadata {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.next_epoch_external_network_address.is_some() {
            len += 1;
        }
        if self.next_epoch_internal_network_address.is_some() {
            len += 1;
        }
        if self.next_epoch_network_pubkey.is_some() {
            len += 1;
        }
        if self.next_epoch_object_server_address.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.UpdateEncoderMetadata", len)?;
        if let Some(v) = self.next_epoch_external_network_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochExternalNetworkAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_internal_network_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochInternalNetworkAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_network_pubkey.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochNetworkPubkey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.next_epoch_object_server_address.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nextEpochObjectServerAddress", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for UpdateEncoderMetadata {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "next_epoch_external_network_address",
            "nextEpochExternalNetworkAddress",
            "next_epoch_internal_network_address",
            "nextEpochInternalNetworkAddress",
            "next_epoch_network_pubkey",
            "nextEpochNetworkPubkey",
            "next_epoch_object_server_address",
            "nextEpochObjectServerAddress",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            NextEpochExternalNetworkAddress,
            NextEpochInternalNetworkAddress,
            NextEpochNetworkPubkey,
            NextEpochObjectServerAddress,
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
                            "nextEpochExternalNetworkAddress" | "next_epoch_external_network_address" => Ok(GeneratedField::NextEpochExternalNetworkAddress),
                            "nextEpochInternalNetworkAddress" | "next_epoch_internal_network_address" => Ok(GeneratedField::NextEpochInternalNetworkAddress),
                            "nextEpochNetworkPubkey" | "next_epoch_network_pubkey" => Ok(GeneratedField::NextEpochNetworkPubkey),
                            "nextEpochObjectServerAddress" | "next_epoch_object_server_address" => Ok(GeneratedField::NextEpochObjectServerAddress),
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
            type Value = UpdateEncoderMetadata;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.UpdateEncoderMetadata")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<UpdateEncoderMetadata, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut next_epoch_external_network_address__ = None;
                let mut next_epoch_internal_network_address__ = None;
                let mut next_epoch_network_pubkey__ = None;
                let mut next_epoch_object_server_address__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::NextEpochExternalNetworkAddress => {
                            if next_epoch_external_network_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochExternalNetworkAddress"));
                            }
                            next_epoch_external_network_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NextEpochInternalNetworkAddress => {
                            if next_epoch_internal_network_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochInternalNetworkAddress"));
                            }
                            next_epoch_internal_network_address__ = 
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
                        GeneratedField::NextEpochObjectServerAddress => {
                            if next_epoch_object_server_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochObjectServerAddress"));
                            }
                            next_epoch_object_server_address__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(UpdateEncoderMetadata {
                    next_epoch_external_network_address: next_epoch_external_network_address__,
                    next_epoch_internal_network_address: next_epoch_internal_network_address__,
                    next_epoch_network_pubkey: next_epoch_network_pubkey__,
                    next_epoch_object_server_address: next_epoch_object_server_address__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.UpdateEncoderMetadata", FIELDS, GeneratedVisitor)
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
        if self.next_epoch_protocol_pubkey.is_some() {
            len += 1;
        }
        if self.next_epoch_worker_pubkey.is_some() {
            len += 1;
        }
        if self.next_epoch_network_pubkey.is_some() {
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
            "next_epoch_protocol_pubkey",
            "nextEpochProtocolPubkey",
            "next_epoch_worker_pubkey",
            "nextEpochWorkerPubkey",
            "next_epoch_network_pubkey",
            "nextEpochNetworkPubkey",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            NextEpochNetworkAddress,
            NextEpochP2pAddress,
            NextEpochPrimaryAddress,
            NextEpochProtocolPubkey,
            NextEpochWorkerPubkey,
            NextEpochNetworkPubkey,
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
                            "nextEpochProtocolPubkey" | "next_epoch_protocol_pubkey" => Ok(GeneratedField::NextEpochProtocolPubkey),
                            "nextEpochWorkerPubkey" | "next_epoch_worker_pubkey" => Ok(GeneratedField::NextEpochWorkerPubkey),
                            "nextEpochNetworkPubkey" | "next_epoch_network_pubkey" => Ok(GeneratedField::NextEpochNetworkPubkey),
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
                let mut next_epoch_protocol_pubkey__ = None;
                let mut next_epoch_worker_pubkey__ = None;
                let mut next_epoch_network_pubkey__ = None;
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
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(UpdateValidatorMetadata {
                    next_epoch_network_address: next_epoch_network_address__,
                    next_epoch_p2p_address: next_epoch_p2p_address__,
                    next_epoch_primary_address: next_epoch_primary_address__,
                    next_epoch_protocol_pubkey: next_epoch_protocol_pubkey__,
                    next_epoch_worker_pubkey: next_epoch_worker_pubkey__,
                    next_epoch_network_pubkey: next_epoch_network_pubkey__,
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
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Scheme,
            Simple,
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
        if self.encoder_validator_address.is_some() {
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
        if self.next_epoch_encoder_validator_address.is_some() {
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
        if let Some(v) = self.encoder_validator_address.as_ref() {
            struct_ser.serialize_field("encoderValidatorAddress", v)?;
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
        if let Some(v) = self.next_epoch_encoder_validator_address.as_ref() {
            struct_ser.serialize_field("nextEpochEncoderValidatorAddress", v)?;
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
            "encoder_validator_address",
            "encoderValidatorAddress",
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
            "next_epoch_encoder_validator_address",
            "nextEpochEncoderValidatorAddress",
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
            EncoderValidatorAddress,
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
            NextEpochEncoderValidatorAddress,
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
                            "encoderValidatorAddress" | "encoder_validator_address" => Ok(GeneratedField::EncoderValidatorAddress),
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
                            "nextEpochEncoderValidatorAddress" | "next_epoch_encoder_validator_address" => Ok(GeneratedField::NextEpochEncoderValidatorAddress),
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
                let mut encoder_validator_address__ = None;
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
                let mut next_epoch_encoder_validator_address__ = None;
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
                        GeneratedField::EncoderValidatorAddress => {
                            if encoder_validator_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("encoderValidatorAddress"));
                            }
                            encoder_validator_address__ = map_.next_value()?;
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
                        GeneratedField::NextEpochEncoderValidatorAddress => {
                            if next_epoch_encoder_validator_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochEncoderValidatorAddress"));
                            }
                            next_epoch_encoder_validator_address__ = map_.next_value()?;
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
                    encoder_validator_address: encoder_validator_address__,
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
                    next_epoch_encoder_validator_address: next_epoch_encoder_validator_address__,
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
        if !self.consensus_validators.is_empty() {
            len += 1;
        }
        if !self.networking_validators.is_empty() {
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
        if !self.consensus_validators.is_empty() {
            struct_ser.serialize_field("consensusValidators", &self.consensus_validators)?;
        }
        if !self.networking_validators.is_empty() {
            struct_ser.serialize_field("networkingValidators", &self.networking_validators)?;
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
            "consensus_validators",
            "consensusValidators",
            "networking_validators",
            "networkingValidators",
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
            ConsensusValidators,
            NetworkingValidators,
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
                            "consensusValidators" | "consensus_validators" => Ok(GeneratedField::ConsensusValidators),
                            "networkingValidators" | "networking_validators" => Ok(GeneratedField::NetworkingValidators),
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
                let mut consensus_validators__ = None;
                let mut networking_validators__ = None;
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
                        GeneratedField::ConsensusValidators => {
                            if consensus_validators__.is_some() {
                                return Err(serde::de::Error::duplicate_field("consensusValidators"));
                            }
                            consensus_validators__ = Some(map_.next_value()?);
                        }
                        GeneratedField::NetworkingValidators => {
                            if networking_validators__.is_some() {
                                return Err(serde::de::Error::duplicate_field("networkingValidators"));
                            }
                            networking_validators__ = Some(map_.next_value()?);
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
                            pending_removals__ = Some(map_.next_value()?);
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
                    consensus_validators: consensus_validators__.unwrap_or_default(),
                    networking_validators: networking_validators__.unwrap_or_default(),
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
