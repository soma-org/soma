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
            struct_ser.serialize_field("encoderPubkeyBytes", v)?;
        }
        if let Some(v) = self.network_pubkey_bytes.as_ref() {
            struct_ser.serialize_field("networkPubkeyBytes", v)?;
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
                            encoder_pubkey_bytes__ = map_.next_value()?;
                        }
                        GeneratedField::NetworkPubkeyBytes => {
                            if network_pubkey_bytes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("networkPubkeyBytes"));
                            }
                            network_pubkey_bytes__ = map_.next_value()?;
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
            struct_ser.serialize_field("pubkeyBytes", v)?;
        }
        if let Some(v) = self.network_pubkey_bytes.as_ref() {
            struct_ser.serialize_field("networkPubkeyBytes", v)?;
        }
        if let Some(v) = self.worker_pubkey_bytes.as_ref() {
            struct_ser.serialize_field("workerPubkeyBytes", v)?;
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
                            pubkey_bytes__ = map_.next_value()?;
                        }
                        GeneratedField::NetworkPubkeyBytes => {
                            if network_pubkey_bytes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("networkPubkeyBytes"));
                            }
                            network_pubkey_bytes__ = map_.next_value()?;
                        }
                        GeneratedField::WorkerPubkeyBytes => {
                            if worker_pubkey_bytes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("workerPubkeyBytes"));
                            }
                            worker_pubkey_bytes__ = map_.next_value()?;
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
impl serde::Serialize for Bcs {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.name.is_some() {
            len += 1;
        }
        if self.value.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Bcs", len)?;
        if let Some(v) = self.name.as_ref() {
            struct_ser.serialize_field("name", v)?;
        }
        if let Some(v) = self.value.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("value", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Bcs {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "name",
            "value",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Name,
            Value,
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
                            "name" => Ok(GeneratedField::Name),
                            "value" => Ok(GeneratedField::Value),
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
            type Value = Bcs;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Bcs")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Bcs, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut name__ = None;
                let mut value__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Name => {
                            if name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("name"));
                            }
                            name__ = map_.next_value()?;
                        }
                        GeneratedField::Value => {
                            if value__.is_some() {
                                return Err(serde::de::Error::duplicate_field("value"));
                            }
                            value__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Bcs {
                    name: name__,
                    value: value__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Bcs", FIELDS, GeneratedVisitor)
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
        if self.digest.is_some() {
            len += 1;
        }
        if self.data_size_bytes.is_some() {
            len += 1;
        }
        if self.coin_ref.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.EmbedData", len)?;
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.data_size_bytes.as_ref() {
            struct_ser.serialize_field("dataSizeBytes", v)?;
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
            "digest",
            "data_size_bytes",
            "dataSizeBytes",
            "coin_ref",
            "coinRef",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Digest,
            DataSizeBytes,
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
                            "digest" => Ok(GeneratedField::Digest),
                            "dataSizeBytes" | "data_size_bytes" => Ok(GeneratedField::DataSizeBytes),
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
                let mut digest__ = None;
                let mut data_size_bytes__ = None;
                let mut coin_ref__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::DataSizeBytes => {
                            if data_size_bytes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("dataSizeBytes"));
                            }
                            data_size_bytes__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
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
                    digest: digest__,
                    data_size_bytes: data_size_bytes__,
                    coin_ref: coin_ref__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.EmbedData", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for GasPayment {
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
        if self.owner.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GasPayment", len)?;
        if !self.objects.is_empty() {
            struct_ser.serialize_field("objects", &self.objects)?;
        }
        if let Some(v) = self.owner.as_ref() {
            struct_ser.serialize_field("owner", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GasPayment {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "objects",
            "owner",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Objects,
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
                            "objects" => Ok(GeneratedField::Objects),
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
            type Value = GasPayment;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GasPayment")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GasPayment, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut objects__ = None;
                let mut owner__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Objects => {
                            if objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objects"));
                            }
                            objects__ = Some(map_.next_value()?);
                        }
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
                Ok(GasPayment {
                    objects: objects__.unwrap_or_default(),
                    owner: owner__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GasPayment", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for Object {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.bcs.is_some() {
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
        if self.json.is_some() {
            len += 1;
        }
        if self.balance.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Object", len)?;
        if let Some(v) = self.bcs.as_ref() {
            struct_ser.serialize_field("bcs", v)?;
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
        if let Some(v) = self.owner.as_ref() {
            struct_ser.serialize_field("owner", v)?;
        }
        if let Some(v) = self.object_type.as_ref() {
            struct_ser.serialize_field("objectType", v)?;
        }
        if let Some(v) = self.contents.as_ref() {
            struct_ser.serialize_field("contents", v)?;
        }
        if let Some(v) = self.previous_transaction.as_ref() {
            struct_ser.serialize_field("previousTransaction", v)?;
        }
        if let Some(v) = self.json.as_ref() {
            struct_ser.serialize_field("json", &crate::utils::_serde::ValueSerializer(v))?;
        }
        if let Some(v) = self.balance.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("balance", ToString::to_string(&v).as_str())?;
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
            "bcs",
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
            "json",
            "balance",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Bcs,
            ObjectId,
            Version,
            Digest,
            Owner,
            ObjectType,
            Contents,
            PreviousTransaction,
            Json,
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
                            "bcs" => Ok(GeneratedField::Bcs),
                            "objectId" | "object_id" => Ok(GeneratedField::ObjectId),
                            "version" => Ok(GeneratedField::Version),
                            "digest" => Ok(GeneratedField::Digest),
                            "owner" => Ok(GeneratedField::Owner),
                            "objectType" | "object_type" => Ok(GeneratedField::ObjectType),
                            "contents" => Ok(GeneratedField::Contents),
                            "previousTransaction" | "previous_transaction" => Ok(GeneratedField::PreviousTransaction),
                            "json" => Ok(GeneratedField::Json),
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
            type Value = Object;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Object")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Object, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut bcs__ = None;
                let mut object_id__ = None;
                let mut version__ = None;
                let mut digest__ = None;
                let mut owner__ = None;
                let mut object_type__ = None;
                let mut contents__ = None;
                let mut previous_transaction__ = None;
                let mut json__ = None;
                let mut balance__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Bcs => {
                            if bcs__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bcs"));
                            }
                            bcs__ = map_.next_value()?;
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
                            contents__ = map_.next_value()?;
                        }
                        GeneratedField::PreviousTransaction => {
                            if previous_transaction__.is_some() {
                                return Err(serde::de::Error::duplicate_field("previousTransaction"));
                            }
                            previous_transaction__ = map_.next_value()?;
                        }
                        GeneratedField::Json => {
                            if json__.is_some() {
                                return Err(serde::de::Error::duplicate_field("json"));
                            }
                            json__ = map_.next_value::<::std::option::Option<crate::utils::_serde::ValueDeserializer>>()?.map(|x| x.0.into());
                        }
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
                Ok(Object {
                    bcs: bcs__,
                    object_id: object_id__,
                    version: version__,
                    digest: digest__,
                    owner: owner__,
                    object_type: object_type__,
                    contents: contents__,
                    previous_transaction: previous_transaction__,
                    json: json__,
                    balance: balance__,
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
            struct_ser.serialize_field("encoderPubkeyBytes", v)?;
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
                            encoder_pubkey_bytes__ = map_.next_value()?;
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
            struct_ser.serialize_field("pubkeyBytes", v)?;
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
                            pubkey_bytes__ = map_.next_value()?;
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
impl serde::Serialize for ReportScores {
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
        if self.scores.is_some() {
            len += 1;
        }
        if self.encoder_aggregate_signature.is_some() {
            len += 1;
        }
        if !self.signers.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ReportScores", len)?;
        if let Some(v) = self.shard_input_ref.as_ref() {
            struct_ser.serialize_field("shardInputRef", v)?;
        }
        if let Some(v) = self.scores.as_ref() {
            struct_ser.serialize_field("scores", v)?;
        }
        if let Some(v) = self.encoder_aggregate_signature.as_ref() {
            struct_ser.serialize_field("encoderAggregateSignature", v)?;
        }
        if !self.signers.is_empty() {
            struct_ser.serialize_field("signers", &self.signers)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ReportScores {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "shard_input_ref",
            "shardInputRef",
            "scores",
            "encoder_aggregate_signature",
            "encoderAggregateSignature",
            "signers",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ShardInputRef,
            Scores,
            EncoderAggregateSignature,
            Signers,
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
                            "scores" => Ok(GeneratedField::Scores),
                            "encoderAggregateSignature" | "encoder_aggregate_signature" => Ok(GeneratedField::EncoderAggregateSignature),
                            "signers" => Ok(GeneratedField::Signers),
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
            type Value = ReportScores;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ReportScores")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ReportScores, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut shard_input_ref__ = None;
                let mut scores__ = None;
                let mut encoder_aggregate_signature__ = None;
                let mut signers__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ShardInputRef => {
                            if shard_input_ref__.is_some() {
                                return Err(serde::de::Error::duplicate_field("shardInputRef"));
                            }
                            shard_input_ref__ = map_.next_value()?;
                        }
                        GeneratedField::Scores => {
                            if scores__.is_some() {
                                return Err(serde::de::Error::duplicate_field("scores"));
                            }
                            scores__ = map_.next_value()?;
                        }
                        GeneratedField::EncoderAggregateSignature => {
                            if encoder_aggregate_signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("encoderAggregateSignature"));
                            }
                            encoder_aggregate_signature__ = map_.next_value()?;
                        }
                        GeneratedField::Signers => {
                            if signers__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signers"));
                            }
                            signers__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ReportScores {
                    shard_input_ref: shard_input_ref__,
                    scores: scores__,
                    encoder_aggregate_signature: encoder_aggregate_signature__,
                    signers: signers__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ReportScores", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for Transaction {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.bcs.is_some() {
            len += 1;
        }
        if self.digest.is_some() {
            len += 1;
        }
        if self.kind.is_some() {
            len += 1;
        }
        if self.sender.is_some() {
            len += 1;
        }
        if self.gas_payment.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Transaction", len)?;
        if let Some(v) = self.bcs.as_ref() {
            struct_ser.serialize_field("bcs", v)?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.kind.as_ref() {
            struct_ser.serialize_field("kind", v)?;
        }
        if let Some(v) = self.sender.as_ref() {
            struct_ser.serialize_field("sender", v)?;
        }
        if let Some(v) = self.gas_payment.as_ref() {
            struct_ser.serialize_field("gasPayment", v)?;
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
            "bcs",
            "digest",
            "kind",
            "sender",
            "gas_payment",
            "gasPayment",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Bcs,
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
                            "bcs" => Ok(GeneratedField::Bcs),
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
                let mut bcs__ = None;
                let mut digest__ = None;
                let mut kind__ = None;
                let mut sender__ = None;
                let mut gas_payment__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Bcs => {
                            if bcs__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bcs"));
                            }
                            bcs__ = map_.next_value()?;
                        }
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
                            gas_payment__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Transaction {
                    bcs: bcs__,
                    digest: digest__,
                    kind: kind__,
                    sender: sender__,
                    gas_payment: gas_payment__,
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
        if self.bcs.is_some() {
            len += 1;
        }
        if self.digest.is_some() {
            len += 1;
        }
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
        if self.gas_object.is_some() {
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
        if let Some(v) = self.bcs.as_ref() {
            struct_ser.serialize_field("bcs", v)?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
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
        if let Some(v) = self.gas_object.as_ref() {
            struct_ser.serialize_field("gasObject", v)?;
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
            "bcs",
            "digest",
            "status",
            "epoch",
            "fee",
            "transaction_digest",
            "transactionDigest",
            "gas_object",
            "gasObject",
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
            Bcs,
            Digest,
            Status,
            Epoch,
            Fee,
            TransactionDigest,
            GasObject,
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
                            "bcs" => Ok(GeneratedField::Bcs),
                            "digest" => Ok(GeneratedField::Digest),
                            "status" => Ok(GeneratedField::Status),
                            "epoch" => Ok(GeneratedField::Epoch),
                            "fee" => Ok(GeneratedField::Fee),
                            "transactionDigest" | "transaction_digest" => Ok(GeneratedField::TransactionDigest),
                            "gasObject" | "gas_object" => Ok(GeneratedField::GasObject),
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
                let mut bcs__ = None;
                let mut digest__ = None;
                let mut status__ = None;
                let mut epoch__ = None;
                let mut fee__ = None;
                let mut transaction_digest__ = None;
                let mut gas_object__ = None;
                let mut dependencies__ = None;
                let mut lamport_version__ = None;
                let mut changed_objects__ = None;
                let mut unchanged_shared_objects__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Bcs => {
                            if bcs__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bcs"));
                            }
                            bcs__ = map_.next_value()?;
                        }
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
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
                        GeneratedField::GasObject => {
                            if gas_object__.is_some() {
                                return Err(serde::de::Error::duplicate_field("gasObject"));
                            }
                            gas_object__ = map_.next_value()?;
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
                    bcs: bcs__,
                    digest: digest__,
                    status: status__,
                    epoch: epoch__,
                    fee: fee__,
                    transaction_digest: transaction_digest__,
                    gas_object: gas_object__,
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
                transaction_kind::Kind::UndoValidatorMetadata(v) => {
                    struct_ser.serialize_field("undoValidatorMetadata", v)?;
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
                transaction_kind::Kind::ReportScores(v) => {
                    struct_ser.serialize_field("reportScores", v)?;
                }
                transaction_kind::Kind::ChangeEpoch(v) => {
                    struct_ser.serialize_field("changeEpoch", v)?;
                }
                transaction_kind::Kind::Genesis(v) => {
                    struct_ser.serialize_field("genesis", v)?;
                }
                transaction_kind::Kind::ConsensusCommitPrologue(v) => {
                    struct_ser.serialize_field("consensusCommitPrologue", v)?;
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
            "add_validator",
            "addValidator",
            "remove_validator",
            "removeValidator",
            "report_validator",
            "reportValidator",
            "undo_report_validator",
            "undoReportValidator",
            "undo_validator_metadata",
            "undoValidatorMetadata",
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
            "report_scores",
            "reportScores",
            "change_epoch",
            "changeEpoch",
            "genesis",
            "consensus_commit_prologue",
            "consensusCommitPrologue",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            AddValidator,
            RemoveValidator,
            ReportValidator,
            UndoReportValidator,
            UndoValidatorMetadata,
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
            ReportScores,
            ChangeEpoch,
            Genesis,
            ConsensusCommitPrologue,
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
                            "addValidator" | "add_validator" => Ok(GeneratedField::AddValidator),
                            "removeValidator" | "remove_validator" => Ok(GeneratedField::RemoveValidator),
                            "reportValidator" | "report_validator" => Ok(GeneratedField::ReportValidator),
                            "undoReportValidator" | "undo_report_validator" => Ok(GeneratedField::UndoReportValidator),
                            "undoValidatorMetadata" | "undo_validator_metadata" => Ok(GeneratedField::UndoValidatorMetadata),
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
                            "reportScores" | "report_scores" => Ok(GeneratedField::ReportScores),
                            "changeEpoch" | "change_epoch" => Ok(GeneratedField::ChangeEpoch),
                            "genesis" => Ok(GeneratedField::Genesis),
                            "consensusCommitPrologue" | "consensus_commit_prologue" => Ok(GeneratedField::ConsensusCommitPrologue),
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
                        GeneratedField::UndoValidatorMetadata => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("undoValidatorMetadata"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::UndoValidatorMetadata)
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
                        GeneratedField::ReportScores => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reportScores"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ReportScores)
;
                        }
                        GeneratedField::ChangeEpoch => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("changeEpoch"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ChangeEpoch)
;
                        }
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
            struct_ser.serialize_field("nextEpochExternalNetworkAddress", v)?;
        }
        if let Some(v) = self.next_epoch_internal_network_address.as_ref() {
            struct_ser.serialize_field("nextEpochInternalNetworkAddress", v)?;
        }
        if let Some(v) = self.next_epoch_network_pubkey.as_ref() {
            struct_ser.serialize_field("nextEpochNetworkPubkey", v)?;
        }
        if let Some(v) = self.next_epoch_object_server_address.as_ref() {
            struct_ser.serialize_field("nextEpochObjectServerAddress", v)?;
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
                            next_epoch_external_network_address__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochInternalNetworkAddress => {
                            if next_epoch_internal_network_address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochInternalNetworkAddress"));
                            }
                            next_epoch_internal_network_address__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochNetworkPubkey => {
                            if next_epoch_network_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochNetworkPubkey"));
                            }
                            next_epoch_network_pubkey__ = map_.next_value()?;
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
            struct_ser.serialize_field("nextEpochNetworkAddress", v)?;
        }
        if let Some(v) = self.next_epoch_p2p_address.as_ref() {
            struct_ser.serialize_field("nextEpochP2pAddress", v)?;
        }
        if let Some(v) = self.next_epoch_primary_address.as_ref() {
            struct_ser.serialize_field("nextEpochPrimaryAddress", v)?;
        }
        if let Some(v) = self.next_epoch_protocol_pubkey.as_ref() {
            struct_ser.serialize_field("nextEpochProtocolPubkey", v)?;
        }
        if let Some(v) = self.next_epoch_worker_pubkey.as_ref() {
            struct_ser.serialize_field("nextEpochWorkerPubkey", v)?;
        }
        if let Some(v) = self.next_epoch_network_pubkey.as_ref() {
            struct_ser.serialize_field("nextEpochNetworkPubkey", v)?;
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
                            next_epoch_network_address__ = map_.next_value()?;
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
                        GeneratedField::NextEpochProtocolPubkey => {
                            if next_epoch_protocol_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochProtocolPubkey"));
                            }
                            next_epoch_protocol_pubkey__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochWorkerPubkey => {
                            if next_epoch_worker_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochWorkerPubkey"));
                            }
                            next_epoch_worker_pubkey__ = map_.next_value()?;
                        }
                        GeneratedField::NextEpochNetworkPubkey => {
                            if next_epoch_network_pubkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nextEpochNetworkPubkey"));
                            }
                            next_epoch_network_pubkey__ = map_.next_value()?;
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
        if self.bcs.is_some() {
            len += 1;
        }
        if self.scheme.is_some() {
            len += 1;
        }
        if self.signature.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.UserSignature", len)?;
        if let Some(v) = self.bcs.as_ref() {
            struct_ser.serialize_field("bcs", v)?;
        }
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
            "bcs",
            "scheme",
            "simple",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Bcs,
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
                            "bcs" => Ok(GeneratedField::Bcs),
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
                let mut bcs__ = None;
                let mut scheme__ = None;
                let mut signature__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Bcs => {
                            if bcs__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bcs"));
                            }
                            bcs__ = map_.next_value()?;
                        }
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
                    bcs: bcs__,
                    scheme: scheme__,
                    signature: signature__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.UserSignature", FIELDS, GeneratedVisitor)
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
        if self.public_key.is_some() {
            len += 1;
        }
        if self.weight.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ValidatorCommitteeMember", len)?;
        if let Some(v) = self.public_key.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("publicKey", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.weight.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("weight", ToString::to_string(&v).as_str())?;
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
            type Value = ValidatorCommitteeMember;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ValidatorCommitteeMember")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ValidatorCommitteeMember, V::Error>
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
                            public_key__ = 
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
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ValidatorCommitteeMember {
                    public_key: public_key__,
                    weight: weight__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ValidatorCommitteeMember", FIELDS, GeneratedVisitor)
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
