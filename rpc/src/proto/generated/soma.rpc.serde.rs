impl serde::Serialize for Ability {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "ABILITY_UNKNOWN",
            Self::Copy => "COPY",
            Self::Drop => "DROP",
            Self::Store => "STORE",
            Self::Key => "KEY",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for Ability {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "ABILITY_UNKNOWN",
            "COPY",
            "DROP",
            "STORE",
            "KEY",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Ability;

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
                    "ABILITY_UNKNOWN" => Ok(Ability::Unknown),
                    "COPY" => Ok(Ability::Copy),
                    "DROP" => Ok(Ability::Drop),
                    "STORE" => Ok(Ability::Store),
                    "KEY" => Ok(Ability::Key),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for ActiveJwk {
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
        if self.jwk.is_some() {
            len += 1;
        }
        if self.epoch.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ActiveJwk", len)?;
        if let Some(v) = self.id.as_ref() {
            struct_ser.serialize_field("id", v)?;
        }
        if let Some(v) = self.jwk.as_ref() {
            struct_ser.serialize_field("jwk", v)?;
        }
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ActiveJwk {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "id",
            "jwk",
            "epoch",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Id,
            Jwk,
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
                            "id" => Ok(GeneratedField::Id),
                            "jwk" => Ok(GeneratedField::Jwk),
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
            type Value = ActiveJwk;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ActiveJwk")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ActiveJwk, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut id__ = None;
                let mut jwk__ = None;
                let mut epoch__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Id => {
                            if id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("id"));
                            }
                            id__ = map_.next_value()?;
                        }
                        GeneratedField::Jwk => {
                            if jwk__.is_some() {
                                return Err(serde::de::Error::duplicate_field("jwk"));
                            }
                            jwk__ = map_.next_value()?;
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
                Ok(ActiveJwk {
                    id: id__,
                    jwk: jwk__,
                    epoch: epoch__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ActiveJwk", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Argument {
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
        if self.input.is_some() {
            len += 1;
        }
        if self.result.is_some() {
            len += 1;
        }
        if self.subresult.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Argument", len)?;
        if let Some(v) = self.kind.as_ref() {
            let v = argument::ArgumentKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.input.as_ref() {
            struct_ser.serialize_field("input", v)?;
        }
        if let Some(v) = self.result.as_ref() {
            struct_ser.serialize_field("result", v)?;
        }
        if let Some(v) = self.subresult.as_ref() {
            struct_ser.serialize_field("subresult", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Argument {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "input",
            "result",
            "subresult",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            Input,
            Result,
            Subresult,
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
                            "input" => Ok(GeneratedField::Input),
                            "result" => Ok(GeneratedField::Result),
                            "subresult" => Ok(GeneratedField::Subresult),
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
            type Value = Argument;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Argument")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Argument, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut input__ = None;
                let mut result__ = None;
                let mut subresult__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<argument::ArgumentKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::Input => {
                            if input__.is_some() {
                                return Err(serde::de::Error::duplicate_field("input"));
                            }
                            input__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Result => {
                            if result__.is_some() {
                                return Err(serde::de::Error::duplicate_field("result"));
                            }
                            result__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Subresult => {
                            if subresult__.is_some() {
                                return Err(serde::de::Error::duplicate_field("subresult"));
                            }
                            subresult__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Argument {
                    kind: kind__,
                    input: input__,
                    result: result__,
                    subresult: subresult__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Argument", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for argument::ArgumentKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "ARGUMENT_KIND_UNKNOWN",
            Self::Gas => "GAS",
            Self::Input => "INPUT",
            Self::Result => "RESULT",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for argument::ArgumentKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "ARGUMENT_KIND_UNKNOWN",
            "GAS",
            "INPUT",
            "RESULT",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = argument::ArgumentKind;

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
                    "ARGUMENT_KIND_UNKNOWN" => Ok(argument::ArgumentKind::Unknown),
                    "GAS" => Ok(argument::ArgumentKind::Gas),
                    "INPUT" => Ok(argument::ArgumentKind::Input),
                    "RESULT" => Ok(argument::ArgumentKind::Result),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for AuthenticatorStateExpire {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.min_epoch.is_some() {
            len += 1;
        }
        if self.authenticator_object_initial_shared_version.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.AuthenticatorStateExpire", len)?;
        if let Some(v) = self.min_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("minEpoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.authenticator_object_initial_shared_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("authenticatorObjectInitialSharedVersion", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for AuthenticatorStateExpire {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "min_epoch",
            "minEpoch",
            "authenticator_object_initial_shared_version",
            "authenticatorObjectInitialSharedVersion",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            MinEpoch,
            AuthenticatorObjectInitialSharedVersion,
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
                            "minEpoch" | "min_epoch" => Ok(GeneratedField::MinEpoch),
                            "authenticatorObjectInitialSharedVersion" | "authenticator_object_initial_shared_version" => Ok(GeneratedField::AuthenticatorObjectInitialSharedVersion),
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
            type Value = AuthenticatorStateExpire;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.AuthenticatorStateExpire")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<AuthenticatorStateExpire, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut min_epoch__ = None;
                let mut authenticator_object_initial_shared_version__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::MinEpoch => {
                            if min_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("minEpoch"));
                            }
                            min_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::AuthenticatorObjectInitialSharedVersion => {
                            if authenticator_object_initial_shared_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("authenticatorObjectInitialSharedVersion"));
                            }
                            authenticator_object_initial_shared_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(AuthenticatorStateExpire {
                    min_epoch: min_epoch__,
                    authenticator_object_initial_shared_version: authenticator_object_initial_shared_version__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.AuthenticatorStateExpire", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for AuthenticatorStateUpdate {
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
        if !self.new_active_jwks.is_empty() {
            len += 1;
        }
        if self.authenticator_object_initial_shared_version.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.AuthenticatorStateUpdate", len)?;
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
        if !self.new_active_jwks.is_empty() {
            struct_ser.serialize_field("newActiveJwks", &self.new_active_jwks)?;
        }
        if let Some(v) = self.authenticator_object_initial_shared_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("authenticatorObjectInitialSharedVersion", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for AuthenticatorStateUpdate {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epoch",
            "round",
            "new_active_jwks",
            "newActiveJwks",
            "authenticator_object_initial_shared_version",
            "authenticatorObjectInitialSharedVersion",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            Round,
            NewActiveJwks,
            AuthenticatorObjectInitialSharedVersion,
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
                            "newActiveJwks" | "new_active_jwks" => Ok(GeneratedField::NewActiveJwks),
                            "authenticatorObjectInitialSharedVersion" | "authenticator_object_initial_shared_version" => Ok(GeneratedField::AuthenticatorObjectInitialSharedVersion),
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
            type Value = AuthenticatorStateUpdate;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.AuthenticatorStateUpdate")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<AuthenticatorStateUpdate, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epoch__ = None;
                let mut round__ = None;
                let mut new_active_jwks__ = None;
                let mut authenticator_object_initial_shared_version__ = None;
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
                        GeneratedField::NewActiveJwks => {
                            if new_active_jwks__.is_some() {
                                return Err(serde::de::Error::duplicate_field("newActiveJwks"));
                            }
                            new_active_jwks__ = Some(map_.next_value()?);
                        }
                        GeneratedField::AuthenticatorObjectInitialSharedVersion => {
                            if authenticator_object_initial_shared_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("authenticatorObjectInitialSharedVersion"));
                            }
                            authenticator_object_initial_shared_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(AuthenticatorStateUpdate {
                    epoch: epoch__,
                    round: round__,
                    new_active_jwks: new_active_jwks__.unwrap_or_default(),
                    authenticator_object_initial_shared_version: authenticator_object_initial_shared_version__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.AuthenticatorStateUpdate", FIELDS, GeneratedVisitor)
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
        if self.coin_type.is_some() {
            len += 1;
        }
        if self.amount.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.BalanceChange", len)?;
        if let Some(v) = self.address.as_ref() {
            struct_ser.serialize_field("address", v)?;
        }
        if let Some(v) = self.coin_type.as_ref() {
            struct_ser.serialize_field("coinType", v)?;
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
            "coin_type",
            "coinType",
            "amount",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Address,
            CoinType,
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
                            "coinType" | "coin_type" => Ok(GeneratedField::CoinType),
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
                let mut coin_type__ = None;
                let mut amount__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Address => {
                            if address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("address"));
                            }
                            address__ = map_.next_value()?;
                        }
                        GeneratedField::CoinType => {
                            if coin_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coinType"));
                            }
                            coin_type__ = map_.next_value()?;
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
                    coin_type: coin_type__,
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
impl serde::Serialize for CanceledTransaction {
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
        if !self.version_assignments.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CanceledTransaction", len)?;
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if !self.version_assignments.is_empty() {
            struct_ser.serialize_field("versionAssignments", &self.version_assignments)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CanceledTransaction {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "digest",
            "version_assignments",
            "versionAssignments",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Digest,
            VersionAssignments,
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
                            "versionAssignments" | "version_assignments" => Ok(GeneratedField::VersionAssignments),
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
            type Value = CanceledTransaction;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CanceledTransaction")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CanceledTransaction, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut digest__ = None;
                let mut version_assignments__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::VersionAssignments => {
                            if version_assignments__.is_some() {
                                return Err(serde::de::Error::duplicate_field("versionAssignments"));
                            }
                            version_assignments__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CanceledTransaction {
                    digest: digest__,
                    version_assignments: version_assignments__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CanceledTransaction", FIELDS, GeneratedVisitor)
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
        if self.protocol_version.is_some() {
            len += 1;
        }
        if self.storage_charge.is_some() {
            len += 1;
        }
        if self.computation_charge.is_some() {
            len += 1;
        }
        if self.storage_rebate.is_some() {
            len += 1;
        }
        if self.non_refundable_storage_fee.is_some() {
            len += 1;
        }
        if self.epoch_start_timestamp.is_some() {
            len += 1;
        }
        if !self.system_packages.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ChangeEpoch", len)?;
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.protocol_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("protocolVersion", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.storage_charge.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("storageCharge", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.computation_charge.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("computationCharge", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.storage_rebate.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("storageRebate", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.non_refundable_storage_fee.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nonRefundableStorageFee", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.epoch_start_timestamp.as_ref() {
            struct_ser.serialize_field("epochStartTimestamp", &crate::utils::_serde::TimestampSerializer(v))?;
        }
        if !self.system_packages.is_empty() {
            struct_ser.serialize_field("systemPackages", &self.system_packages)?;
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
            "protocol_version",
            "protocolVersion",
            "storage_charge",
            "storageCharge",
            "computation_charge",
            "computationCharge",
            "storage_rebate",
            "storageRebate",
            "non_refundable_storage_fee",
            "nonRefundableStorageFee",
            "epoch_start_timestamp",
            "epochStartTimestamp",
            "system_packages",
            "systemPackages",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            ProtocolVersion,
            StorageCharge,
            ComputationCharge,
            StorageRebate,
            NonRefundableStorageFee,
            EpochStartTimestamp,
            SystemPackages,
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
                            "protocolVersion" | "protocol_version" => Ok(GeneratedField::ProtocolVersion),
                            "storageCharge" | "storage_charge" => Ok(GeneratedField::StorageCharge),
                            "computationCharge" | "computation_charge" => Ok(GeneratedField::ComputationCharge),
                            "storageRebate" | "storage_rebate" => Ok(GeneratedField::StorageRebate),
                            "nonRefundableStorageFee" | "non_refundable_storage_fee" => Ok(GeneratedField::NonRefundableStorageFee),
                            "epochStartTimestamp" | "epoch_start_timestamp" => Ok(GeneratedField::EpochStartTimestamp),
                            "systemPackages" | "system_packages" => Ok(GeneratedField::SystemPackages),
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
                let mut protocol_version__ = None;
                let mut storage_charge__ = None;
                let mut computation_charge__ = None;
                let mut storage_rebate__ = None;
                let mut non_refundable_storage_fee__ = None;
                let mut epoch_start_timestamp__ = None;
                let mut system_packages__ = None;
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
                        GeneratedField::ProtocolVersion => {
                            if protocol_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("protocolVersion"));
                            }
                            protocol_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::StorageCharge => {
                            if storage_charge__.is_some() {
                                return Err(serde::de::Error::duplicate_field("storageCharge"));
                            }
                            storage_charge__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ComputationCharge => {
                            if computation_charge__.is_some() {
                                return Err(serde::de::Error::duplicate_field("computationCharge"));
                            }
                            computation_charge__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::StorageRebate => {
                            if storage_rebate__.is_some() {
                                return Err(serde::de::Error::duplicate_field("storageRebate"));
                            }
                            storage_rebate__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NonRefundableStorageFee => {
                            if non_refundable_storage_fee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nonRefundableStorageFee"));
                            }
                            non_refundable_storage_fee__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::EpochStartTimestamp => {
                            if epoch_start_timestamp__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epochStartTimestamp"));
                            }
                            epoch_start_timestamp__ = map_.next_value::<::std::option::Option<crate::utils::_serde::TimestampDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::SystemPackages => {
                            if system_packages__.is_some() {
                                return Err(serde::de::Error::duplicate_field("systemPackages"));
                            }
                            system_packages__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ChangeEpoch {
                    epoch: epoch__,
                    protocol_version: protocol_version__,
                    storage_charge: storage_charge__,
                    computation_charge: computation_charge__,
                    storage_rebate: storage_rebate__,
                    non_refundable_storage_fee: non_refundable_storage_fee__,
                    epoch_start_timestamp: epoch_start_timestamp__,
                    system_packages: system_packages__.unwrap_or_default(),
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
            Self::PackageWrite => "OUTPUT_OBJECT_STATE_PACKAGE_WRITE",
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
            "OUTPUT_OBJECT_STATE_PACKAGE_WRITE",
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
                    "OUTPUT_OBJECT_STATE_PACKAGE_WRITE" => Ok(changed_object::OutputObjectState::PackageWrite),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for CircomG1 {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.e0.is_some() {
            len += 1;
        }
        if self.e1.is_some() {
            len += 1;
        }
        if self.e2.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CircomG1", len)?;
        if let Some(v) = self.e0.as_ref() {
            struct_ser.serialize_field("e0", v)?;
        }
        if let Some(v) = self.e1.as_ref() {
            struct_ser.serialize_field("e1", v)?;
        }
        if let Some(v) = self.e2.as_ref() {
            struct_ser.serialize_field("e2", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CircomG1 {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "e0",
            "e1",
            "e2",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            E0,
            E1,
            E2,
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
                            "e0" => Ok(GeneratedField::E0),
                            "e1" => Ok(GeneratedField::E1),
                            "e2" => Ok(GeneratedField::E2),
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
            type Value = CircomG1;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CircomG1")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CircomG1, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut e0__ = None;
                let mut e1__ = None;
                let mut e2__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::E0 => {
                            if e0__.is_some() {
                                return Err(serde::de::Error::duplicate_field("e0"));
                            }
                            e0__ = map_.next_value()?;
                        }
                        GeneratedField::E1 => {
                            if e1__.is_some() {
                                return Err(serde::de::Error::duplicate_field("e1"));
                            }
                            e1__ = map_.next_value()?;
                        }
                        GeneratedField::E2 => {
                            if e2__.is_some() {
                                return Err(serde::de::Error::duplicate_field("e2"));
                            }
                            e2__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CircomG1 {
                    e0: e0__,
                    e1: e1__,
                    e2: e2__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CircomG1", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CircomG2 {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.e00.is_some() {
            len += 1;
        }
        if self.e01.is_some() {
            len += 1;
        }
        if self.e10.is_some() {
            len += 1;
        }
        if self.e11.is_some() {
            len += 1;
        }
        if self.e20.is_some() {
            len += 1;
        }
        if self.e21.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CircomG2", len)?;
        if let Some(v) = self.e00.as_ref() {
            struct_ser.serialize_field("e00", v)?;
        }
        if let Some(v) = self.e01.as_ref() {
            struct_ser.serialize_field("e01", v)?;
        }
        if let Some(v) = self.e10.as_ref() {
            struct_ser.serialize_field("e10", v)?;
        }
        if let Some(v) = self.e11.as_ref() {
            struct_ser.serialize_field("e11", v)?;
        }
        if let Some(v) = self.e20.as_ref() {
            struct_ser.serialize_field("e20", v)?;
        }
        if let Some(v) = self.e21.as_ref() {
            struct_ser.serialize_field("e21", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CircomG2 {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "e00",
            "e01",
            "e10",
            "e11",
            "e20",
            "e21",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            E00,
            E01,
            E10,
            E11,
            E20,
            E21,
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
                            "e00" => Ok(GeneratedField::E00),
                            "e01" => Ok(GeneratedField::E01),
                            "e10" => Ok(GeneratedField::E10),
                            "e11" => Ok(GeneratedField::E11),
                            "e20" => Ok(GeneratedField::E20),
                            "e21" => Ok(GeneratedField::E21),
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
            type Value = CircomG2;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CircomG2")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CircomG2, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut e00__ = None;
                let mut e01__ = None;
                let mut e10__ = None;
                let mut e11__ = None;
                let mut e20__ = None;
                let mut e21__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::E00 => {
                            if e00__.is_some() {
                                return Err(serde::de::Error::duplicate_field("e00"));
                            }
                            e00__ = map_.next_value()?;
                        }
                        GeneratedField::E01 => {
                            if e01__.is_some() {
                                return Err(serde::de::Error::duplicate_field("e01"));
                            }
                            e01__ = map_.next_value()?;
                        }
                        GeneratedField::E10 => {
                            if e10__.is_some() {
                                return Err(serde::de::Error::duplicate_field("e10"));
                            }
                            e10__ = map_.next_value()?;
                        }
                        GeneratedField::E11 => {
                            if e11__.is_some() {
                                return Err(serde::de::Error::duplicate_field("e11"));
                            }
                            e11__ = map_.next_value()?;
                        }
                        GeneratedField::E20 => {
                            if e20__.is_some() {
                                return Err(serde::de::Error::duplicate_field("e20"));
                            }
                            e20__ = map_.next_value()?;
                        }
                        GeneratedField::E21 => {
                            if e21__.is_some() {
                                return Err(serde::de::Error::duplicate_field("e21"));
                            }
                            e21__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CircomG2 {
                    e00: e00__,
                    e01: e01__,
                    e10: e10__,
                    e11: e11__,
                    e20: e20__,
                    e21: e21__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CircomG2", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CleverError {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.error_code.is_some() {
            len += 1;
        }
        if self.line_number.is_some() {
            len += 1;
        }
        if self.constant_name.is_some() {
            len += 1;
        }
        if self.constant_type.is_some() {
            len += 1;
        }
        if self.value.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CleverError", len)?;
        if let Some(v) = self.error_code.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("errorCode", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.line_number.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("lineNumber", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.constant_name.as_ref() {
            struct_ser.serialize_field("constantName", v)?;
        }
        if let Some(v) = self.constant_type.as_ref() {
            struct_ser.serialize_field("constantType", v)?;
        }
        if let Some(v) = self.value.as_ref() {
            match v {
                clever_error::Value::Rendered(v) => {
                    struct_ser.serialize_field("rendered", v)?;
                }
                clever_error::Value::Raw(v) => {
                    #[allow(clippy::needless_borrow)]
                    #[allow(clippy::needless_borrows_for_generic_args)]
                    struct_ser.serialize_field("raw", crate::utils::_serde::base64::encode(&v).as_str())?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CleverError {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "error_code",
            "errorCode",
            "line_number",
            "lineNumber",
            "constant_name",
            "constantName",
            "constant_type",
            "constantType",
            "rendered",
            "raw",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ErrorCode,
            LineNumber,
            ConstantName,
            ConstantType,
            Rendered,
            Raw,
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
                            "errorCode" | "error_code" => Ok(GeneratedField::ErrorCode),
                            "lineNumber" | "line_number" => Ok(GeneratedField::LineNumber),
                            "constantName" | "constant_name" => Ok(GeneratedField::ConstantName),
                            "constantType" | "constant_type" => Ok(GeneratedField::ConstantType),
                            "rendered" => Ok(GeneratedField::Rendered),
                            "raw" => Ok(GeneratedField::Raw),
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
            type Value = CleverError;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CleverError")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CleverError, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut error_code__ = None;
                let mut line_number__ = None;
                let mut constant_name__ = None;
                let mut constant_type__ = None;
                let mut value__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ErrorCode => {
                            if error_code__.is_some() {
                                return Err(serde::de::Error::duplicate_field("errorCode"));
                            }
                            error_code__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::LineNumber => {
                            if line_number__.is_some() {
                                return Err(serde::de::Error::duplicate_field("lineNumber"));
                            }
                            line_number__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ConstantName => {
                            if constant_name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("constantName"));
                            }
                            constant_name__ = map_.next_value()?;
                        }
                        GeneratedField::ConstantType => {
                            if constant_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("constantType"));
                            }
                            constant_type__ = map_.next_value()?;
                        }
                        GeneratedField::Rendered => {
                            if value__.is_some() {
                                return Err(serde::de::Error::duplicate_field("rendered"));
                            }
                            value__ = map_.next_value::<::std::option::Option<_>>()?.map(clever_error::Value::Rendered);
                        }
                        GeneratedField::Raw => {
                            if value__.is_some() {
                                return Err(serde::de::Error::duplicate_field("raw"));
                            }
                            value__ = map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| clever_error::Value::Raw(x.0));
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CleverError {
                    error_code: error_code__,
                    line_number: line_number__,
                    constant_name: constant_name__,
                    constant_type: constant_type__,
                    value: value__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CleverError", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CoinDenyListError {
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
        if self.coin_type.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CoinDenyListError", len)?;
        if let Some(v) = self.address.as_ref() {
            struct_ser.serialize_field("address", v)?;
        }
        if let Some(v) = self.coin_type.as_ref() {
            struct_ser.serialize_field("coinType", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CoinDenyListError {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "address",
            "coin_type",
            "coinType",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Address,
            CoinType,
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
                            "coinType" | "coin_type" => Ok(GeneratedField::CoinType),
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
            type Value = CoinDenyListError;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CoinDenyListError")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CoinDenyListError, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut address__ = None;
                let mut coin_type__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Address => {
                            if address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("address"));
                            }
                            address__ = map_.next_value()?;
                        }
                        GeneratedField::CoinType => {
                            if coin_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coinType"));
                            }
                            coin_type__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CoinDenyListError {
                    address: address__,
                    coin_type: coin_type__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CoinDenyListError", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Command {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.command.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Command", len)?;
        if let Some(v) = self.command.as_ref() {
            match v {
                command::Command::MoveCall(v) => {
                    struct_ser.serialize_field("moveCall", v)?;
                }
                command::Command::TransferObjects(v) => {
                    struct_ser.serialize_field("transferObjects", v)?;
                }
                command::Command::SplitCoins(v) => {
                    struct_ser.serialize_field("splitCoins", v)?;
                }
                command::Command::MergeCoins(v) => {
                    struct_ser.serialize_field("mergeCoins", v)?;
                }
                command::Command::Publish(v) => {
                    struct_ser.serialize_field("publish", v)?;
                }
                command::Command::MakeMoveVector(v) => {
                    struct_ser.serialize_field("makeMoveVector", v)?;
                }
                command::Command::Upgrade(v) => {
                    struct_ser.serialize_field("upgrade", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Command {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "move_call",
            "moveCall",
            "transfer_objects",
            "transferObjects",
            "split_coins",
            "splitCoins",
            "merge_coins",
            "mergeCoins",
            "publish",
            "make_move_vector",
            "makeMoveVector",
            "upgrade",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            MoveCall,
            TransferObjects,
            SplitCoins,
            MergeCoins,
            Publish,
            MakeMoveVector,
            Upgrade,
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
                            "moveCall" | "move_call" => Ok(GeneratedField::MoveCall),
                            "transferObjects" | "transfer_objects" => Ok(GeneratedField::TransferObjects),
                            "splitCoins" | "split_coins" => Ok(GeneratedField::SplitCoins),
                            "mergeCoins" | "merge_coins" => Ok(GeneratedField::MergeCoins),
                            "publish" => Ok(GeneratedField::Publish),
                            "makeMoveVector" | "make_move_vector" => Ok(GeneratedField::MakeMoveVector),
                            "upgrade" => Ok(GeneratedField::Upgrade),
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
            type Value = Command;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Command")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Command, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut command__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::MoveCall => {
                            if command__.is_some() {
                                return Err(serde::de::Error::duplicate_field("moveCall"));
                            }
                            command__ = map_.next_value::<::std::option::Option<_>>()?.map(command::Command::MoveCall)
;
                        }
                        GeneratedField::TransferObjects => {
                            if command__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transferObjects"));
                            }
                            command__ = map_.next_value::<::std::option::Option<_>>()?.map(command::Command::TransferObjects)
;
                        }
                        GeneratedField::SplitCoins => {
                            if command__.is_some() {
                                return Err(serde::de::Error::duplicate_field("splitCoins"));
                            }
                            command__ = map_.next_value::<::std::option::Option<_>>()?.map(command::Command::SplitCoins)
;
                        }
                        GeneratedField::MergeCoins => {
                            if command__.is_some() {
                                return Err(serde::de::Error::duplicate_field("mergeCoins"));
                            }
                            command__ = map_.next_value::<::std::option::Option<_>>()?.map(command::Command::MergeCoins)
;
                        }
                        GeneratedField::Publish => {
                            if command__.is_some() {
                                return Err(serde::de::Error::duplicate_field("publish"));
                            }
                            command__ = map_.next_value::<::std::option::Option<_>>()?.map(command::Command::Publish)
;
                        }
                        GeneratedField::MakeMoveVector => {
                            if command__.is_some() {
                                return Err(serde::de::Error::duplicate_field("makeMoveVector"));
                            }
                            command__ = map_.next_value::<::std::option::Option<_>>()?.map(command::Command::MakeMoveVector)
;
                        }
                        GeneratedField::Upgrade => {
                            if command__.is_some() {
                                return Err(serde::de::Error::duplicate_field("upgrade"));
                            }
                            command__ = map_.next_value::<::std::option::Option<_>>()?.map(command::Command::Upgrade)
;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Command {
                    command: command__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Command", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CommandArgumentError {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.argument.is_some() {
            len += 1;
        }
        if self.kind.is_some() {
            len += 1;
        }
        if self.index_error.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CommandArgumentError", len)?;
        if let Some(v) = self.argument.as_ref() {
            struct_ser.serialize_field("argument", v)?;
        }
        if let Some(v) = self.kind.as_ref() {
            let v = command_argument_error::CommandArgumentErrorKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.index_error.as_ref() {
            struct_ser.serialize_field("indexError", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CommandArgumentError {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "argument",
            "kind",
            "index_error",
            "indexError",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Argument,
            Kind,
            IndexError,
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
                            "argument" => Ok(GeneratedField::Argument),
                            "kind" => Ok(GeneratedField::Kind),
                            "indexError" | "index_error" => Ok(GeneratedField::IndexError),
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
            type Value = CommandArgumentError;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CommandArgumentError")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CommandArgumentError, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut argument__ = None;
                let mut kind__ = None;
                let mut index_error__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Argument => {
                            if argument__.is_some() {
                                return Err(serde::de::Error::duplicate_field("argument"));
                            }
                            argument__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<command_argument_error::CommandArgumentErrorKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::IndexError => {
                            if index_error__.is_some() {
                                return Err(serde::de::Error::duplicate_field("indexError"));
                            }
                            index_error__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(CommandArgumentError {
                    argument: argument__,
                    kind: kind__,
                    index_error: index_error__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CommandArgumentError", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for command_argument_error::CommandArgumentErrorKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "COMMAND_ARGUMENT_ERROR_KIND_UNKNOWN",
            Self::TypeMismatch => "TYPE_MISMATCH",
            Self::InvalidBcsBytes => "INVALID_BCS_BYTES",
            Self::InvalidUsageOfPureArgument => "INVALID_USAGE_OF_PURE_ARGUMENT",
            Self::InvalidArgumentToPrivateEntryFunction => "INVALID_ARGUMENT_TO_PRIVATE_ENTRY_FUNCTION",
            Self::IndexOutOfBounds => "INDEX_OUT_OF_BOUNDS",
            Self::SecondaryIndexOutOfBounds => "SECONDARY_INDEX_OUT_OF_BOUNDS",
            Self::InvalidResultArity => "INVALID_RESULT_ARITY",
            Self::InvalidGasCoinUsage => "INVALID_GAS_COIN_USAGE",
            Self::InvalidValueUsage => "INVALID_VALUE_USAGE",
            Self::InvalidObjectByValue => "INVALID_OBJECT_BY_VALUE",
            Self::InvalidObjectByMutRef => "INVALID_OBJECT_BY_MUT_REF",
            Self::ConsensusObjectOperationNotAllowed => "CONSENSUS_OBJECT_OPERATION_NOT_ALLOWED",
            Self::InvalidArgumentArity => "INVALID_ARGUMENT_ARITY",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for command_argument_error::CommandArgumentErrorKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "COMMAND_ARGUMENT_ERROR_KIND_UNKNOWN",
            "TYPE_MISMATCH",
            "INVALID_BCS_BYTES",
            "INVALID_USAGE_OF_PURE_ARGUMENT",
            "INVALID_ARGUMENT_TO_PRIVATE_ENTRY_FUNCTION",
            "INDEX_OUT_OF_BOUNDS",
            "SECONDARY_INDEX_OUT_OF_BOUNDS",
            "INVALID_RESULT_ARITY",
            "INVALID_GAS_COIN_USAGE",
            "INVALID_VALUE_USAGE",
            "INVALID_OBJECT_BY_VALUE",
            "INVALID_OBJECT_BY_MUT_REF",
            "CONSENSUS_OBJECT_OPERATION_NOT_ALLOWED",
            "INVALID_ARGUMENT_ARITY",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = command_argument_error::CommandArgumentErrorKind;

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
                    "COMMAND_ARGUMENT_ERROR_KIND_UNKNOWN" => Ok(command_argument_error::CommandArgumentErrorKind::Unknown),
                    "TYPE_MISMATCH" => Ok(command_argument_error::CommandArgumentErrorKind::TypeMismatch),
                    "INVALID_BCS_BYTES" => Ok(command_argument_error::CommandArgumentErrorKind::InvalidBcsBytes),
                    "INVALID_USAGE_OF_PURE_ARGUMENT" => Ok(command_argument_error::CommandArgumentErrorKind::InvalidUsageOfPureArgument),
                    "INVALID_ARGUMENT_TO_PRIVATE_ENTRY_FUNCTION" => Ok(command_argument_error::CommandArgumentErrorKind::InvalidArgumentToPrivateEntryFunction),
                    "INDEX_OUT_OF_BOUNDS" => Ok(command_argument_error::CommandArgumentErrorKind::IndexOutOfBounds),
                    "SECONDARY_INDEX_OUT_OF_BOUNDS" => Ok(command_argument_error::CommandArgumentErrorKind::SecondaryIndexOutOfBounds),
                    "INVALID_RESULT_ARITY" => Ok(command_argument_error::CommandArgumentErrorKind::InvalidResultArity),
                    "INVALID_GAS_COIN_USAGE" => Ok(command_argument_error::CommandArgumentErrorKind::InvalidGasCoinUsage),
                    "INVALID_VALUE_USAGE" => Ok(command_argument_error::CommandArgumentErrorKind::InvalidValueUsage),
                    "INVALID_OBJECT_BY_VALUE" => Ok(command_argument_error::CommandArgumentErrorKind::InvalidObjectByValue),
                    "INVALID_OBJECT_BY_MUT_REF" => Ok(command_argument_error::CommandArgumentErrorKind::InvalidObjectByMutRef),
                    "CONSENSUS_OBJECT_OPERATION_NOT_ALLOWED" => Ok(command_argument_error::CommandArgumentErrorKind::ConsensusObjectOperationNotAllowed),
                    "INVALID_ARGUMENT_ARITY" => Ok(command_argument_error::CommandArgumentErrorKind::InvalidArgumentArity),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for CongestedObjects {
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
        let mut struct_ser = serializer.serialize_struct("soma.rpc.CongestedObjects", len)?;
        if !self.objects.is_empty() {
            struct_ser.serialize_field("objects", &self.objects)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CongestedObjects {
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
            type Value = CongestedObjects;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.CongestedObjects")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CongestedObjects, V::Error>
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
                Ok(CongestedObjects {
                    objects: objects__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.CongestedObjects", FIELDS, GeneratedVisitor)
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
        if self.consensus_determined_version_assignments.is_some() {
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
        if let Some(v) = self.consensus_determined_version_assignments.as_ref() {
            struct_ser.serialize_field("consensusDeterminedVersionAssignments", v)?;
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
            "commit_timestamp",
            "commitTimestamp",
            "consensus_commit_digest",
            "consensusCommitDigest",
            "sub_dag_index",
            "subDagIndex",
            "consensus_determined_version_assignments",
            "consensusDeterminedVersionAssignments",
            "additional_state_digest",
            "additionalStateDigest",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            Round,
            CommitTimestamp,
            ConsensusCommitDigest,
            SubDagIndex,
            ConsensusDeterminedVersionAssignments,
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
                            "commitTimestamp" | "commit_timestamp" => Ok(GeneratedField::CommitTimestamp),
                            "consensusCommitDigest" | "consensus_commit_digest" => Ok(GeneratedField::ConsensusCommitDigest),
                            "subDagIndex" | "sub_dag_index" => Ok(GeneratedField::SubDagIndex),
                            "consensusDeterminedVersionAssignments" | "consensus_determined_version_assignments" => Ok(GeneratedField::ConsensusDeterminedVersionAssignments),
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
                let mut commit_timestamp__ = None;
                let mut consensus_commit_digest__ = None;
                let mut sub_dag_index__ = None;
                let mut consensus_determined_version_assignments__ = None;
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
                        GeneratedField::ConsensusDeterminedVersionAssignments => {
                            if consensus_determined_version_assignments__.is_some() {
                                return Err(serde::de::Error::duplicate_field("consensusDeterminedVersionAssignments"));
                            }
                            consensus_determined_version_assignments__ = map_.next_value()?;
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
                    commit_timestamp: commit_timestamp__,
                    consensus_commit_digest: consensus_commit_digest__,
                    sub_dag_index: sub_dag_index__,
                    consensus_determined_version_assignments: consensus_determined_version_assignments__,
                    additional_state_digest: additional_state_digest__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ConsensusCommitPrologue", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ConsensusDeterminedVersionAssignments {
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
        if !self.canceled_transactions.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ConsensusDeterminedVersionAssignments", len)?;
        if let Some(v) = self.version.as_ref() {
            struct_ser.serialize_field("version", v)?;
        }
        if !self.canceled_transactions.is_empty() {
            struct_ser.serialize_field("canceledTransactions", &self.canceled_transactions)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ConsensusDeterminedVersionAssignments {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "version",
            "canceled_transactions",
            "canceledTransactions",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Version,
            CanceledTransactions,
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
                            "version" => Ok(GeneratedField::Version),
                            "canceledTransactions" | "canceled_transactions" => Ok(GeneratedField::CanceledTransactions),
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
            type Value = ConsensusDeterminedVersionAssignments;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ConsensusDeterminedVersionAssignments")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ConsensusDeterminedVersionAssignments, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut version__ = None;
                let mut canceled_transactions__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::CanceledTransactions => {
                            if canceled_transactions__.is_some() {
                                return Err(serde::de::Error::duplicate_field("canceledTransactions"));
                            }
                            canceled_transactions__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ConsensusDeterminedVersionAssignments {
                    version: version__,
                    canceled_transactions: canceled_transactions__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ConsensusDeterminedVersionAssignments", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for DatatypeDescriptor {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.type_name.is_some() {
            len += 1;
        }
        if self.defining_id.is_some() {
            len += 1;
        }
        if self.module.is_some() {
            len += 1;
        }
        if self.name.is_some() {
            len += 1;
        }
        if !self.abilities.is_empty() {
            len += 1;
        }
        if !self.type_parameters.is_empty() {
            len += 1;
        }
        if self.kind.is_some() {
            len += 1;
        }
        if !self.fields.is_empty() {
            len += 1;
        }
        if !self.variants.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.DatatypeDescriptor", len)?;
        if let Some(v) = self.type_name.as_ref() {
            struct_ser.serialize_field("typeName", v)?;
        }
        if let Some(v) = self.defining_id.as_ref() {
            struct_ser.serialize_field("definingId", v)?;
        }
        if let Some(v) = self.module.as_ref() {
            struct_ser.serialize_field("module", v)?;
        }
        if let Some(v) = self.name.as_ref() {
            struct_ser.serialize_field("name", v)?;
        }
        if !self.abilities.is_empty() {
            let v = self.abilities.iter().cloned().map(|v| {
                Ability::try_from(v)
                    .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", v)))
                }).collect::<std::result::Result<Vec<_>, _>>()?;
            struct_ser.serialize_field("abilities", &v)?;
        }
        if !self.type_parameters.is_empty() {
            struct_ser.serialize_field("typeParameters", &self.type_parameters)?;
        }
        if let Some(v) = self.kind.as_ref() {
            let v = datatype_descriptor::DatatypeKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if !self.fields.is_empty() {
            struct_ser.serialize_field("fields", &self.fields)?;
        }
        if !self.variants.is_empty() {
            struct_ser.serialize_field("variants", &self.variants)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for DatatypeDescriptor {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "type_name",
            "typeName",
            "defining_id",
            "definingId",
            "module",
            "name",
            "abilities",
            "type_parameters",
            "typeParameters",
            "kind",
            "fields",
            "variants",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TypeName,
            DefiningId,
            Module,
            Name,
            Abilities,
            TypeParameters,
            Kind,
            Fields,
            Variants,
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
                            "typeName" | "type_name" => Ok(GeneratedField::TypeName),
                            "definingId" | "defining_id" => Ok(GeneratedField::DefiningId),
                            "module" => Ok(GeneratedField::Module),
                            "name" => Ok(GeneratedField::Name),
                            "abilities" => Ok(GeneratedField::Abilities),
                            "typeParameters" | "type_parameters" => Ok(GeneratedField::TypeParameters),
                            "kind" => Ok(GeneratedField::Kind),
                            "fields" => Ok(GeneratedField::Fields),
                            "variants" => Ok(GeneratedField::Variants),
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
            type Value = DatatypeDescriptor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.DatatypeDescriptor")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<DatatypeDescriptor, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut type_name__ = None;
                let mut defining_id__ = None;
                let mut module__ = None;
                let mut name__ = None;
                let mut abilities__ = None;
                let mut type_parameters__ = None;
                let mut kind__ = None;
                let mut fields__ = None;
                let mut variants__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::TypeName => {
                            if type_name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("typeName"));
                            }
                            type_name__ = map_.next_value()?;
                        }
                        GeneratedField::DefiningId => {
                            if defining_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("definingId"));
                            }
                            defining_id__ = map_.next_value()?;
                        }
                        GeneratedField::Module => {
                            if module__.is_some() {
                                return Err(serde::de::Error::duplicate_field("module"));
                            }
                            module__ = map_.next_value()?;
                        }
                        GeneratedField::Name => {
                            if name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("name"));
                            }
                            name__ = map_.next_value()?;
                        }
                        GeneratedField::Abilities => {
                            if abilities__.is_some() {
                                return Err(serde::de::Error::duplicate_field("abilities"));
                            }
                            abilities__ = Some(map_.next_value::<Vec<Ability>>()?.into_iter().map(|x| x as i32).collect());
                        }
                        GeneratedField::TypeParameters => {
                            if type_parameters__.is_some() {
                                return Err(serde::de::Error::duplicate_field("typeParameters"));
                            }
                            type_parameters__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<datatype_descriptor::DatatypeKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::Fields => {
                            if fields__.is_some() {
                                return Err(serde::de::Error::duplicate_field("fields"));
                            }
                            fields__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Variants => {
                            if variants__.is_some() {
                                return Err(serde::de::Error::duplicate_field("variants"));
                            }
                            variants__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(DatatypeDescriptor {
                    type_name: type_name__,
                    defining_id: defining_id__,
                    module: module__,
                    name: name__,
                    abilities: abilities__.unwrap_or_default(),
                    type_parameters: type_parameters__.unwrap_or_default(),
                    kind: kind__,
                    fields: fields__.unwrap_or_default(),
                    variants: variants__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.DatatypeDescriptor", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for datatype_descriptor::DatatypeKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "DATATYPE_KIND_UNKNOWN",
            Self::Struct => "STRUCT",
            Self::Enum => "ENUM",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for datatype_descriptor::DatatypeKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "DATATYPE_KIND_UNKNOWN",
            "STRUCT",
            "ENUM",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = datatype_descriptor::DatatypeKind;

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
                    "DATATYPE_KIND_UNKNOWN" => Ok(datatype_descriptor::DatatypeKind::Unknown),
                    "STRUCT" => Ok(datatype_descriptor::DatatypeKind::Struct),
                    "ENUM" => Ok(datatype_descriptor::DatatypeKind::Enum),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for EndOfEpochTransaction {
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
        let mut struct_ser = serializer.serialize_struct("soma.rpc.EndOfEpochTransaction", len)?;
        if !self.transactions.is_empty() {
            struct_ser.serialize_field("transactions", &self.transactions)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for EndOfEpochTransaction {
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
            type Value = EndOfEpochTransaction;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.EndOfEpochTransaction")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<EndOfEpochTransaction, V::Error>
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
                Ok(EndOfEpochTransaction {
                    transactions: transactions__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.EndOfEpochTransaction", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for EndOfEpochTransactionKind {
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
        let mut struct_ser = serializer.serialize_struct("soma.rpc.EndOfEpochTransactionKind", len)?;
        if let Some(v) = self.kind.as_ref() {
            match v {
                end_of_epoch_transaction_kind::Kind::ChangeEpoch(v) => {
                    struct_ser.serialize_field("changeEpoch", v)?;
                }
                end_of_epoch_transaction_kind::Kind::AuthenticatorStateExpire(v) => {
                    struct_ser.serialize_field("authenticatorStateExpire", v)?;
                }
                end_of_epoch_transaction_kind::Kind::ExecutionTimeObservations(v) => {
                    struct_ser.serialize_field("executionTimeObservations", v)?;
                }
                end_of_epoch_transaction_kind::Kind::AuthenticatorStateCreate(v) => {
                    struct_ser.serialize_field("authenticatorStateCreate", &crate::utils::_serde::EmptySerializer(v))?;
                }
                end_of_epoch_transaction_kind::Kind::RandomnessStateCreate(v) => {
                    struct_ser.serialize_field("randomnessStateCreate", &crate::utils::_serde::EmptySerializer(v))?;
                }
                end_of_epoch_transaction_kind::Kind::DenyListStateCreate(v) => {
                    struct_ser.serialize_field("denyListStateCreate", &crate::utils::_serde::EmptySerializer(v))?;
                }
                end_of_epoch_transaction_kind::Kind::BridgeStateCreate(v) => {
                    struct_ser.serialize_field("bridgeStateCreate", v)?;
                }
                end_of_epoch_transaction_kind::Kind::BridgeCommitteeInit(v) => {
                    #[allow(clippy::needless_borrow)]
                    #[allow(clippy::needless_borrows_for_generic_args)]
                    struct_ser.serialize_field("bridgeCommitteeInit", ToString::to_string(&v).as_str())?;
                }
                end_of_epoch_transaction_kind::Kind::AccumulatorRootCreate(v) => {
                    struct_ser.serialize_field("accumulatorRootCreate", &crate::utils::_serde::EmptySerializer(v))?;
                }
                end_of_epoch_transaction_kind::Kind::CoinRegistryCreate(v) => {
                    struct_ser.serialize_field("coinRegistryCreate", &crate::utils::_serde::EmptySerializer(v))?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for EndOfEpochTransactionKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "change_epoch",
            "changeEpoch",
            "authenticator_state_expire",
            "authenticatorStateExpire",
            "execution_time_observations",
            "executionTimeObservations",
            "authenticator_state_create",
            "authenticatorStateCreate",
            "randomness_state_create",
            "randomnessStateCreate",
            "deny_list_state_create",
            "denyListStateCreate",
            "bridge_state_create",
            "bridgeStateCreate",
            "bridge_committee_init",
            "bridgeCommitteeInit",
            "accumulator_root_create",
            "accumulatorRootCreate",
            "coin_registry_create",
            "coinRegistryCreate",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ChangeEpoch,
            AuthenticatorStateExpire,
            ExecutionTimeObservations,
            AuthenticatorStateCreate,
            RandomnessStateCreate,
            DenyListStateCreate,
            BridgeStateCreate,
            BridgeCommitteeInit,
            AccumulatorRootCreate,
            CoinRegistryCreate,
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
                            "changeEpoch" | "change_epoch" => Ok(GeneratedField::ChangeEpoch),
                            "authenticatorStateExpire" | "authenticator_state_expire" => Ok(GeneratedField::AuthenticatorStateExpire),
                            "executionTimeObservations" | "execution_time_observations" => Ok(GeneratedField::ExecutionTimeObservations),
                            "authenticatorStateCreate" | "authenticator_state_create" => Ok(GeneratedField::AuthenticatorStateCreate),
                            "randomnessStateCreate" | "randomness_state_create" => Ok(GeneratedField::RandomnessStateCreate),
                            "denyListStateCreate" | "deny_list_state_create" => Ok(GeneratedField::DenyListStateCreate),
                            "bridgeStateCreate" | "bridge_state_create" => Ok(GeneratedField::BridgeStateCreate),
                            "bridgeCommitteeInit" | "bridge_committee_init" => Ok(GeneratedField::BridgeCommitteeInit),
                            "accumulatorRootCreate" | "accumulator_root_create" => Ok(GeneratedField::AccumulatorRootCreate),
                            "coinRegistryCreate" | "coin_registry_create" => Ok(GeneratedField::CoinRegistryCreate),
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
            type Value = EndOfEpochTransactionKind;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.EndOfEpochTransactionKind")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<EndOfEpochTransactionKind, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ChangeEpoch => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("changeEpoch"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(end_of_epoch_transaction_kind::Kind::ChangeEpoch)
;
                        }
                        GeneratedField::AuthenticatorStateExpire => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("authenticatorStateExpire"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(end_of_epoch_transaction_kind::Kind::AuthenticatorStateExpire)
;
                        }
                        GeneratedField::ExecutionTimeObservations => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("executionTimeObservations"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(end_of_epoch_transaction_kind::Kind::ExecutionTimeObservations)
;
                        }
                        GeneratedField::AuthenticatorStateCreate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("authenticatorStateCreate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<crate::utils::_serde::EmptyDeserializer>>()?.map(|x| end_of_epoch_transaction_kind::Kind::AuthenticatorStateCreate(x.0));
                        }
                        GeneratedField::RandomnessStateCreate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("randomnessStateCreate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<crate::utils::_serde::EmptyDeserializer>>()?.map(|x| end_of_epoch_transaction_kind::Kind::RandomnessStateCreate(x.0));
                        }
                        GeneratedField::DenyListStateCreate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("denyListStateCreate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<crate::utils::_serde::EmptyDeserializer>>()?.map(|x| end_of_epoch_transaction_kind::Kind::DenyListStateCreate(x.0));
                        }
                        GeneratedField::BridgeStateCreate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bridgeStateCreate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(end_of_epoch_transaction_kind::Kind::BridgeStateCreate);
                        }
                        GeneratedField::BridgeCommitteeInit => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bridgeCommitteeInit"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| end_of_epoch_transaction_kind::Kind::BridgeCommitteeInit(x.0));
                        }
                        GeneratedField::AccumulatorRootCreate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("accumulatorRootCreate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<crate::utils::_serde::EmptyDeserializer>>()?.map(|x| end_of_epoch_transaction_kind::Kind::AccumulatorRootCreate(x.0));
                        }
                        GeneratedField::CoinRegistryCreate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coinRegistryCreate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<crate::utils::_serde::EmptyDeserializer>>()?.map(|x| end_of_epoch_transaction_kind::Kind::CoinRegistryCreate(x.0));
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(EndOfEpochTransactionKind {
                    kind: kind__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.EndOfEpochTransactionKind", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for Event {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.package_id.is_some() {
            len += 1;
        }
        if self.module.is_some() {
            len += 1;
        }
        if self.sender.is_some() {
            len += 1;
        }
        if self.event_type.is_some() {
            len += 1;
        }
        if self.contents.is_some() {
            len += 1;
        }
        if self.json.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Event", len)?;
        if let Some(v) = self.package_id.as_ref() {
            struct_ser.serialize_field("packageId", v)?;
        }
        if let Some(v) = self.module.as_ref() {
            struct_ser.serialize_field("module", v)?;
        }
        if let Some(v) = self.sender.as_ref() {
            struct_ser.serialize_field("sender", v)?;
        }
        if let Some(v) = self.event_type.as_ref() {
            struct_ser.serialize_field("eventType", v)?;
        }
        if let Some(v) = self.contents.as_ref() {
            struct_ser.serialize_field("contents", v)?;
        }
        if let Some(v) = self.json.as_ref() {
            struct_ser.serialize_field("json", &crate::utils::_serde::ValueSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Event {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "package_id",
            "packageId",
            "module",
            "sender",
            "event_type",
            "eventType",
            "contents",
            "json",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            PackageId,
            Module,
            Sender,
            EventType,
            Contents,
            Json,
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
                            "packageId" | "package_id" => Ok(GeneratedField::PackageId),
                            "module" => Ok(GeneratedField::Module),
                            "sender" => Ok(GeneratedField::Sender),
                            "eventType" | "event_type" => Ok(GeneratedField::EventType),
                            "contents" => Ok(GeneratedField::Contents),
                            "json" => Ok(GeneratedField::Json),
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
            type Value = Event;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Event")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Event, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut package_id__ = None;
                let mut module__ = None;
                let mut sender__ = None;
                let mut event_type__ = None;
                let mut contents__ = None;
                let mut json__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::PackageId => {
                            if package_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("packageId"));
                            }
                            package_id__ = map_.next_value()?;
                        }
                        GeneratedField::Module => {
                            if module__.is_some() {
                                return Err(serde::de::Error::duplicate_field("module"));
                            }
                            module__ = map_.next_value()?;
                        }
                        GeneratedField::Sender => {
                            if sender__.is_some() {
                                return Err(serde::de::Error::duplicate_field("sender"));
                            }
                            sender__ = map_.next_value()?;
                        }
                        GeneratedField::EventType => {
                            if event_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("eventType"));
                            }
                            event_type__ = map_.next_value()?;
                        }
                        GeneratedField::Contents => {
                            if contents__.is_some() {
                                return Err(serde::de::Error::duplicate_field("contents"));
                            }
                            contents__ = map_.next_value()?;
                        }
                        GeneratedField::Json => {
                            if json__.is_some() {
                                return Err(serde::de::Error::duplicate_field("json"));
                            }
                            json__ = map_.next_value::<::std::option::Option<crate::utils::_serde::ValueDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Event {
                    package_id: package_id__,
                    module: module__,
                    sender: sender__,
                    event_type: event_type__,
                    contents: contents__,
                    json: json__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Event", FIELDS, GeneratedVisitor)
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
        if self.events.is_some() {
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
        if let Some(v) = self.events.as_ref() {
            struct_ser.serialize_field("events", v)?;
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
            "events",
            "checkpoint",
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
            Events,
            Checkpoint,
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
                            "events" => Ok(GeneratedField::Events),
                            "checkpoint" => Ok(GeneratedField::Checkpoint),
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
                let mut events__ = None;
                let mut checkpoint__ = None;
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
                        GeneratedField::Events => {
                            if events__.is_some() {
                                return Err(serde::de::Error::duplicate_field("events"));
                            }
                            events__ = map_.next_value()?;
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
                    events: events__,
                    checkpoint: checkpoint__,
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
        if self.command.is_some() {
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
        if let Some(v) = self.command.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("command", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.kind.as_ref() {
            let v = execution_error::ExecutionErrorKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.error_details.as_ref() {
            match v {
                execution_error::ErrorDetails::Abort(v) => {
                    struct_ser.serialize_field("abort", v)?;
                }
                execution_error::ErrorDetails::SizeError(v) => {
                    struct_ser.serialize_field("sizeError", v)?;
                }
                execution_error::ErrorDetails::CommandArgumentError(v) => {
                    struct_ser.serialize_field("commandArgumentError", v)?;
                }
                execution_error::ErrorDetails::TypeArgumentError(v) => {
                    struct_ser.serialize_field("typeArgumentError", v)?;
                }
                execution_error::ErrorDetails::PackageUpgradeError(v) => {
                    struct_ser.serialize_field("packageUpgradeError", v)?;
                }
                execution_error::ErrorDetails::IndexError(v) => {
                    struct_ser.serialize_field("indexError", v)?;
                }
                execution_error::ErrorDetails::ObjectId(v) => {
                    struct_ser.serialize_field("objectId", v)?;
                }
                execution_error::ErrorDetails::CoinDenyListError(v) => {
                    struct_ser.serialize_field("coinDenyListError", v)?;
                }
                execution_error::ErrorDetails::CongestedObjects(v) => {
                    struct_ser.serialize_field("congestedObjects", v)?;
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
            "command",
            "kind",
            "abort",
            "size_error",
            "sizeError",
            "command_argument_error",
            "commandArgumentError",
            "type_argument_error",
            "typeArgumentError",
            "package_upgrade_error",
            "packageUpgradeError",
            "index_error",
            "indexError",
            "object_id",
            "objectId",
            "coin_deny_list_error",
            "coinDenyListError",
            "congested_objects",
            "congestedObjects",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Description,
            Command,
            Kind,
            Abort,
            SizeError,
            CommandArgumentError,
            TypeArgumentError,
            PackageUpgradeError,
            IndexError,
            ObjectId,
            CoinDenyListError,
            CongestedObjects,
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
                            "command" => Ok(GeneratedField::Command),
                            "kind" => Ok(GeneratedField::Kind),
                            "abort" => Ok(GeneratedField::Abort),
                            "sizeError" | "size_error" => Ok(GeneratedField::SizeError),
                            "commandArgumentError" | "command_argument_error" => Ok(GeneratedField::CommandArgumentError),
                            "typeArgumentError" | "type_argument_error" => Ok(GeneratedField::TypeArgumentError),
                            "packageUpgradeError" | "package_upgrade_error" => Ok(GeneratedField::PackageUpgradeError),
                            "indexError" | "index_error" => Ok(GeneratedField::IndexError),
                            "objectId" | "object_id" => Ok(GeneratedField::ObjectId),
                            "coinDenyListError" | "coin_deny_list_error" => Ok(GeneratedField::CoinDenyListError),
                            "congestedObjects" | "congested_objects" => Ok(GeneratedField::CongestedObjects),
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
                let mut command__ = None;
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
                        GeneratedField::Command => {
                            if command__.is_some() {
                                return Err(serde::de::Error::duplicate_field("command"));
                            }
                            command__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<execution_error::ExecutionErrorKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::Abort => {
                            if error_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("abort"));
                            }
                            error_details__ = map_.next_value::<::std::option::Option<_>>()?.map(execution_error::ErrorDetails::Abort)
;
                        }
                        GeneratedField::SizeError => {
                            if error_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("sizeError"));
                            }
                            error_details__ = map_.next_value::<::std::option::Option<_>>()?.map(execution_error::ErrorDetails::SizeError)
;
                        }
                        GeneratedField::CommandArgumentError => {
                            if error_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commandArgumentError"));
                            }
                            error_details__ = map_.next_value::<::std::option::Option<_>>()?.map(execution_error::ErrorDetails::CommandArgumentError)
;
                        }
                        GeneratedField::TypeArgumentError => {
                            if error_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("typeArgumentError"));
                            }
                            error_details__ = map_.next_value::<::std::option::Option<_>>()?.map(execution_error::ErrorDetails::TypeArgumentError)
;
                        }
                        GeneratedField::PackageUpgradeError => {
                            if error_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("packageUpgradeError"));
                            }
                            error_details__ = map_.next_value::<::std::option::Option<_>>()?.map(execution_error::ErrorDetails::PackageUpgradeError)
;
                        }
                        GeneratedField::IndexError => {
                            if error_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("indexError"));
                            }
                            error_details__ = map_.next_value::<::std::option::Option<_>>()?.map(execution_error::ErrorDetails::IndexError)
;
                        }
                        GeneratedField::ObjectId => {
                            if error_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectId"));
                            }
                            error_details__ = map_.next_value::<::std::option::Option<_>>()?.map(execution_error::ErrorDetails::ObjectId);
                        }
                        GeneratedField::CoinDenyListError => {
                            if error_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coinDenyListError"));
                            }
                            error_details__ = map_.next_value::<::std::option::Option<_>>()?.map(execution_error::ErrorDetails::CoinDenyListError)
;
                        }
                        GeneratedField::CongestedObjects => {
                            if error_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("congestedObjects"));
                            }
                            error_details__ = map_.next_value::<::std::option::Option<_>>()?.map(execution_error::ErrorDetails::CongestedObjects)
;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ExecutionError {
                    description: description__,
                    command: command__,
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
            Self::InvalidGasObject => "INVALID_GAS_OBJECT",
            Self::InvariantViolation => "INVARIANT_VIOLATION",
            Self::FeatureNotYetSupported => "FEATURE_NOT_YET_SUPPORTED",
            Self::ObjectTooBig => "OBJECT_TOO_BIG",
            Self::PackageTooBig => "PACKAGE_TOO_BIG",
            Self::CircularObjectOwnership => "CIRCULAR_OBJECT_OWNERSHIP",
            Self::InsufficientCoinBalance => "INSUFFICIENT_COIN_BALANCE",
            Self::CoinBalanceOverflow => "COIN_BALANCE_OVERFLOW",
            Self::PublishErrorNonZeroAddress => "PUBLISH_ERROR_NON_ZERO_ADDRESS",
            Self::SuiMoveVerificationError => "SUI_MOVE_VERIFICATION_ERROR",
            Self::MovePrimitiveRuntimeError => "MOVE_PRIMITIVE_RUNTIME_ERROR",
            Self::MoveAbort => "MOVE_ABORT",
            Self::VmVerificationOrDeserializationError => "VM_VERIFICATION_OR_DESERIALIZATION_ERROR",
            Self::VmInvariantViolation => "VM_INVARIANT_VIOLATION",
            Self::FunctionNotFound => "FUNCTION_NOT_FOUND",
            Self::ArityMismatch => "ARITY_MISMATCH",
            Self::TypeArityMismatch => "TYPE_ARITY_MISMATCH",
            Self::NonEntryFunctionInvoked => "NON_ENTRY_FUNCTION_INVOKED",
            Self::CommandArgumentError => "COMMAND_ARGUMENT_ERROR",
            Self::TypeArgumentError => "TYPE_ARGUMENT_ERROR",
            Self::UnusedValueWithoutDrop => "UNUSED_VALUE_WITHOUT_DROP",
            Self::InvalidPublicFunctionReturnType => "INVALID_PUBLIC_FUNCTION_RETURN_TYPE",
            Self::InvalidTransferObject => "INVALID_TRANSFER_OBJECT",
            Self::EffectsTooLarge => "EFFECTS_TOO_LARGE",
            Self::PublishUpgradeMissingDependency => "PUBLISH_UPGRADE_MISSING_DEPENDENCY",
            Self::PublishUpgradeDependencyDowngrade => "PUBLISH_UPGRADE_DEPENDENCY_DOWNGRADE",
            Self::PackageUpgradeError => "PACKAGE_UPGRADE_ERROR",
            Self::WrittenObjectsTooLarge => "WRITTEN_OBJECTS_TOO_LARGE",
            Self::CertificateDenied => "CERTIFICATE_DENIED",
            Self::SuiMoveVerificationTimedout => "SUI_MOVE_VERIFICATION_TIMEDOUT",
            Self::ConsensusObjectOperationNotAllowed => "CONSENSUS_OBJECT_OPERATION_NOT_ALLOWED",
            Self::InputObjectDeleted => "INPUT_OBJECT_DELETED",
            Self::ExecutionCanceledDueToConsensusObjectCongestion => "EXECUTION_CANCELED_DUE_TO_CONSENSUS_OBJECT_CONGESTION",
            Self::AddressDeniedForCoin => "ADDRESS_DENIED_FOR_COIN",
            Self::CoinTypeGlobalPause => "COIN_TYPE_GLOBAL_PAUSE",
            Self::ExecutionCanceledDueToRandomnessUnavailable => "EXECUTION_CANCELED_DUE_TO_RANDOMNESS_UNAVAILABLE",
            Self::MoveVectorElemTooBig => "MOVE_VECTOR_ELEM_TOO_BIG",
            Self::MoveRawValueTooBig => "MOVE_RAW_VALUE_TOO_BIG",
            Self::InvalidLinkage => "INVALID_LINKAGE",
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
            "INVALID_GAS_OBJECT",
            "INVARIANT_VIOLATION",
            "FEATURE_NOT_YET_SUPPORTED",
            "OBJECT_TOO_BIG",
            "PACKAGE_TOO_BIG",
            "CIRCULAR_OBJECT_OWNERSHIP",
            "INSUFFICIENT_COIN_BALANCE",
            "COIN_BALANCE_OVERFLOW",
            "PUBLISH_ERROR_NON_ZERO_ADDRESS",
            "SUI_MOVE_VERIFICATION_ERROR",
            "MOVE_PRIMITIVE_RUNTIME_ERROR",
            "MOVE_ABORT",
            "VM_VERIFICATION_OR_DESERIALIZATION_ERROR",
            "VM_INVARIANT_VIOLATION",
            "FUNCTION_NOT_FOUND",
            "ARITY_MISMATCH",
            "TYPE_ARITY_MISMATCH",
            "NON_ENTRY_FUNCTION_INVOKED",
            "COMMAND_ARGUMENT_ERROR",
            "TYPE_ARGUMENT_ERROR",
            "UNUSED_VALUE_WITHOUT_DROP",
            "INVALID_PUBLIC_FUNCTION_RETURN_TYPE",
            "INVALID_TRANSFER_OBJECT",
            "EFFECTS_TOO_LARGE",
            "PUBLISH_UPGRADE_MISSING_DEPENDENCY",
            "PUBLISH_UPGRADE_DEPENDENCY_DOWNGRADE",
            "PACKAGE_UPGRADE_ERROR",
            "WRITTEN_OBJECTS_TOO_LARGE",
            "CERTIFICATE_DENIED",
            "SUI_MOVE_VERIFICATION_TIMEDOUT",
            "CONSENSUS_OBJECT_OPERATION_NOT_ALLOWED",
            "INPUT_OBJECT_DELETED",
            "EXECUTION_CANCELED_DUE_TO_CONSENSUS_OBJECT_CONGESTION",
            "ADDRESS_DENIED_FOR_COIN",
            "COIN_TYPE_GLOBAL_PAUSE",
            "EXECUTION_CANCELED_DUE_TO_RANDOMNESS_UNAVAILABLE",
            "MOVE_VECTOR_ELEM_TOO_BIG",
            "MOVE_RAW_VALUE_TOO_BIG",
            "INVALID_LINKAGE",
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
                    "INVALID_GAS_OBJECT" => Ok(execution_error::ExecutionErrorKind::InvalidGasObject),
                    "INVARIANT_VIOLATION" => Ok(execution_error::ExecutionErrorKind::InvariantViolation),
                    "FEATURE_NOT_YET_SUPPORTED" => Ok(execution_error::ExecutionErrorKind::FeatureNotYetSupported),
                    "OBJECT_TOO_BIG" => Ok(execution_error::ExecutionErrorKind::ObjectTooBig),
                    "PACKAGE_TOO_BIG" => Ok(execution_error::ExecutionErrorKind::PackageTooBig),
                    "CIRCULAR_OBJECT_OWNERSHIP" => Ok(execution_error::ExecutionErrorKind::CircularObjectOwnership),
                    "INSUFFICIENT_COIN_BALANCE" => Ok(execution_error::ExecutionErrorKind::InsufficientCoinBalance),
                    "COIN_BALANCE_OVERFLOW" => Ok(execution_error::ExecutionErrorKind::CoinBalanceOverflow),
                    "PUBLISH_ERROR_NON_ZERO_ADDRESS" => Ok(execution_error::ExecutionErrorKind::PublishErrorNonZeroAddress),
                    "SUI_MOVE_VERIFICATION_ERROR" => Ok(execution_error::ExecutionErrorKind::SuiMoveVerificationError),
                    "MOVE_PRIMITIVE_RUNTIME_ERROR" => Ok(execution_error::ExecutionErrorKind::MovePrimitiveRuntimeError),
                    "MOVE_ABORT" => Ok(execution_error::ExecutionErrorKind::MoveAbort),
                    "VM_VERIFICATION_OR_DESERIALIZATION_ERROR" => Ok(execution_error::ExecutionErrorKind::VmVerificationOrDeserializationError),
                    "VM_INVARIANT_VIOLATION" => Ok(execution_error::ExecutionErrorKind::VmInvariantViolation),
                    "FUNCTION_NOT_FOUND" => Ok(execution_error::ExecutionErrorKind::FunctionNotFound),
                    "ARITY_MISMATCH" => Ok(execution_error::ExecutionErrorKind::ArityMismatch),
                    "TYPE_ARITY_MISMATCH" => Ok(execution_error::ExecutionErrorKind::TypeArityMismatch),
                    "NON_ENTRY_FUNCTION_INVOKED" => Ok(execution_error::ExecutionErrorKind::NonEntryFunctionInvoked),
                    "COMMAND_ARGUMENT_ERROR" => Ok(execution_error::ExecutionErrorKind::CommandArgumentError),
                    "TYPE_ARGUMENT_ERROR" => Ok(execution_error::ExecutionErrorKind::TypeArgumentError),
                    "UNUSED_VALUE_WITHOUT_DROP" => Ok(execution_error::ExecutionErrorKind::UnusedValueWithoutDrop),
                    "INVALID_PUBLIC_FUNCTION_RETURN_TYPE" => Ok(execution_error::ExecutionErrorKind::InvalidPublicFunctionReturnType),
                    "INVALID_TRANSFER_OBJECT" => Ok(execution_error::ExecutionErrorKind::InvalidTransferObject),
                    "EFFECTS_TOO_LARGE" => Ok(execution_error::ExecutionErrorKind::EffectsTooLarge),
                    "PUBLISH_UPGRADE_MISSING_DEPENDENCY" => Ok(execution_error::ExecutionErrorKind::PublishUpgradeMissingDependency),
                    "PUBLISH_UPGRADE_DEPENDENCY_DOWNGRADE" => Ok(execution_error::ExecutionErrorKind::PublishUpgradeDependencyDowngrade),
                    "PACKAGE_UPGRADE_ERROR" => Ok(execution_error::ExecutionErrorKind::PackageUpgradeError),
                    "WRITTEN_OBJECTS_TOO_LARGE" => Ok(execution_error::ExecutionErrorKind::WrittenObjectsTooLarge),
                    "CERTIFICATE_DENIED" => Ok(execution_error::ExecutionErrorKind::CertificateDenied),
                    "SUI_MOVE_VERIFICATION_TIMEDOUT" => Ok(execution_error::ExecutionErrorKind::SuiMoveVerificationTimedout),
                    "CONSENSUS_OBJECT_OPERATION_NOT_ALLOWED" => Ok(execution_error::ExecutionErrorKind::ConsensusObjectOperationNotAllowed),
                    "INPUT_OBJECT_DELETED" => Ok(execution_error::ExecutionErrorKind::InputObjectDeleted),
                    "EXECUTION_CANCELED_DUE_TO_CONSENSUS_OBJECT_CONGESTION" => Ok(execution_error::ExecutionErrorKind::ExecutionCanceledDueToConsensusObjectCongestion),
                    "ADDRESS_DENIED_FOR_COIN" => Ok(execution_error::ExecutionErrorKind::AddressDeniedForCoin),
                    "COIN_TYPE_GLOBAL_PAUSE" => Ok(execution_error::ExecutionErrorKind::CoinTypeGlobalPause),
                    "EXECUTION_CANCELED_DUE_TO_RANDOMNESS_UNAVAILABLE" => Ok(execution_error::ExecutionErrorKind::ExecutionCanceledDueToRandomnessUnavailable),
                    "MOVE_VECTOR_ELEM_TOO_BIG" => Ok(execution_error::ExecutionErrorKind::MoveVectorElemTooBig),
                    "MOVE_RAW_VALUE_TOO_BIG" => Ok(execution_error::ExecutionErrorKind::MoveRawValueTooBig),
                    "INVALID_LINKAGE" => Ok(execution_error::ExecutionErrorKind::InvalidLinkage),
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
impl serde::Serialize for ExecutionTimeObservation {
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
        if self.move_entry_point.is_some() {
            len += 1;
        }
        if !self.validator_observations.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ExecutionTimeObservation", len)?;
        if let Some(v) = self.kind.as_ref() {
            let v = execution_time_observation::ExecutionTimeObservationKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.move_entry_point.as_ref() {
            struct_ser.serialize_field("moveEntryPoint", v)?;
        }
        if !self.validator_observations.is_empty() {
            struct_ser.serialize_field("validatorObservations", &self.validator_observations)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ExecutionTimeObservation {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "move_entry_point",
            "moveEntryPoint",
            "validator_observations",
            "validatorObservations",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            MoveEntryPoint,
            ValidatorObservations,
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
                            "moveEntryPoint" | "move_entry_point" => Ok(GeneratedField::MoveEntryPoint),
                            "validatorObservations" | "validator_observations" => Ok(GeneratedField::ValidatorObservations),
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
            type Value = ExecutionTimeObservation;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ExecutionTimeObservation")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ExecutionTimeObservation, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut move_entry_point__ = None;
                let mut validator_observations__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<execution_time_observation::ExecutionTimeObservationKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::MoveEntryPoint => {
                            if move_entry_point__.is_some() {
                                return Err(serde::de::Error::duplicate_field("moveEntryPoint"));
                            }
                            move_entry_point__ = map_.next_value()?;
                        }
                        GeneratedField::ValidatorObservations => {
                            if validator_observations__.is_some() {
                                return Err(serde::de::Error::duplicate_field("validatorObservations"));
                            }
                            validator_observations__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ExecutionTimeObservation {
                    kind: kind__,
                    move_entry_point: move_entry_point__,
                    validator_observations: validator_observations__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ExecutionTimeObservation", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for execution_time_observation::ExecutionTimeObservationKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "EXECUTION_TIME_OBSERVATION_KIND_UNKNOWN",
            Self::MoveEntryPoint => "MOVE_ENTRY_POINT",
            Self::TransferObjects => "TRANSFER_OBJECTS",
            Self::SplitCoins => "SPLIT_COINS",
            Self::MergeCoins => "MERGE_COINS",
            Self::Publish => "PUBLISH",
            Self::MakeMoveVector => "MAKE_MOVE_VECTOR",
            Self::Upgrade => "UPGRADE",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for execution_time_observation::ExecutionTimeObservationKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "EXECUTION_TIME_OBSERVATION_KIND_UNKNOWN",
            "MOVE_ENTRY_POINT",
            "TRANSFER_OBJECTS",
            "SPLIT_COINS",
            "MERGE_COINS",
            "PUBLISH",
            "MAKE_MOVE_VECTOR",
            "UPGRADE",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = execution_time_observation::ExecutionTimeObservationKind;

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
                    "EXECUTION_TIME_OBSERVATION_KIND_UNKNOWN" => Ok(execution_time_observation::ExecutionTimeObservationKind::Unknown),
                    "MOVE_ENTRY_POINT" => Ok(execution_time_observation::ExecutionTimeObservationKind::MoveEntryPoint),
                    "TRANSFER_OBJECTS" => Ok(execution_time_observation::ExecutionTimeObservationKind::TransferObjects),
                    "SPLIT_COINS" => Ok(execution_time_observation::ExecutionTimeObservationKind::SplitCoins),
                    "MERGE_COINS" => Ok(execution_time_observation::ExecutionTimeObservationKind::MergeCoins),
                    "PUBLISH" => Ok(execution_time_observation::ExecutionTimeObservationKind::Publish),
                    "MAKE_MOVE_VECTOR" => Ok(execution_time_observation::ExecutionTimeObservationKind::MakeMoveVector),
                    "UPGRADE" => Ok(execution_time_observation::ExecutionTimeObservationKind::Upgrade),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for ExecutionTimeObservations {
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
        if !self.observations.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ExecutionTimeObservations", len)?;
        if let Some(v) = self.version.as_ref() {
            struct_ser.serialize_field("version", v)?;
        }
        if !self.observations.is_empty() {
            struct_ser.serialize_field("observations", &self.observations)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ExecutionTimeObservations {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "version",
            "observations",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Version,
            Observations,
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
                            "version" => Ok(GeneratedField::Version),
                            "observations" => Ok(GeneratedField::Observations),
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
            type Value = ExecutionTimeObservations;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ExecutionTimeObservations")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ExecutionTimeObservations, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut version__ = None;
                let mut observations__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Observations => {
                            if observations__.is_some() {
                                return Err(serde::de::Error::duplicate_field("observations"));
                            }
                            observations__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ExecutionTimeObservations {
                    version: version__,
                    observations: observations__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ExecutionTimeObservations", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for FieldDescriptor {
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
        if self.position.is_some() {
            len += 1;
        }
        if self.r#type.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.FieldDescriptor", len)?;
        if let Some(v) = self.name.as_ref() {
            struct_ser.serialize_field("name", v)?;
        }
        if let Some(v) = self.position.as_ref() {
            struct_ser.serialize_field("position", v)?;
        }
        if let Some(v) = self.r#type.as_ref() {
            struct_ser.serialize_field("type", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for FieldDescriptor {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "name",
            "position",
            "type",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Name,
            Position,
            Type,
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
                            "position" => Ok(GeneratedField::Position),
                            "type" => Ok(GeneratedField::Type),
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
            type Value = FieldDescriptor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.FieldDescriptor")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<FieldDescriptor, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut name__ = None;
                let mut position__ = None;
                let mut r#type__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Name => {
                            if name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("name"));
                            }
                            name__ = map_.next_value()?;
                        }
                        GeneratedField::Position => {
                            if position__.is_some() {
                                return Err(serde::de::Error::duplicate_field("position"));
                            }
                            position__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Type => {
                            if r#type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("type"));
                            }
                            r#type__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(FieldDescriptor {
                    name: name__,
                    position: position__,
                    r#type: r#type__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.FieldDescriptor", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for FunctionDescriptor {
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
        if self.visibility.is_some() {
            len += 1;
        }
        if self.is_entry.is_some() {
            len += 1;
        }
        if !self.type_parameters.is_empty() {
            len += 1;
        }
        if !self.parameters.is_empty() {
            len += 1;
        }
        if !self.returns.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.FunctionDescriptor", len)?;
        if let Some(v) = self.name.as_ref() {
            struct_ser.serialize_field("name", v)?;
        }
        if let Some(v) = self.visibility.as_ref() {
            let v = function_descriptor::Visibility::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("visibility", &v)?;
        }
        if let Some(v) = self.is_entry.as_ref() {
            struct_ser.serialize_field("isEntry", v)?;
        }
        if !self.type_parameters.is_empty() {
            struct_ser.serialize_field("typeParameters", &self.type_parameters)?;
        }
        if !self.parameters.is_empty() {
            struct_ser.serialize_field("parameters", &self.parameters)?;
        }
        if !self.returns.is_empty() {
            struct_ser.serialize_field("returns", &self.returns)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for FunctionDescriptor {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "name",
            "visibility",
            "is_entry",
            "isEntry",
            "type_parameters",
            "typeParameters",
            "parameters",
            "returns",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Name,
            Visibility,
            IsEntry,
            TypeParameters,
            Parameters,
            Returns,
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
                            "visibility" => Ok(GeneratedField::Visibility),
                            "isEntry" | "is_entry" => Ok(GeneratedField::IsEntry),
                            "typeParameters" | "type_parameters" => Ok(GeneratedField::TypeParameters),
                            "parameters" => Ok(GeneratedField::Parameters),
                            "returns" => Ok(GeneratedField::Returns),
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
            type Value = FunctionDescriptor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.FunctionDescriptor")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<FunctionDescriptor, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut name__ = None;
                let mut visibility__ = None;
                let mut is_entry__ = None;
                let mut type_parameters__ = None;
                let mut parameters__ = None;
                let mut returns__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Name => {
                            if name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("name"));
                            }
                            name__ = map_.next_value()?;
                        }
                        GeneratedField::Visibility => {
                            if visibility__.is_some() {
                                return Err(serde::de::Error::duplicate_field("visibility"));
                            }
                            visibility__ = map_.next_value::<::std::option::Option<function_descriptor::Visibility>>()?.map(|x| x as i32);
                        }
                        GeneratedField::IsEntry => {
                            if is_entry__.is_some() {
                                return Err(serde::de::Error::duplicate_field("isEntry"));
                            }
                            is_entry__ = map_.next_value()?;
                        }
                        GeneratedField::TypeParameters => {
                            if type_parameters__.is_some() {
                                return Err(serde::de::Error::duplicate_field("typeParameters"));
                            }
                            type_parameters__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Parameters => {
                            if parameters__.is_some() {
                                return Err(serde::de::Error::duplicate_field("parameters"));
                            }
                            parameters__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Returns => {
                            if returns__.is_some() {
                                return Err(serde::de::Error::duplicate_field("returns"));
                            }
                            returns__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(FunctionDescriptor {
                    name: name__,
                    visibility: visibility__,
                    is_entry: is_entry__,
                    type_parameters: type_parameters__.unwrap_or_default(),
                    parameters: parameters__.unwrap_or_default(),
                    returns: returns__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.FunctionDescriptor", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for function_descriptor::Visibility {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "VISIBILITY_UNKNOWN",
            Self::Private => "PRIVATE",
            Self::Public => "PUBLIC",
            Self::Friend => "FRIEND",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for function_descriptor::Visibility {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "VISIBILITY_UNKNOWN",
            "PRIVATE",
            "PUBLIC",
            "FRIEND",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = function_descriptor::Visibility;

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
                    "VISIBILITY_UNKNOWN" => Ok(function_descriptor::Visibility::Unknown),
                    "PRIVATE" => Ok(function_descriptor::Visibility::Private),
                    "PUBLIC" => Ok(function_descriptor::Visibility::Public),
                    "FRIEND" => Ok(function_descriptor::Visibility::Friend),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for GasCostSummary {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.computation_cost.is_some() {
            len += 1;
        }
        if self.storage_cost.is_some() {
            len += 1;
        }
        if self.storage_rebate.is_some() {
            len += 1;
        }
        if self.non_refundable_storage_fee.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GasCostSummary", len)?;
        if let Some(v) = self.computation_cost.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("computationCost", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.storage_cost.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("storageCost", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.storage_rebate.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("storageRebate", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.non_refundable_storage_fee.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("nonRefundableStorageFee", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GasCostSummary {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "computation_cost",
            "computationCost",
            "storage_cost",
            "storageCost",
            "storage_rebate",
            "storageRebate",
            "non_refundable_storage_fee",
            "nonRefundableStorageFee",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ComputationCost,
            StorageCost,
            StorageRebate,
            NonRefundableStorageFee,
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
                            "computationCost" | "computation_cost" => Ok(GeneratedField::ComputationCost),
                            "storageCost" | "storage_cost" => Ok(GeneratedField::StorageCost),
                            "storageRebate" | "storage_rebate" => Ok(GeneratedField::StorageRebate),
                            "nonRefundableStorageFee" | "non_refundable_storage_fee" => Ok(GeneratedField::NonRefundableStorageFee),
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
            type Value = GasCostSummary;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.GasCostSummary")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GasCostSummary, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut computation_cost__ = None;
                let mut storage_cost__ = None;
                let mut storage_rebate__ = None;
                let mut non_refundable_storage_fee__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ComputationCost => {
                            if computation_cost__.is_some() {
                                return Err(serde::de::Error::duplicate_field("computationCost"));
                            }
                            computation_cost__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::StorageCost => {
                            if storage_cost__.is_some() {
                                return Err(serde::de::Error::duplicate_field("storageCost"));
                            }
                            storage_cost__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::StorageRebate => {
                            if storage_rebate__.is_some() {
                                return Err(serde::de::Error::duplicate_field("storageRebate"));
                            }
                            storage_rebate__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::NonRefundableStorageFee => {
                            if non_refundable_storage_fee__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nonRefundableStorageFee"));
                            }
                            non_refundable_storage_fee__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GasCostSummary {
                    computation_cost: computation_cost__,
                    storage_cost: storage_cost__,
                    storage_rebate: storage_rebate__,
                    non_refundable_storage_fee: non_refundable_storage_fee__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.GasCostSummary", FIELDS, GeneratedVisitor)
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
        if self.price.is_some() {
            len += 1;
        }
        if self.budget.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.GasPayment", len)?;
        if !self.objects.is_empty() {
            struct_ser.serialize_field("objects", &self.objects)?;
        }
        if let Some(v) = self.owner.as_ref() {
            struct_ser.serialize_field("owner", v)?;
        }
        if let Some(v) = self.price.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("price", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.budget.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("budget", ToString::to_string(&v).as_str())?;
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
            "price",
            "budget",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Objects,
            Owner,
            Price,
            Budget,
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
                            "price" => Ok(GeneratedField::Price),
                            "budget" => Ok(GeneratedField::Budget),
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
                let mut price__ = None;
                let mut budget__ = None;
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
                        GeneratedField::Price => {
                            if price__.is_some() {
                                return Err(serde::de::Error::duplicate_field("price"));
                            }
                            price__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Budget => {
                            if budget__.is_some() {
                                return Err(serde::de::Error::duplicate_field("budget"));
                            }
                            budget__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(GasPayment {
                    objects: objects__.unwrap_or_default(),
                    owner: owner__,
                    price: price__,
                    budget: budget__,
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
impl serde::Serialize for IndexError {
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
        if self.subresult.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.IndexError", len)?;
        if let Some(v) = self.index.as_ref() {
            struct_ser.serialize_field("index", v)?;
        }
        if let Some(v) = self.subresult.as_ref() {
            struct_ser.serialize_field("subresult", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for IndexError {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "index",
            "subresult",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Index,
            Subresult,
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
                            "subresult" => Ok(GeneratedField::Subresult),
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
            type Value = IndexError;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.IndexError")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<IndexError, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut index__ = None;
                let mut subresult__ = None;
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
                        GeneratedField::Subresult => {
                            if subresult__.is_some() {
                                return Err(serde::de::Error::duplicate_field("subresult"));
                            }
                            subresult__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(IndexError {
                    index: index__,
                    subresult: subresult__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.IndexError", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Input {
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
        if self.pure.is_some() {
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
        if self.mutable.is_some() {
            len += 1;
        }
        if self.literal.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Input", len)?;
        if let Some(v) = self.kind.as_ref() {
            let v = input::InputKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.pure.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("pure", crate::utils::_serde::base64::encode(&v).as_str())?;
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
        if let Some(v) = self.mutable.as_ref() {
            struct_ser.serialize_field("mutable", v)?;
        }
        if let Some(v) = self.literal.as_ref() {
            struct_ser.serialize_field("literal", &crate::utils::_serde::ValueSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Input {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "pure",
            "object_id",
            "objectId",
            "version",
            "digest",
            "mutable",
            "literal",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            Pure,
            ObjectId,
            Version,
            Digest,
            Mutable,
            Literal,
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
                            "pure" => Ok(GeneratedField::Pure),
                            "objectId" | "object_id" => Ok(GeneratedField::ObjectId),
                            "version" => Ok(GeneratedField::Version),
                            "digest" => Ok(GeneratedField::Digest),
                            "mutable" => Ok(GeneratedField::Mutable),
                            "literal" => Ok(GeneratedField::Literal),
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
            type Value = Input;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Input")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Input, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut pure__ = None;
                let mut object_id__ = None;
                let mut version__ = None;
                let mut digest__ = None;
                let mut mutable__ = None;
                let mut literal__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<input::InputKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::Pure => {
                            if pure__.is_some() {
                                return Err(serde::de::Error::duplicate_field("pure"));
                            }
                            pure__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
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
                        GeneratedField::Mutable => {
                            if mutable__.is_some() {
                                return Err(serde::de::Error::duplicate_field("mutable"));
                            }
                            mutable__ = map_.next_value()?;
                        }
                        GeneratedField::Literal => {
                            if literal__.is_some() {
                                return Err(serde::de::Error::duplicate_field("literal"));
                            }
                            literal__ = map_.next_value::<::std::option::Option<crate::utils::_serde::ValueDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Input {
                    kind: kind__,
                    pure: pure__,
                    object_id: object_id__,
                    version: version__,
                    digest: digest__,
                    mutable: mutable__,
                    literal: literal__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Input", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for input::InputKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "INPUT_KIND_UNKNOWN",
            Self::Pure => "PURE",
            Self::ImmutableOrOwned => "IMMUTABLE_OR_OWNED",
            Self::Shared => "SHARED",
            Self::Receiving => "RECEIVING",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for input::InputKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "INPUT_KIND_UNKNOWN",
            "PURE",
            "IMMUTABLE_OR_OWNED",
            "SHARED",
            "RECEIVING",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = input::InputKind;

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
                    "INPUT_KIND_UNKNOWN" => Ok(input::InputKind::Unknown),
                    "PURE" => Ok(input::InputKind::Pure),
                    "IMMUTABLE_OR_OWNED" => Ok(input::InputKind::ImmutableOrOwned),
                    "SHARED" => Ok(input::InputKind::Shared),
                    "RECEIVING" => Ok(input::InputKind::Receiving),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for Jwk {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.kty.is_some() {
            len += 1;
        }
        if self.e.is_some() {
            len += 1;
        }
        if self.n.is_some() {
            len += 1;
        }
        if self.alg.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Jwk", len)?;
        if let Some(v) = self.kty.as_ref() {
            struct_ser.serialize_field("kty", v)?;
        }
        if let Some(v) = self.e.as_ref() {
            struct_ser.serialize_field("e", v)?;
        }
        if let Some(v) = self.n.as_ref() {
            struct_ser.serialize_field("n", v)?;
        }
        if let Some(v) = self.alg.as_ref() {
            struct_ser.serialize_field("alg", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Jwk {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kty",
            "e",
            "n",
            "alg",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kty,
            E,
            N,
            Alg,
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
                            "kty" => Ok(GeneratedField::Kty),
                            "e" => Ok(GeneratedField::E),
                            "n" => Ok(GeneratedField::N),
                            "alg" => Ok(GeneratedField::Alg),
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
            type Value = Jwk;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Jwk")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Jwk, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kty__ = None;
                let mut e__ = None;
                let mut n__ = None;
                let mut alg__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kty => {
                            if kty__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kty"));
                            }
                            kty__ = map_.next_value()?;
                        }
                        GeneratedField::E => {
                            if e__.is_some() {
                                return Err(serde::de::Error::duplicate_field("e"));
                            }
                            e__ = map_.next_value()?;
                        }
                        GeneratedField::N => {
                            if n__.is_some() {
                                return Err(serde::de::Error::duplicate_field("n"));
                            }
                            n__ = map_.next_value()?;
                        }
                        GeneratedField::Alg => {
                            if alg__.is_some() {
                                return Err(serde::de::Error::duplicate_field("alg"));
                            }
                            alg__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Jwk {
                    kty: kty__,
                    e: e__,
                    n: n__,
                    alg: alg__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Jwk", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for JwkId {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.iss.is_some() {
            len += 1;
        }
        if self.kid.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.JwkId", len)?;
        if let Some(v) = self.iss.as_ref() {
            struct_ser.serialize_field("iss", v)?;
        }
        if let Some(v) = self.kid.as_ref() {
            struct_ser.serialize_field("kid", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for JwkId {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "iss",
            "kid",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Iss,
            Kid,
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
                            "iss" => Ok(GeneratedField::Iss),
                            "kid" => Ok(GeneratedField::Kid),
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
            type Value = JwkId;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.JwkId")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<JwkId, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut iss__ = None;
                let mut kid__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Iss => {
                            if iss__.is_some() {
                                return Err(serde::de::Error::duplicate_field("iss"));
                            }
                            iss__ = map_.next_value()?;
                        }
                        GeneratedField::Kid => {
                            if kid__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kid"));
                            }
                            kid__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(JwkId {
                    iss: iss__,
                    kid: kid__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.JwkId", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Linkage {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.original_id.is_some() {
            len += 1;
        }
        if self.upgraded_id.is_some() {
            len += 1;
        }
        if self.upgraded_version.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Linkage", len)?;
        if let Some(v) = self.original_id.as_ref() {
            struct_ser.serialize_field("originalId", v)?;
        }
        if let Some(v) = self.upgraded_id.as_ref() {
            struct_ser.serialize_field("upgradedId", v)?;
        }
        if let Some(v) = self.upgraded_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("upgradedVersion", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Linkage {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "original_id",
            "originalId",
            "upgraded_id",
            "upgradedId",
            "upgraded_version",
            "upgradedVersion",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            OriginalId,
            UpgradedId,
            UpgradedVersion,
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
                            "originalId" | "original_id" => Ok(GeneratedField::OriginalId),
                            "upgradedId" | "upgraded_id" => Ok(GeneratedField::UpgradedId),
                            "upgradedVersion" | "upgraded_version" => Ok(GeneratedField::UpgradedVersion),
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
            type Value = Linkage;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Linkage")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Linkage, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut original_id__ = None;
                let mut upgraded_id__ = None;
                let mut upgraded_version__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::OriginalId => {
                            if original_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("originalId"));
                            }
                            original_id__ = map_.next_value()?;
                        }
                        GeneratedField::UpgradedId => {
                            if upgraded_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("upgradedId"));
                            }
                            upgraded_id__ = map_.next_value()?;
                        }
                        GeneratedField::UpgradedVersion => {
                            if upgraded_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("upgradedVersion"));
                            }
                            upgraded_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Linkage {
                    original_id: original_id__,
                    upgraded_id: upgraded_id__,
                    upgraded_version: upgraded_version__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Linkage", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for MakeMoveVector {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.element_type.is_some() {
            len += 1;
        }
        if !self.elements.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.MakeMoveVector", len)?;
        if let Some(v) = self.element_type.as_ref() {
            struct_ser.serialize_field("elementType", v)?;
        }
        if !self.elements.is_empty() {
            struct_ser.serialize_field("elements", &self.elements)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MakeMoveVector {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "element_type",
            "elementType",
            "elements",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ElementType,
            Elements,
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
                            "elementType" | "element_type" => Ok(GeneratedField::ElementType),
                            "elements" => Ok(GeneratedField::Elements),
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
            type Value = MakeMoveVector;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.MakeMoveVector")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MakeMoveVector, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut element_type__ = None;
                let mut elements__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ElementType => {
                            if element_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("elementType"));
                            }
                            element_type__ = map_.next_value()?;
                        }
                        GeneratedField::Elements => {
                            if elements__.is_some() {
                                return Err(serde::de::Error::duplicate_field("elements"));
                            }
                            elements__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MakeMoveVector {
                    element_type: element_type__,
                    elements: elements__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.MakeMoveVector", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for MergeCoins {
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
        if !self.coins_to_merge.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.MergeCoins", len)?;
        if let Some(v) = self.coin.as_ref() {
            struct_ser.serialize_field("coin", v)?;
        }
        if !self.coins_to_merge.is_empty() {
            struct_ser.serialize_field("coinsToMerge", &self.coins_to_merge)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MergeCoins {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "coin",
            "coins_to_merge",
            "coinsToMerge",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Coin,
            CoinsToMerge,
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
                            "coinsToMerge" | "coins_to_merge" => Ok(GeneratedField::CoinsToMerge),
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
            type Value = MergeCoins;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.MergeCoins")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MergeCoins, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut coin__ = None;
                let mut coins_to_merge__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Coin => {
                            if coin__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coin"));
                            }
                            coin__ = map_.next_value()?;
                        }
                        GeneratedField::CoinsToMerge => {
                            if coins_to_merge__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coinsToMerge"));
                            }
                            coins_to_merge__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MergeCoins {
                    coin: coin__,
                    coins_to_merge: coins_to_merge__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.MergeCoins", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Module {
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
        if self.contents.is_some() {
            len += 1;
        }
        if !self.datatypes.is_empty() {
            len += 1;
        }
        if !self.functions.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Module", len)?;
        if let Some(v) = self.name.as_ref() {
            struct_ser.serialize_field("name", v)?;
        }
        if let Some(v) = self.contents.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("contents", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if !self.datatypes.is_empty() {
            struct_ser.serialize_field("datatypes", &self.datatypes)?;
        }
        if !self.functions.is_empty() {
            struct_ser.serialize_field("functions", &self.functions)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Module {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "name",
            "contents",
            "datatypes",
            "functions",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Name,
            Contents,
            Datatypes,
            Functions,
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
                            "contents" => Ok(GeneratedField::Contents),
                            "datatypes" => Ok(GeneratedField::Datatypes),
                            "functions" => Ok(GeneratedField::Functions),
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
            type Value = Module;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Module")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Module, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut name__ = None;
                let mut contents__ = None;
                let mut datatypes__ = None;
                let mut functions__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Name => {
                            if name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("name"));
                            }
                            name__ = map_.next_value()?;
                        }
                        GeneratedField::Contents => {
                            if contents__.is_some() {
                                return Err(serde::de::Error::duplicate_field("contents"));
                            }
                            contents__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Datatypes => {
                            if datatypes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("datatypes"));
                            }
                            datatypes__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Functions => {
                            if functions__.is_some() {
                                return Err(serde::de::Error::duplicate_field("functions"));
                            }
                            functions__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Module {
                    name: name__,
                    contents: contents__,
                    datatypes: datatypes__.unwrap_or_default(),
                    functions: functions__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Module", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for MoveAbort {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.abort_code.is_some() {
            len += 1;
        }
        if self.location.is_some() {
            len += 1;
        }
        if self.clever_error.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.MoveAbort", len)?;
        if let Some(v) = self.abort_code.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("abortCode", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.location.as_ref() {
            struct_ser.serialize_field("location", v)?;
        }
        if let Some(v) = self.clever_error.as_ref() {
            struct_ser.serialize_field("cleverError", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MoveAbort {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "abort_code",
            "abortCode",
            "location",
            "clever_error",
            "cleverError",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            AbortCode,
            Location,
            CleverError,
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
                            "abortCode" | "abort_code" => Ok(GeneratedField::AbortCode),
                            "location" => Ok(GeneratedField::Location),
                            "cleverError" | "clever_error" => Ok(GeneratedField::CleverError),
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
            type Value = MoveAbort;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.MoveAbort")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MoveAbort, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut abort_code__ = None;
                let mut location__ = None;
                let mut clever_error__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::AbortCode => {
                            if abort_code__.is_some() {
                                return Err(serde::de::Error::duplicate_field("abortCode"));
                            }
                            abort_code__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Location => {
                            if location__.is_some() {
                                return Err(serde::de::Error::duplicate_field("location"));
                            }
                            location__ = map_.next_value()?;
                        }
                        GeneratedField::CleverError => {
                            if clever_error__.is_some() {
                                return Err(serde::de::Error::duplicate_field("cleverError"));
                            }
                            clever_error__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MoveAbort {
                    abort_code: abort_code__,
                    location: location__,
                    clever_error: clever_error__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.MoveAbort", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for MoveCall {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.package.is_some() {
            len += 1;
        }
        if self.module.is_some() {
            len += 1;
        }
        if self.function.is_some() {
            len += 1;
        }
        if !self.type_arguments.is_empty() {
            len += 1;
        }
        if !self.arguments.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.MoveCall", len)?;
        if let Some(v) = self.package.as_ref() {
            struct_ser.serialize_field("package", v)?;
        }
        if let Some(v) = self.module.as_ref() {
            struct_ser.serialize_field("module", v)?;
        }
        if let Some(v) = self.function.as_ref() {
            struct_ser.serialize_field("function", v)?;
        }
        if !self.type_arguments.is_empty() {
            struct_ser.serialize_field("typeArguments", &self.type_arguments)?;
        }
        if !self.arguments.is_empty() {
            struct_ser.serialize_field("arguments", &self.arguments)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MoveCall {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "package",
            "module",
            "function",
            "type_arguments",
            "typeArguments",
            "arguments",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Package,
            Module,
            Function,
            TypeArguments,
            Arguments,
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
                            "package" => Ok(GeneratedField::Package),
                            "module" => Ok(GeneratedField::Module),
                            "function" => Ok(GeneratedField::Function),
                            "typeArguments" | "type_arguments" => Ok(GeneratedField::TypeArguments),
                            "arguments" => Ok(GeneratedField::Arguments),
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
            type Value = MoveCall;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.MoveCall")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MoveCall, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut package__ = None;
                let mut module__ = None;
                let mut function__ = None;
                let mut type_arguments__ = None;
                let mut arguments__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Package => {
                            if package__.is_some() {
                                return Err(serde::de::Error::duplicate_field("package"));
                            }
                            package__ = map_.next_value()?;
                        }
                        GeneratedField::Module => {
                            if module__.is_some() {
                                return Err(serde::de::Error::duplicate_field("module"));
                            }
                            module__ = map_.next_value()?;
                        }
                        GeneratedField::Function => {
                            if function__.is_some() {
                                return Err(serde::de::Error::duplicate_field("function"));
                            }
                            function__ = map_.next_value()?;
                        }
                        GeneratedField::TypeArguments => {
                            if type_arguments__.is_some() {
                                return Err(serde::de::Error::duplicate_field("typeArguments"));
                            }
                            type_arguments__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Arguments => {
                            if arguments__.is_some() {
                                return Err(serde::de::Error::duplicate_field("arguments"));
                            }
                            arguments__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MoveCall {
                    package: package__,
                    module: module__,
                    function: function__,
                    type_arguments: type_arguments__.unwrap_or_default(),
                    arguments: arguments__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.MoveCall", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for MoveLocation {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.package.is_some() {
            len += 1;
        }
        if self.module.is_some() {
            len += 1;
        }
        if self.function.is_some() {
            len += 1;
        }
        if self.instruction.is_some() {
            len += 1;
        }
        if self.function_name.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.MoveLocation", len)?;
        if let Some(v) = self.package.as_ref() {
            struct_ser.serialize_field("package", v)?;
        }
        if let Some(v) = self.module.as_ref() {
            struct_ser.serialize_field("module", v)?;
        }
        if let Some(v) = self.function.as_ref() {
            struct_ser.serialize_field("function", v)?;
        }
        if let Some(v) = self.instruction.as_ref() {
            struct_ser.serialize_field("instruction", v)?;
        }
        if let Some(v) = self.function_name.as_ref() {
            struct_ser.serialize_field("functionName", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MoveLocation {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "package",
            "module",
            "function",
            "instruction",
            "function_name",
            "functionName",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Package,
            Module,
            Function,
            Instruction,
            FunctionName,
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
                            "package" => Ok(GeneratedField::Package),
                            "module" => Ok(GeneratedField::Module),
                            "function" => Ok(GeneratedField::Function),
                            "instruction" => Ok(GeneratedField::Instruction),
                            "functionName" | "function_name" => Ok(GeneratedField::FunctionName),
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
            type Value = MoveLocation;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.MoveLocation")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MoveLocation, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut package__ = None;
                let mut module__ = None;
                let mut function__ = None;
                let mut instruction__ = None;
                let mut function_name__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Package => {
                            if package__.is_some() {
                                return Err(serde::de::Error::duplicate_field("package"));
                            }
                            package__ = map_.next_value()?;
                        }
                        GeneratedField::Module => {
                            if module__.is_some() {
                                return Err(serde::de::Error::duplicate_field("module"));
                            }
                            module__ = map_.next_value()?;
                        }
                        GeneratedField::Function => {
                            if function__.is_some() {
                                return Err(serde::de::Error::duplicate_field("function"));
                            }
                            function__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Instruction => {
                            if instruction__.is_some() {
                                return Err(serde::de::Error::duplicate_field("instruction"));
                            }
                            instruction__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::FunctionName => {
                            if function_name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("functionName"));
                            }
                            function_name__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MoveLocation {
                    package: package__,
                    module: module__,
                    function: function__,
                    instruction: instruction__,
                    function_name: function_name__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.MoveLocation", FIELDS, GeneratedVisitor)
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
        if !self.legacy_bitmap.is_empty() {
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
        if !self.legacy_bitmap.is_empty() {
            struct_ser.serialize_field("legacyBitmap", &self.legacy_bitmap)?;
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
            "legacy_bitmap",
            "legacyBitmap",
            "committee",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Signatures,
            Bitmap,
            LegacyBitmap,
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
                            "legacyBitmap" | "legacy_bitmap" => Ok(GeneratedField::LegacyBitmap),
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
                let mut legacy_bitmap__ = None;
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
                        GeneratedField::LegacyBitmap => {
                            if legacy_bitmap__.is_some() {
                                return Err(serde::de::Error::duplicate_field("legacyBitmap"));
                            }
                            legacy_bitmap__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
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
                    legacy_bitmap: legacy_bitmap__.unwrap_or_default(),
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
        if self.zklogin.is_some() {
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
        if let Some(v) = self.zklogin.as_ref() {
            struct_ser.serialize_field("zklogin", v)?;
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
            "zklogin",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Scheme,
            PublicKey,
            Zklogin,
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
                            "zklogin" => Ok(GeneratedField::Zklogin),
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
                let mut zklogin__ = None;
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
                        GeneratedField::Zklogin => {
                            if zklogin__.is_some() {
                                return Err(serde::de::Error::duplicate_field("zklogin"));
                            }
                            zklogin__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MultisigMemberPublicKey {
                    scheme: scheme__,
                    public_key: public_key__,
                    zklogin: zklogin__,
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
        if self.zklogin.is_some() {
            len += 1;
        }
        if self.passkey.is_some() {
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
        if let Some(v) = self.zklogin.as_ref() {
            struct_ser.serialize_field("zklogin", v)?;
        }
        if let Some(v) = self.passkey.as_ref() {
            struct_ser.serialize_field("passkey", v)?;
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
            "zklogin",
            "passkey",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Scheme,
            Signature,
            Zklogin,
            Passkey,
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
                            "zklogin" => Ok(GeneratedField::Zklogin),
                            "passkey" => Ok(GeneratedField::Passkey),
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
                let mut zklogin__ = None;
                let mut passkey__ = None;
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
                        GeneratedField::Zklogin => {
                            if zklogin__.is_some() {
                                return Err(serde::de::Error::duplicate_field("zklogin"));
                            }
                            zklogin__ = map_.next_value()?;
                        }
                        GeneratedField::Passkey => {
                            if passkey__.is_some() {
                                return Err(serde::de::Error::duplicate_field("passkey"));
                            }
                            passkey__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(MultisigMemberSignature {
                    scheme: scheme__,
                    signature: signature__,
                    zklogin: zklogin__,
                    passkey: passkey__,
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
        if self.has_public_transfer.is_some() {
            len += 1;
        }
        if self.contents.is_some() {
            len += 1;
        }
        if self.package.is_some() {
            len += 1;
        }
        if self.previous_transaction.is_some() {
            len += 1;
        }
        if self.storage_rebate.is_some() {
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
        if let Some(v) = self.has_public_transfer.as_ref() {
            struct_ser.serialize_field("hasPublicTransfer", v)?;
        }
        if let Some(v) = self.contents.as_ref() {
            struct_ser.serialize_field("contents", v)?;
        }
        if let Some(v) = self.package.as_ref() {
            struct_ser.serialize_field("package", v)?;
        }
        if let Some(v) = self.previous_transaction.as_ref() {
            struct_ser.serialize_field("previousTransaction", v)?;
        }
        if let Some(v) = self.storage_rebate.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("storageRebate", ToString::to_string(&v).as_str())?;
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
            "has_public_transfer",
            "hasPublicTransfer",
            "contents",
            "package",
            "previous_transaction",
            "previousTransaction",
            "storage_rebate",
            "storageRebate",
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
            HasPublicTransfer,
            Contents,
            Package,
            PreviousTransaction,
            StorageRebate,
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
                            "hasPublicTransfer" | "has_public_transfer" => Ok(GeneratedField::HasPublicTransfer),
                            "contents" => Ok(GeneratedField::Contents),
                            "package" => Ok(GeneratedField::Package),
                            "previousTransaction" | "previous_transaction" => Ok(GeneratedField::PreviousTransaction),
                            "storageRebate" | "storage_rebate" => Ok(GeneratedField::StorageRebate),
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
                let mut has_public_transfer__ = None;
                let mut contents__ = None;
                let mut package__ = None;
                let mut previous_transaction__ = None;
                let mut storage_rebate__ = None;
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
                        GeneratedField::HasPublicTransfer => {
                            if has_public_transfer__.is_some() {
                                return Err(serde::de::Error::duplicate_field("hasPublicTransfer"));
                            }
                            has_public_transfer__ = map_.next_value()?;
                        }
                        GeneratedField::Contents => {
                            if contents__.is_some() {
                                return Err(serde::de::Error::duplicate_field("contents"));
                            }
                            contents__ = map_.next_value()?;
                        }
                        GeneratedField::Package => {
                            if package__.is_some() {
                                return Err(serde::de::Error::duplicate_field("package"));
                            }
                            package__ = map_.next_value()?;
                        }
                        GeneratedField::PreviousTransaction => {
                            if previous_transaction__.is_some() {
                                return Err(serde::de::Error::duplicate_field("previousTransaction"));
                            }
                            previous_transaction__ = map_.next_value()?;
                        }
                        GeneratedField::StorageRebate => {
                            if storage_rebate__.is_some() {
                                return Err(serde::de::Error::duplicate_field("storageRebate"));
                            }
                            storage_rebate__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
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
                    has_public_transfer: has_public_transfer__,
                    contents: contents__,
                    package: package__,
                    previous_transaction: previous_transaction__,
                    storage_rebate: storage_rebate__,
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
impl serde::Serialize for OpenSignature {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.reference.is_some() {
            len += 1;
        }
        if self.body.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.OpenSignature", len)?;
        if let Some(v) = self.reference.as_ref() {
            let v = open_signature::Reference::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("reference", &v)?;
        }
        if let Some(v) = self.body.as_ref() {
            struct_ser.serialize_field("body", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for OpenSignature {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "reference",
            "body",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Reference,
            Body,
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
                            "reference" => Ok(GeneratedField::Reference),
                            "body" => Ok(GeneratedField::Body),
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
            type Value = OpenSignature;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.OpenSignature")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<OpenSignature, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut reference__ = None;
                let mut body__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Reference => {
                            if reference__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reference"));
                            }
                            reference__ = map_.next_value::<::std::option::Option<open_signature::Reference>>()?.map(|x| x as i32);
                        }
                        GeneratedField::Body => {
                            if body__.is_some() {
                                return Err(serde::de::Error::duplicate_field("body"));
                            }
                            body__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(OpenSignature {
                    reference: reference__,
                    body: body__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.OpenSignature", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for open_signature::Reference {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "REFERENCE_UNKNOWN",
            Self::Immutable => "IMMUTABLE",
            Self::Mutable => "MUTABLE",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for open_signature::Reference {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "REFERENCE_UNKNOWN",
            "IMMUTABLE",
            "MUTABLE",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = open_signature::Reference;

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
                    "REFERENCE_UNKNOWN" => Ok(open_signature::Reference::Unknown),
                    "IMMUTABLE" => Ok(open_signature::Reference::Immutable),
                    "MUTABLE" => Ok(open_signature::Reference::Mutable),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for OpenSignatureBody {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.r#type.is_some() {
            len += 1;
        }
        if self.type_name.is_some() {
            len += 1;
        }
        if !self.type_parameter_instantiation.is_empty() {
            len += 1;
        }
        if self.type_parameter.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.OpenSignatureBody", len)?;
        if let Some(v) = self.r#type.as_ref() {
            let v = open_signature_body::Type::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("type", &v)?;
        }
        if let Some(v) = self.type_name.as_ref() {
            struct_ser.serialize_field("typeName", v)?;
        }
        if !self.type_parameter_instantiation.is_empty() {
            struct_ser.serialize_field("typeParameterInstantiation", &self.type_parameter_instantiation)?;
        }
        if let Some(v) = self.type_parameter.as_ref() {
            struct_ser.serialize_field("typeParameter", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for OpenSignatureBody {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "type",
            "type_name",
            "typeName",
            "type_parameter_instantiation",
            "typeParameterInstantiation",
            "type_parameter",
            "typeParameter",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Type,
            TypeName,
            TypeParameterInstantiation,
            TypeParameter,
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
                            "type" => Ok(GeneratedField::Type),
                            "typeName" | "type_name" => Ok(GeneratedField::TypeName),
                            "typeParameterInstantiation" | "type_parameter_instantiation" => Ok(GeneratedField::TypeParameterInstantiation),
                            "typeParameter" | "type_parameter" => Ok(GeneratedField::TypeParameter),
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
            type Value = OpenSignatureBody;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.OpenSignatureBody")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<OpenSignatureBody, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut r#type__ = None;
                let mut type_name__ = None;
                let mut type_parameter_instantiation__ = None;
                let mut type_parameter__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Type => {
                            if r#type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("type"));
                            }
                            r#type__ = map_.next_value::<::std::option::Option<open_signature_body::Type>>()?.map(|x| x as i32);
                        }
                        GeneratedField::TypeName => {
                            if type_name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("typeName"));
                            }
                            type_name__ = map_.next_value()?;
                        }
                        GeneratedField::TypeParameterInstantiation => {
                            if type_parameter_instantiation__.is_some() {
                                return Err(serde::de::Error::duplicate_field("typeParameterInstantiation"));
                            }
                            type_parameter_instantiation__ = Some(map_.next_value()?);
                        }
                        GeneratedField::TypeParameter => {
                            if type_parameter__.is_some() {
                                return Err(serde::de::Error::duplicate_field("typeParameter"));
                            }
                            type_parameter__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(OpenSignatureBody {
                    r#type: r#type__,
                    type_name: type_name__,
                    type_parameter_instantiation: type_parameter_instantiation__.unwrap_or_default(),
                    type_parameter: type_parameter__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.OpenSignatureBody", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for open_signature_body::Type {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "TYPE_UNKNOWN",
            Self::Address => "ADDRESS",
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::U64 => "U64",
            Self::U128 => "U128",
            Self::U256 => "U256",
            Self::Vector => "VECTOR",
            Self::Datatype => "DATATYPE",
            Self::Parameter => "TYPE_PARAMETER",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for open_signature_body::Type {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "TYPE_UNKNOWN",
            "ADDRESS",
            "BOOL",
            "U8",
            "U16",
            "U32",
            "U64",
            "U128",
            "U256",
            "VECTOR",
            "DATATYPE",
            "TYPE_PARAMETER",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = open_signature_body::Type;

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
                    "TYPE_UNKNOWN" => Ok(open_signature_body::Type::Unknown),
                    "ADDRESS" => Ok(open_signature_body::Type::Address),
                    "BOOL" => Ok(open_signature_body::Type::Bool),
                    "U8" => Ok(open_signature_body::Type::U8),
                    "U16" => Ok(open_signature_body::Type::U16),
                    "U32" => Ok(open_signature_body::Type::U32),
                    "U64" => Ok(open_signature_body::Type::U64),
                    "U128" => Ok(open_signature_body::Type::U128),
                    "U256" => Ok(open_signature_body::Type::U256),
                    "VECTOR" => Ok(open_signature_body::Type::Vector),
                    "DATATYPE" => Ok(open_signature_body::Type::Datatype),
                    "TYPE_PARAMETER" => Ok(open_signature_body::Type::Parameter),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
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
            Self::Object => "OBJECT",
            Self::Shared => "SHARED",
            Self::Immutable => "IMMUTABLE",
            Self::ConsensusAddress => "CONSENSUS_ADDRESS",
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
            "OBJECT",
            "SHARED",
            "IMMUTABLE",
            "CONSENSUS_ADDRESS",
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
                    "OBJECT" => Ok(owner::OwnerKind::Object),
                    "SHARED" => Ok(owner::OwnerKind::Shared),
                    "IMMUTABLE" => Ok(owner::OwnerKind::Immutable),
                    "CONSENSUS_ADDRESS" => Ok(owner::OwnerKind::ConsensusAddress),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for Package {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.storage_id.is_some() {
            len += 1;
        }
        if self.original_id.is_some() {
            len += 1;
        }
        if self.version.is_some() {
            len += 1;
        }
        if !self.modules.is_empty() {
            len += 1;
        }
        if !self.type_origins.is_empty() {
            len += 1;
        }
        if !self.linkage.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Package", len)?;
        if let Some(v) = self.storage_id.as_ref() {
            struct_ser.serialize_field("storageId", v)?;
        }
        if let Some(v) = self.original_id.as_ref() {
            struct_ser.serialize_field("originalId", v)?;
        }
        if let Some(v) = self.version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("version", ToString::to_string(&v).as_str())?;
        }
        if !self.modules.is_empty() {
            struct_ser.serialize_field("modules", &self.modules)?;
        }
        if !self.type_origins.is_empty() {
            struct_ser.serialize_field("typeOrigins", &self.type_origins)?;
        }
        if !self.linkage.is_empty() {
            struct_ser.serialize_field("linkage", &self.linkage)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Package {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "storage_id",
            "storageId",
            "original_id",
            "originalId",
            "version",
            "modules",
            "type_origins",
            "typeOrigins",
            "linkage",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            StorageId,
            OriginalId,
            Version,
            Modules,
            TypeOrigins,
            Linkage,
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
                            "storageId" | "storage_id" => Ok(GeneratedField::StorageId),
                            "originalId" | "original_id" => Ok(GeneratedField::OriginalId),
                            "version" => Ok(GeneratedField::Version),
                            "modules" => Ok(GeneratedField::Modules),
                            "typeOrigins" | "type_origins" => Ok(GeneratedField::TypeOrigins),
                            "linkage" => Ok(GeneratedField::Linkage),
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
            type Value = Package;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Package")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Package, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut storage_id__ = None;
                let mut original_id__ = None;
                let mut version__ = None;
                let mut modules__ = None;
                let mut type_origins__ = None;
                let mut linkage__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::StorageId => {
                            if storage_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("storageId"));
                            }
                            storage_id__ = map_.next_value()?;
                        }
                        GeneratedField::OriginalId => {
                            if original_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("originalId"));
                            }
                            original_id__ = map_.next_value()?;
                        }
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Modules => {
                            if modules__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modules"));
                            }
                            modules__ = Some(map_.next_value()?);
                        }
                        GeneratedField::TypeOrigins => {
                            if type_origins__.is_some() {
                                return Err(serde::de::Error::duplicate_field("typeOrigins"));
                            }
                            type_origins__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Linkage => {
                            if linkage__.is_some() {
                                return Err(serde::de::Error::duplicate_field("linkage"));
                            }
                            linkage__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Package {
                    storage_id: storage_id__,
                    original_id: original_id__,
                    version: version__,
                    modules: modules__.unwrap_or_default(),
                    type_origins: type_origins__.unwrap_or_default(),
                    linkage: linkage__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Package", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for PackageUpgradeError {
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
        if self.package_id.is_some() {
            len += 1;
        }
        if self.digest.is_some() {
            len += 1;
        }
        if self.policy.is_some() {
            len += 1;
        }
        if self.ticket_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.PackageUpgradeError", len)?;
        if let Some(v) = self.kind.as_ref() {
            let v = package_upgrade_error::PackageUpgradeErrorKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.package_id.as_ref() {
            struct_ser.serialize_field("packageId", v)?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.policy.as_ref() {
            struct_ser.serialize_field("policy", v)?;
        }
        if let Some(v) = self.ticket_id.as_ref() {
            struct_ser.serialize_field("ticketId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for PackageUpgradeError {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "package_id",
            "packageId",
            "digest",
            "policy",
            "ticket_id",
            "ticketId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            PackageId,
            Digest,
            Policy,
            TicketId,
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
                            "packageId" | "package_id" => Ok(GeneratedField::PackageId),
                            "digest" => Ok(GeneratedField::Digest),
                            "policy" => Ok(GeneratedField::Policy),
                            "ticketId" | "ticket_id" => Ok(GeneratedField::TicketId),
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
            type Value = PackageUpgradeError;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.PackageUpgradeError")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<PackageUpgradeError, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut package_id__ = None;
                let mut digest__ = None;
                let mut policy__ = None;
                let mut ticket_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<package_upgrade_error::PackageUpgradeErrorKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::PackageId => {
                            if package_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("packageId"));
                            }
                            package_id__ = map_.next_value()?;
                        }
                        GeneratedField::Digest => {
                            if digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("digest"));
                            }
                            digest__ = map_.next_value()?;
                        }
                        GeneratedField::Policy => {
                            if policy__.is_some() {
                                return Err(serde::de::Error::duplicate_field("policy"));
                            }
                            policy__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::TicketId => {
                            if ticket_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("ticketId"));
                            }
                            ticket_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(PackageUpgradeError {
                    kind: kind__,
                    package_id: package_id__,
                    digest: digest__,
                    policy: policy__,
                    ticket_id: ticket_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.PackageUpgradeError", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for package_upgrade_error::PackageUpgradeErrorKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "PACKAGE_UPGRADE_ERROR_KIND_UNKNOWN",
            Self::UnableToFetchPackage => "UNABLE_TO_FETCH_PACKAGE",
            Self::NotAPackage => "NOT_A_PACKAGE",
            Self::IncompatibleUpgrade => "INCOMPATIBLE_UPGRADE",
            Self::DigestDoesNotMatch => "DIGEST_DOES_NOT_MATCH",
            Self::UnknownUpgradePolicy => "UNKNOWN_UPGRADE_POLICY",
            Self::PackageIdDoesNotMatch => "PACKAGE_ID_DOES_NOT_MATCH",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for package_upgrade_error::PackageUpgradeErrorKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "PACKAGE_UPGRADE_ERROR_KIND_UNKNOWN",
            "UNABLE_TO_FETCH_PACKAGE",
            "NOT_A_PACKAGE",
            "INCOMPATIBLE_UPGRADE",
            "DIGEST_DOES_NOT_MATCH",
            "UNKNOWN_UPGRADE_POLICY",
            "PACKAGE_ID_DOES_NOT_MATCH",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = package_upgrade_error::PackageUpgradeErrorKind;

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
                    "PACKAGE_UPGRADE_ERROR_KIND_UNKNOWN" => Ok(package_upgrade_error::PackageUpgradeErrorKind::Unknown),
                    "UNABLE_TO_FETCH_PACKAGE" => Ok(package_upgrade_error::PackageUpgradeErrorKind::UnableToFetchPackage),
                    "NOT_A_PACKAGE" => Ok(package_upgrade_error::PackageUpgradeErrorKind::NotAPackage),
                    "INCOMPATIBLE_UPGRADE" => Ok(package_upgrade_error::PackageUpgradeErrorKind::IncompatibleUpgrade),
                    "DIGEST_DOES_NOT_MATCH" => Ok(package_upgrade_error::PackageUpgradeErrorKind::DigestDoesNotMatch),
                    "UNKNOWN_UPGRADE_POLICY" => Ok(package_upgrade_error::PackageUpgradeErrorKind::UnknownUpgradePolicy),
                    "PACKAGE_ID_DOES_NOT_MATCH" => Ok(package_upgrade_error::PackageUpgradeErrorKind::PackageIdDoesNotMatch),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for PasskeyAuthenticator {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.authenticator_data.is_some() {
            len += 1;
        }
        if self.client_data_json.is_some() {
            len += 1;
        }
        if self.signature.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.PasskeyAuthenticator", len)?;
        if let Some(v) = self.authenticator_data.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("authenticatorData", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.client_data_json.as_ref() {
            struct_ser.serialize_field("clientDataJson", v)?;
        }
        if let Some(v) = self.signature.as_ref() {
            struct_ser.serialize_field("signature", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for PasskeyAuthenticator {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "authenticator_data",
            "authenticatorData",
            "client_data_json",
            "clientDataJson",
            "signature",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            AuthenticatorData,
            ClientDataJson,
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
                            "authenticatorData" | "authenticator_data" => Ok(GeneratedField::AuthenticatorData),
                            "clientDataJson" | "client_data_json" => Ok(GeneratedField::ClientDataJson),
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
            type Value = PasskeyAuthenticator;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.PasskeyAuthenticator")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<PasskeyAuthenticator, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut authenticator_data__ = None;
                let mut client_data_json__ = None;
                let mut signature__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::AuthenticatorData => {
                            if authenticator_data__.is_some() {
                                return Err(serde::de::Error::duplicate_field("authenticatorData"));
                            }
                            authenticator_data__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::ClientDataJson => {
                            if client_data_json__.is_some() {
                                return Err(serde::de::Error::duplicate_field("clientDataJson"));
                            }
                            client_data_json__ = map_.next_value()?;
                        }
                        GeneratedField::Signature => {
                            if signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signature"));
                            }
                            signature__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(PasskeyAuthenticator {
                    authenticator_data: authenticator_data__,
                    client_data_json: client_data_json__,
                    signature: signature__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.PasskeyAuthenticator", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ProgrammableTransaction {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.inputs.is_empty() {
            len += 1;
        }
        if !self.commands.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ProgrammableTransaction", len)?;
        if !self.inputs.is_empty() {
            struct_ser.serialize_field("inputs", &self.inputs)?;
        }
        if !self.commands.is_empty() {
            struct_ser.serialize_field("commands", &self.commands)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ProgrammableTransaction {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "inputs",
            "commands",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Inputs,
            Commands,
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
                            "inputs" => Ok(GeneratedField::Inputs),
                            "commands" => Ok(GeneratedField::Commands),
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
            type Value = ProgrammableTransaction;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ProgrammableTransaction")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ProgrammableTransaction, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut inputs__ = None;
                let mut commands__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Inputs => {
                            if inputs__.is_some() {
                                return Err(serde::de::Error::duplicate_field("inputs"));
                            }
                            inputs__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Commands => {
                            if commands__.is_some() {
                                return Err(serde::de::Error::duplicate_field("commands"));
                            }
                            commands__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ProgrammableTransaction {
                    inputs: inputs__.unwrap_or_default(),
                    commands: commands__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ProgrammableTransaction", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Publish {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.modules.is_empty() {
            len += 1;
        }
        if !self.dependencies.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Publish", len)?;
        if !self.modules.is_empty() {
            struct_ser.serialize_field("modules", &self.modules.iter().map(crate::utils::_serde::base64::encode).collect::<Vec<_>>())?;
        }
        if !self.dependencies.is_empty() {
            struct_ser.serialize_field("dependencies", &self.dependencies)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Publish {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "modules",
            "dependencies",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Modules,
            Dependencies,
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
                            "modules" => Ok(GeneratedField::Modules),
                            "dependencies" => Ok(GeneratedField::Dependencies),
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
            type Value = Publish;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Publish")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Publish, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut modules__ = None;
                let mut dependencies__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Modules => {
                            if modules__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modules"));
                            }
                            modules__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::BytesDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::Dependencies => {
                            if dependencies__.is_some() {
                                return Err(serde::de::Error::duplicate_field("dependencies"));
                            }
                            dependencies__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Publish {
                    modules: modules__.unwrap_or_default(),
                    dependencies: dependencies__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Publish", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for RandomnessStateUpdate {
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
        if self.randomness_round.is_some() {
            len += 1;
        }
        if self.random_bytes.is_some() {
            len += 1;
        }
        if self.randomness_object_initial_shared_version.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.RandomnessStateUpdate", len)?;
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.randomness_round.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("randomnessRound", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.random_bytes.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("randomBytes", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.randomness_object_initial_shared_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("randomnessObjectInitialSharedVersion", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for RandomnessStateUpdate {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epoch",
            "randomness_round",
            "randomnessRound",
            "random_bytes",
            "randomBytes",
            "randomness_object_initial_shared_version",
            "randomnessObjectInitialSharedVersion",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epoch,
            RandomnessRound,
            RandomBytes,
            RandomnessObjectInitialSharedVersion,
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
                            "randomnessRound" | "randomness_round" => Ok(GeneratedField::RandomnessRound),
                            "randomBytes" | "random_bytes" => Ok(GeneratedField::RandomBytes),
                            "randomnessObjectInitialSharedVersion" | "randomness_object_initial_shared_version" => Ok(GeneratedField::RandomnessObjectInitialSharedVersion),
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
            type Value = RandomnessStateUpdate;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.RandomnessStateUpdate")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<RandomnessStateUpdate, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epoch__ = None;
                let mut randomness_round__ = None;
                let mut random_bytes__ = None;
                let mut randomness_object_initial_shared_version__ = None;
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
                        GeneratedField::RandomnessRound => {
                            if randomness_round__.is_some() {
                                return Err(serde::de::Error::duplicate_field("randomnessRound"));
                            }
                            randomness_round__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::RandomBytes => {
                            if random_bytes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("randomBytes"));
                            }
                            random_bytes__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::RandomnessObjectInitialSharedVersion => {
                            if randomness_object_initial_shared_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("randomnessObjectInitialSharedVersion"));
                            }
                            randomness_object_initial_shared_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(RandomnessStateUpdate {
                    epoch: epoch__,
                    randomness_round: randomness_round__,
                    random_bytes: random_bytes__,
                    randomness_object_initial_shared_version: randomness_object_initial_shared_version__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.RandomnessStateUpdate", FIELDS, GeneratedVisitor)
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
            Self::Secp256k1 => "SECP256K1",
            Self::Secp256r1 => "SECP256R1",
            Self::Multisig => "MULTISIG",
            Self::Bls12381 => "BLS12381",
            Self::Zklogin => "ZKLOGIN",
            Self::Passkey => "PASSKEY",
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
            "SECP256K1",
            "SECP256R1",
            "MULTISIG",
            "BLS12381",
            "ZKLOGIN",
            "PASSKEY",
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
                    "SECP256K1" => Ok(SignatureScheme::Secp256k1),
                    "SECP256R1" => Ok(SignatureScheme::Secp256r1),
                    "MULTISIG" => Ok(SignatureScheme::Multisig),
                    "BLS12381" => Ok(SignatureScheme::Bls12381),
                    "ZKLOGIN" => Ok(SignatureScheme::Zklogin),
                    "PASSKEY" => Ok(SignatureScheme::Passkey),
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
impl serde::Serialize for SizeError {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.size.is_some() {
            len += 1;
        }
        if self.max_size.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SizeError", len)?;
        if let Some(v) = self.size.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("size", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.max_size.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("maxSize", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SizeError {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "size",
            "max_size",
            "maxSize",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Size,
            MaxSize,
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
                            "size" => Ok(GeneratedField::Size),
                            "maxSize" | "max_size" => Ok(GeneratedField::MaxSize),
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
            type Value = SizeError;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SizeError")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SizeError, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut size__ = None;
                let mut max_size__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Size => {
                            if size__.is_some() {
                                return Err(serde::de::Error::duplicate_field("size"));
                            }
                            size__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::MaxSize => {
                            if max_size__.is_some() {
                                return Err(serde::de::Error::duplicate_field("maxSize"));
                            }
                            max_size__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SizeError {
                    size: size__,
                    max_size: max_size__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SizeError", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SplitCoins {
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
        if !self.amounts.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SplitCoins", len)?;
        if let Some(v) = self.coin.as_ref() {
            struct_ser.serialize_field("coin", v)?;
        }
        if !self.amounts.is_empty() {
            struct_ser.serialize_field("amounts", &self.amounts)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SplitCoins {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "coin",
            "amounts",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Coin,
            Amounts,
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
                            "amounts" => Ok(GeneratedField::Amounts),
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
            type Value = SplitCoins;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SplitCoins")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SplitCoins, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut coin__ = None;
                let mut amounts__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Coin => {
                            if coin__.is_some() {
                                return Err(serde::de::Error::duplicate_field("coin"));
                            }
                            coin__ = map_.next_value()?;
                        }
                        GeneratedField::Amounts => {
                            if amounts__.is_some() {
                                return Err(serde::de::Error::duplicate_field("amounts"));
                            }
                            amounts__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SplitCoins {
                    coin: coin__,
                    amounts: amounts__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SplitCoins", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SystemPackage {
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
        if !self.modules.is_empty() {
            len += 1;
        }
        if !self.dependencies.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.SystemPackage", len)?;
        if let Some(v) = self.version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("version", ToString::to_string(&v).as_str())?;
        }
        if !self.modules.is_empty() {
            struct_ser.serialize_field("modules", &self.modules.iter().map(crate::utils::_serde::base64::encode).collect::<Vec<_>>())?;
        }
        if !self.dependencies.is_empty() {
            struct_ser.serialize_field("dependencies", &self.dependencies)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SystemPackage {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "version",
            "modules",
            "dependencies",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Version,
            Modules,
            Dependencies,
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
                            "version" => Ok(GeneratedField::Version),
                            "modules" => Ok(GeneratedField::Modules),
                            "dependencies" => Ok(GeneratedField::Dependencies),
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
            type Value = SystemPackage;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.SystemPackage")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SystemPackage, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut version__ = None;
                let mut modules__ = None;
                let mut dependencies__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Modules => {
                            if modules__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modules"));
                            }
                            modules__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::BytesDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::Dependencies => {
                            if dependencies__.is_some() {
                                return Err(serde::de::Error::duplicate_field("dependencies"));
                            }
                            dependencies__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(SystemPackage {
                    version: version__,
                    modules: modules__.unwrap_or_default(),
                    dependencies: dependencies__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.SystemPackage", FIELDS, GeneratedVisitor)
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
        if self.version.is_some() {
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
        if self.expiration.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Transaction", len)?;
        if let Some(v) = self.bcs.as_ref() {
            struct_ser.serialize_field("bcs", v)?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.version.as_ref() {
            struct_ser.serialize_field("version", v)?;
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
        if let Some(v) = self.expiration.as_ref() {
            struct_ser.serialize_field("expiration", v)?;
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
            "version",
            "kind",
            "sender",
            "gas_payment",
            "gasPayment",
            "expiration",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Bcs,
            Digest,
            Version,
            Kind,
            Sender,
            GasPayment,
            Expiration,
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
                            "version" => Ok(GeneratedField::Version),
                            "kind" => Ok(GeneratedField::Kind),
                            "sender" => Ok(GeneratedField::Sender),
                            "gasPayment" | "gas_payment" => Ok(GeneratedField::GasPayment),
                            "expiration" => Ok(GeneratedField::Expiration),
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
                let mut version__ = None;
                let mut kind__ = None;
                let mut sender__ = None;
                let mut gas_payment__ = None;
                let mut expiration__ = None;
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
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
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
                        GeneratedField::Expiration => {
                            if expiration__.is_some() {
                                return Err(serde::de::Error::duplicate_field("expiration"));
                            }
                            expiration__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Transaction {
                    bcs: bcs__,
                    digest: digest__,
                    version: version__,
                    kind: kind__,
                    sender: sender__,
                    gas_payment: gas_payment__,
                    expiration: expiration__,
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
        if self.version.is_some() {
            len += 1;
        }
        if self.status.is_some() {
            len += 1;
        }
        if self.epoch.is_some() {
            len += 1;
        }
        if self.gas_used.is_some() {
            len += 1;
        }
        if self.transaction_digest.is_some() {
            len += 1;
        }
        if self.gas_object.is_some() {
            len += 1;
        }
        if self.events_digest.is_some() {
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
        if !self.unchanged_consensus_objects.is_empty() {
            len += 1;
        }
        if self.auxiliary_data_digest.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TransactionEffects", len)?;
        if let Some(v) = self.bcs.as_ref() {
            struct_ser.serialize_field("bcs", v)?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if let Some(v) = self.version.as_ref() {
            struct_ser.serialize_field("version", v)?;
        }
        if let Some(v) = self.status.as_ref() {
            struct_ser.serialize_field("status", v)?;
        }
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.gas_used.as_ref() {
            struct_ser.serialize_field("gasUsed", v)?;
        }
        if let Some(v) = self.transaction_digest.as_ref() {
            struct_ser.serialize_field("transactionDigest", v)?;
        }
        if let Some(v) = self.gas_object.as_ref() {
            struct_ser.serialize_field("gasObject", v)?;
        }
        if let Some(v) = self.events_digest.as_ref() {
            struct_ser.serialize_field("eventsDigest", v)?;
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
        if !self.unchanged_consensus_objects.is_empty() {
            struct_ser.serialize_field("unchangedConsensusObjects", &self.unchanged_consensus_objects)?;
        }
        if let Some(v) = self.auxiliary_data_digest.as_ref() {
            struct_ser.serialize_field("auxiliaryDataDigest", v)?;
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
            "version",
            "status",
            "epoch",
            "gas_used",
            "gasUsed",
            "transaction_digest",
            "transactionDigest",
            "gas_object",
            "gasObject",
            "events_digest",
            "eventsDigest",
            "dependencies",
            "lamport_version",
            "lamportVersion",
            "changed_objects",
            "changedObjects",
            "unchanged_consensus_objects",
            "unchangedConsensusObjects",
            "auxiliary_data_digest",
            "auxiliaryDataDigest",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Bcs,
            Digest,
            Version,
            Status,
            Epoch,
            GasUsed,
            TransactionDigest,
            GasObject,
            EventsDigest,
            Dependencies,
            LamportVersion,
            ChangedObjects,
            UnchangedConsensusObjects,
            AuxiliaryDataDigest,
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
                            "version" => Ok(GeneratedField::Version),
                            "status" => Ok(GeneratedField::Status),
                            "epoch" => Ok(GeneratedField::Epoch),
                            "gasUsed" | "gas_used" => Ok(GeneratedField::GasUsed),
                            "transactionDigest" | "transaction_digest" => Ok(GeneratedField::TransactionDigest),
                            "gasObject" | "gas_object" => Ok(GeneratedField::GasObject),
                            "eventsDigest" | "events_digest" => Ok(GeneratedField::EventsDigest),
                            "dependencies" => Ok(GeneratedField::Dependencies),
                            "lamportVersion" | "lamport_version" => Ok(GeneratedField::LamportVersion),
                            "changedObjects" | "changed_objects" => Ok(GeneratedField::ChangedObjects),
                            "unchangedConsensusObjects" | "unchanged_consensus_objects" => Ok(GeneratedField::UnchangedConsensusObjects),
                            "auxiliaryDataDigest" | "auxiliary_data_digest" => Ok(GeneratedField::AuxiliaryDataDigest),
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
                let mut version__ = None;
                let mut status__ = None;
                let mut epoch__ = None;
                let mut gas_used__ = None;
                let mut transaction_digest__ = None;
                let mut gas_object__ = None;
                let mut events_digest__ = None;
                let mut dependencies__ = None;
                let mut lamport_version__ = None;
                let mut changed_objects__ = None;
                let mut unchanged_consensus_objects__ = None;
                let mut auxiliary_data_digest__ = None;
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
                        GeneratedField::Version => {
                            if version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("version"));
                            }
                            version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
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
                        GeneratedField::GasUsed => {
                            if gas_used__.is_some() {
                                return Err(serde::de::Error::duplicate_field("gasUsed"));
                            }
                            gas_used__ = map_.next_value()?;
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
                        GeneratedField::EventsDigest => {
                            if events_digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("eventsDigest"));
                            }
                            events_digest__ = map_.next_value()?;
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
                        GeneratedField::UnchangedConsensusObjects => {
                            if unchanged_consensus_objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("unchangedConsensusObjects"));
                            }
                            unchanged_consensus_objects__ = Some(map_.next_value()?);
                        }
                        GeneratedField::AuxiliaryDataDigest => {
                            if auxiliary_data_digest__.is_some() {
                                return Err(serde::de::Error::duplicate_field("auxiliaryDataDigest"));
                            }
                            auxiliary_data_digest__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TransactionEffects {
                    bcs: bcs__,
                    digest: digest__,
                    version: version__,
                    status: status__,
                    epoch: epoch__,
                    gas_used: gas_used__,
                    transaction_digest: transaction_digest__,
                    gas_object: gas_object__,
                    events_digest: events_digest__,
                    dependencies: dependencies__.unwrap_or_default(),
                    lamport_version: lamport_version__,
                    changed_objects: changed_objects__.unwrap_or_default(),
                    unchanged_consensus_objects: unchanged_consensus_objects__.unwrap_or_default(),
                    auxiliary_data_digest: auxiliary_data_digest__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TransactionEffects", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TransactionEvents {
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
        if !self.events.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TransactionEvents", len)?;
        if let Some(v) = self.bcs.as_ref() {
            struct_ser.serialize_field("bcs", v)?;
        }
        if let Some(v) = self.digest.as_ref() {
            struct_ser.serialize_field("digest", v)?;
        }
        if !self.events.is_empty() {
            struct_ser.serialize_field("events", &self.events)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TransactionEvents {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "bcs",
            "digest",
            "events",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Bcs,
            Digest,
            Events,
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
                            "events" => Ok(GeneratedField::Events),
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
            type Value = TransactionEvents;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TransactionEvents")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TransactionEvents, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut bcs__ = None;
                let mut digest__ = None;
                let mut events__ = None;
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
                        GeneratedField::Events => {
                            if events__.is_some() {
                                return Err(serde::de::Error::duplicate_field("events"));
                            }
                            events__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TransactionEvents {
                    bcs: bcs__,
                    digest: digest__,
                    events: events__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TransactionEvents", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TransactionExpiration {
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
        if self.epoch.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TransactionExpiration", len)?;
        if let Some(v) = self.kind.as_ref() {
            let v = transaction_expiration::TransactionExpirationKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("epoch", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TransactionExpiration {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "epoch",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
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
                            "kind" => Ok(GeneratedField::Kind),
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
            type Value = TransactionExpiration;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TransactionExpiration")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TransactionExpiration, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut epoch__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<transaction_expiration::TransactionExpirationKind>>()?.map(|x| x as i32);
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
                Ok(TransactionExpiration {
                    kind: kind__,
                    epoch: epoch__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TransactionExpiration", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for transaction_expiration::TransactionExpirationKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "TRANSACTION_EXPIRATION_KIND_UNKNOWN",
            Self::None => "NONE",
            Self::Epoch => "EPOCH",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for transaction_expiration::TransactionExpirationKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "TRANSACTION_EXPIRATION_KIND_UNKNOWN",
            "NONE",
            "EPOCH",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = transaction_expiration::TransactionExpirationKind;

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
                    "TRANSACTION_EXPIRATION_KIND_UNKNOWN" => Ok(transaction_expiration::TransactionExpirationKind::Unknown),
                    "NONE" => Ok(transaction_expiration::TransactionExpirationKind::None),
                    "EPOCH" => Ok(transaction_expiration::TransactionExpirationKind::Epoch),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
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
                transaction_finality::Finality::Checkpointed(v) => {
                    #[allow(clippy::needless_borrow)]
                    #[allow(clippy::needless_borrows_for_generic_args)]
                    struct_ser.serialize_field("checkpointed", ToString::to_string(&v).as_str())?;
                }
                transaction_finality::Finality::QuorumExecuted(v) => {
                    struct_ser.serialize_field("quorumExecuted", &crate::utils::_serde::EmptySerializer(v))?;
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
            "checkpointed",
            "quorum_executed",
            "quorumExecuted",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Certified,
            Checkpointed,
            QuorumExecuted,
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
                            "checkpointed" => Ok(GeneratedField::Checkpointed),
                            "quorumExecuted" | "quorum_executed" => Ok(GeneratedField::QuorumExecuted),
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
                        GeneratedField::Checkpointed => {
                            if finality__.is_some() {
                                return Err(serde::de::Error::duplicate_field("checkpointed"));
                            }
                            finality__ = map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| transaction_finality::Finality::Checkpointed(x.0));
                        }
                        GeneratedField::QuorumExecuted => {
                            if finality__.is_some() {
                                return Err(serde::de::Error::duplicate_field("quorumExecuted"));
                            }
                            finality__ = map_.next_value::<::std::option::Option<crate::utils::_serde::EmptyDeserializer>>()?.map(|x| transaction_finality::Finality::QuorumExecuted(x.0));
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
                transaction_kind::Kind::ProgrammableTransaction(v) => {
                    struct_ser.serialize_field("programmableTransaction", v)?;
                }
                transaction_kind::Kind::ProgrammableSystemTransaction(v) => {
                    struct_ser.serialize_field("programmableSystemTransaction", v)?;
                }
                transaction_kind::Kind::ChangeEpoch(v) => {
                    struct_ser.serialize_field("changeEpoch", v)?;
                }
                transaction_kind::Kind::Genesis(v) => {
                    struct_ser.serialize_field("genesis", v)?;
                }
                transaction_kind::Kind::ConsensusCommitPrologueV1(v) => {
                    struct_ser.serialize_field("consensusCommitPrologueV1", v)?;
                }
                transaction_kind::Kind::AuthenticatorStateUpdate(v) => {
                    struct_ser.serialize_field("authenticatorStateUpdate", v)?;
                }
                transaction_kind::Kind::EndOfEpoch(v) => {
                    struct_ser.serialize_field("endOfEpoch", v)?;
                }
                transaction_kind::Kind::RandomnessStateUpdate(v) => {
                    struct_ser.serialize_field("randomnessStateUpdate", v)?;
                }
                transaction_kind::Kind::ConsensusCommitPrologueV2(v) => {
                    struct_ser.serialize_field("consensusCommitPrologueV2", v)?;
                }
                transaction_kind::Kind::ConsensusCommitPrologueV3(v) => {
                    struct_ser.serialize_field("consensusCommitPrologueV3", v)?;
                }
                transaction_kind::Kind::ConsensusCommitPrologueV4(v) => {
                    struct_ser.serialize_field("consensusCommitPrologueV4", v)?;
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
            "programmable_transaction",
            "programmableTransaction",
            "programmable_system_transaction",
            "programmableSystemTransaction",
            "change_epoch",
            "changeEpoch",
            "genesis",
            "consensus_commit_prologue_v1",
            "consensusCommitPrologueV1",
            "authenticator_state_update",
            "authenticatorStateUpdate",
            "end_of_epoch",
            "endOfEpoch",
            "randomness_state_update",
            "randomnessStateUpdate",
            "consensus_commit_prologue_v2",
            "consensusCommitPrologueV2",
            "consensus_commit_prologue_v3",
            "consensusCommitPrologueV3",
            "consensus_commit_prologue_v4",
            "consensusCommitPrologueV4",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ProgrammableTransaction,
            ProgrammableSystemTransaction,
            ChangeEpoch,
            Genesis,
            ConsensusCommitPrologueV1,
            AuthenticatorStateUpdate,
            EndOfEpoch,
            RandomnessStateUpdate,
            ConsensusCommitPrologueV2,
            ConsensusCommitPrologueV3,
            ConsensusCommitPrologueV4,
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
                            "programmableTransaction" | "programmable_transaction" => Ok(GeneratedField::ProgrammableTransaction),
                            "programmableSystemTransaction" | "programmable_system_transaction" => Ok(GeneratedField::ProgrammableSystemTransaction),
                            "changeEpoch" | "change_epoch" => Ok(GeneratedField::ChangeEpoch),
                            "genesis" => Ok(GeneratedField::Genesis),
                            "consensusCommitPrologueV1" | "consensus_commit_prologue_v1" => Ok(GeneratedField::ConsensusCommitPrologueV1),
                            "authenticatorStateUpdate" | "authenticator_state_update" => Ok(GeneratedField::AuthenticatorStateUpdate),
                            "endOfEpoch" | "end_of_epoch" => Ok(GeneratedField::EndOfEpoch),
                            "randomnessStateUpdate" | "randomness_state_update" => Ok(GeneratedField::RandomnessStateUpdate),
                            "consensusCommitPrologueV2" | "consensus_commit_prologue_v2" => Ok(GeneratedField::ConsensusCommitPrologueV2),
                            "consensusCommitPrologueV3" | "consensus_commit_prologue_v3" => Ok(GeneratedField::ConsensusCommitPrologueV3),
                            "consensusCommitPrologueV4" | "consensus_commit_prologue_v4" => Ok(GeneratedField::ConsensusCommitPrologueV4),
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
                        GeneratedField::ProgrammableTransaction => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("programmableTransaction"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ProgrammableTransaction)
;
                        }
                        GeneratedField::ProgrammableSystemTransaction => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("programmableSystemTransaction"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ProgrammableSystemTransaction)
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
                        GeneratedField::ConsensusCommitPrologueV1 => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("consensusCommitPrologueV1"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ConsensusCommitPrologueV1)
;
                        }
                        GeneratedField::AuthenticatorStateUpdate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("authenticatorStateUpdate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::AuthenticatorStateUpdate)
;
                        }
                        GeneratedField::EndOfEpoch => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("endOfEpoch"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::EndOfEpoch)
;
                        }
                        GeneratedField::RandomnessStateUpdate => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("randomnessStateUpdate"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::RandomnessStateUpdate)
;
                        }
                        GeneratedField::ConsensusCommitPrologueV2 => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("consensusCommitPrologueV2"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ConsensusCommitPrologueV2)
;
                        }
                        GeneratedField::ConsensusCommitPrologueV3 => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("consensusCommitPrologueV3"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ConsensusCommitPrologueV3)
;
                        }
                        GeneratedField::ConsensusCommitPrologueV4 => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("consensusCommitPrologueV4"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<_>>()?.map(transaction_kind::Kind::ConsensusCommitPrologueV4)
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
        if self.address.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TransferObjects", len)?;
        if !self.objects.is_empty() {
            struct_ser.serialize_field("objects", &self.objects)?;
        }
        if let Some(v) = self.address.as_ref() {
            struct_ser.serialize_field("address", v)?;
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
            "address",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Objects,
            Address,
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
                            "address" => Ok(GeneratedField::Address),
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
                let mut address__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Objects => {
                            if objects__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objects"));
                            }
                            objects__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Address => {
                            if address__.is_some() {
                                return Err(serde::de::Error::duplicate_field("address"));
                            }
                            address__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TransferObjects {
                    objects: objects__.unwrap_or_default(),
                    address: address__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TransferObjects", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TypeArgumentError {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.type_argument.is_some() {
            len += 1;
        }
        if self.kind.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TypeArgumentError", len)?;
        if let Some(v) = self.type_argument.as_ref() {
            struct_ser.serialize_field("typeArgument", v)?;
        }
        if let Some(v) = self.kind.as_ref() {
            let v = type_argument_error::TypeArgumentErrorKind::try_from(*v)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TypeArgumentError {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "type_argument",
            "typeArgument",
            "kind",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TypeArgument,
            Kind,
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
                            "typeArgument" | "type_argument" => Ok(GeneratedField::TypeArgument),
                            "kind" => Ok(GeneratedField::Kind),
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
            type Value = TypeArgumentError;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TypeArgumentError")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TypeArgumentError, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut type_argument__ = None;
                let mut kind__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::TypeArgument => {
                            if type_argument__.is_some() {
                                return Err(serde::de::Error::duplicate_field("typeArgument"));
                            }
                            type_argument__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value::<::std::option::Option<type_argument_error::TypeArgumentErrorKind>>()?.map(|x| x as i32);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TypeArgumentError {
                    type_argument: type_argument__,
                    kind: kind__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TypeArgumentError", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for type_argument_error::TypeArgumentErrorKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "TYPE_ARGUMENT_ERROR_KIND_UNKNOWN",
            Self::TypeNotFound => "TYPE_NOT_FOUND",
            Self::ConstraintNotSatisfied => "CONSTRAINT_NOT_SATISFIED",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for type_argument_error::TypeArgumentErrorKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "TYPE_ARGUMENT_ERROR_KIND_UNKNOWN",
            "TYPE_NOT_FOUND",
            "CONSTRAINT_NOT_SATISFIED",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = type_argument_error::TypeArgumentErrorKind;

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
                    "TYPE_ARGUMENT_ERROR_KIND_UNKNOWN" => Ok(type_argument_error::TypeArgumentErrorKind::Unknown),
                    "TYPE_NOT_FOUND" => Ok(type_argument_error::TypeArgumentErrorKind::TypeNotFound),
                    "CONSTRAINT_NOT_SATISFIED" => Ok(type_argument_error::TypeArgumentErrorKind::ConstraintNotSatisfied),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for TypeOrigin {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.module_name.is_some() {
            len += 1;
        }
        if self.datatype_name.is_some() {
            len += 1;
        }
        if self.package_id.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TypeOrigin", len)?;
        if let Some(v) = self.module_name.as_ref() {
            struct_ser.serialize_field("moduleName", v)?;
        }
        if let Some(v) = self.datatype_name.as_ref() {
            struct_ser.serialize_field("datatypeName", v)?;
        }
        if let Some(v) = self.package_id.as_ref() {
            struct_ser.serialize_field("packageId", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TypeOrigin {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "module_name",
            "moduleName",
            "datatype_name",
            "datatypeName",
            "package_id",
            "packageId",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ModuleName,
            DatatypeName,
            PackageId,
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
                            "moduleName" | "module_name" => Ok(GeneratedField::ModuleName),
                            "datatypeName" | "datatype_name" => Ok(GeneratedField::DatatypeName),
                            "packageId" | "package_id" => Ok(GeneratedField::PackageId),
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
            type Value = TypeOrigin;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TypeOrigin")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TypeOrigin, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut module_name__ = None;
                let mut datatype_name__ = None;
                let mut package_id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ModuleName => {
                            if module_name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("moduleName"));
                            }
                            module_name__ = map_.next_value()?;
                        }
                        GeneratedField::DatatypeName => {
                            if datatype_name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("datatypeName"));
                            }
                            datatype_name__ = map_.next_value()?;
                        }
                        GeneratedField::PackageId => {
                            if package_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("packageId"));
                            }
                            package_id__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TypeOrigin {
                    module_name: module_name__,
                    datatype_name: datatype_name__,
                    package_id: package_id__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TypeOrigin", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TypeParameter {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.constraints.is_empty() {
            len += 1;
        }
        if self.is_phantom.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.TypeParameter", len)?;
        if !self.constraints.is_empty() {
            let v = self.constraints.iter().cloned().map(|v| {
                Ability::try_from(v)
                    .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", v)))
                }).collect::<std::result::Result<Vec<_>, _>>()?;
            struct_ser.serialize_field("constraints", &v)?;
        }
        if let Some(v) = self.is_phantom.as_ref() {
            struct_ser.serialize_field("isPhantom", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TypeParameter {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "constraints",
            "is_phantom",
            "isPhantom",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Constraints,
            IsPhantom,
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
                            "constraints" => Ok(GeneratedField::Constraints),
                            "isPhantom" | "is_phantom" => Ok(GeneratedField::IsPhantom),
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
            type Value = TypeParameter;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.TypeParameter")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TypeParameter, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut constraints__ = None;
                let mut is_phantom__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Constraints => {
                            if constraints__.is_some() {
                                return Err(serde::de::Error::duplicate_field("constraints"));
                            }
                            constraints__ = Some(map_.next_value::<Vec<Ability>>()?.into_iter().map(|x| x as i32).collect());
                        }
                        GeneratedField::IsPhantom => {
                            if is_phantom__.is_some() {
                                return Err(serde::de::Error::duplicate_field("isPhantom"));
                            }
                            is_phantom__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(TypeParameter {
                    constraints: constraints__.unwrap_or_default(),
                    is_phantom: is_phantom__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.TypeParameter", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for UnchangedConsensusObject {
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
        let mut struct_ser = serializer.serialize_struct("soma.rpc.UnchangedConsensusObject", len)?;
        if let Some(v) = self.kind.as_ref() {
            let v = unchanged_consensus_object::UnchangedConsensusObjectKind::try_from(*v)
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
impl<'de> serde::Deserialize<'de> for UnchangedConsensusObject {
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
            type Value = UnchangedConsensusObject;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.UnchangedConsensusObject")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<UnchangedConsensusObject, V::Error>
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
                            kind__ = map_.next_value::<::std::option::Option<unchanged_consensus_object::UnchangedConsensusObjectKind>>()?.map(|x| x as i32);
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
                Ok(UnchangedConsensusObject {
                    kind: kind__,
                    object_id: object_id__,
                    version: version__,
                    digest: digest__,
                    object_type: object_type__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.UnchangedConsensusObject", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for unchanged_consensus_object::UnchangedConsensusObjectKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unknown => "UNCHANGED_CONSENSUS_OBJECT_KIND_UNKNOWN",
            Self::ReadOnlyRoot => "READ_ONLY_ROOT",
            Self::MutateConsensusStreamEnded => "MUTATE_CONSENSUS_STREAM_ENDED",
            Self::ReadConsensusStreamEnded => "READ_CONSENSUS_STREAM_ENDED",
            Self::Canceled => "CANCELED",
            Self::PerEpochConfig => "PER_EPOCH_CONFIG",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for unchanged_consensus_object::UnchangedConsensusObjectKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "UNCHANGED_CONSENSUS_OBJECT_KIND_UNKNOWN",
            "READ_ONLY_ROOT",
            "MUTATE_CONSENSUS_STREAM_ENDED",
            "READ_CONSENSUS_STREAM_ENDED",
            "CANCELED",
            "PER_EPOCH_CONFIG",
        ];

        struct GeneratedVisitor;

        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = unchanged_consensus_object::UnchangedConsensusObjectKind;

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
                    "UNCHANGED_CONSENSUS_OBJECT_KIND_UNKNOWN" => Ok(unchanged_consensus_object::UnchangedConsensusObjectKind::Unknown),
                    "READ_ONLY_ROOT" => Ok(unchanged_consensus_object::UnchangedConsensusObjectKind::ReadOnlyRoot),
                    "MUTATE_CONSENSUS_STREAM_ENDED" => Ok(unchanged_consensus_object::UnchangedConsensusObjectKind::MutateConsensusStreamEnded),
                    "READ_CONSENSUS_STREAM_ENDED" => Ok(unchanged_consensus_object::UnchangedConsensusObjectKind::ReadConsensusStreamEnded),
                    "CANCELED" => Ok(unchanged_consensus_object::UnchangedConsensusObjectKind::Canceled),
                    "PER_EPOCH_CONFIG" => Ok(unchanged_consensus_object::UnchangedConsensusObjectKind::PerEpochConfig),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for Upgrade {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.modules.is_empty() {
            len += 1;
        }
        if !self.dependencies.is_empty() {
            len += 1;
        }
        if self.package.is_some() {
            len += 1;
        }
        if self.ticket.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.Upgrade", len)?;
        if !self.modules.is_empty() {
            struct_ser.serialize_field("modules", &self.modules.iter().map(crate::utils::_serde::base64::encode).collect::<Vec<_>>())?;
        }
        if !self.dependencies.is_empty() {
            struct_ser.serialize_field("dependencies", &self.dependencies)?;
        }
        if let Some(v) = self.package.as_ref() {
            struct_ser.serialize_field("package", v)?;
        }
        if let Some(v) = self.ticket.as_ref() {
            struct_ser.serialize_field("ticket", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Upgrade {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "modules",
            "dependencies",
            "package",
            "ticket",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Modules,
            Dependencies,
            Package,
            Ticket,
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
                            "modules" => Ok(GeneratedField::Modules),
                            "dependencies" => Ok(GeneratedField::Dependencies),
                            "package" => Ok(GeneratedField::Package),
                            "ticket" => Ok(GeneratedField::Ticket),
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
            type Value = Upgrade;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.Upgrade")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Upgrade, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut modules__ = None;
                let mut dependencies__ = None;
                let mut package__ = None;
                let mut ticket__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Modules => {
                            if modules__.is_some() {
                                return Err(serde::de::Error::duplicate_field("modules"));
                            }
                            modules__ = 
                                Some(map_.next_value::<Vec<crate::utils::_serde::BytesDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::Dependencies => {
                            if dependencies__.is_some() {
                                return Err(serde::de::Error::duplicate_field("dependencies"));
                            }
                            dependencies__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Package => {
                            if package__.is_some() {
                                return Err(serde::de::Error::duplicate_field("package"));
                            }
                            package__ = map_.next_value()?;
                        }
                        GeneratedField::Ticket => {
                            if ticket__.is_some() {
                                return Err(serde::de::Error::duplicate_field("ticket"));
                            }
                            ticket__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Upgrade {
                    modules: modules__.unwrap_or_default(),
                    dependencies: dependencies__.unwrap_or_default(),
                    package: package__,
                    ticket: ticket__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.Upgrade", FIELDS, GeneratedVisitor)
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
                user_signature::Signature::Multisig(v) => {
                    struct_ser.serialize_field("multisig", v)?;
                }
                user_signature::Signature::Zklogin(v) => {
                    struct_ser.serialize_field("zklogin", v)?;
                }
                user_signature::Signature::Passkey(v) => {
                    struct_ser.serialize_field("passkey", v)?;
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
            "multisig",
            "zklogin",
            "passkey",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Bcs,
            Scheme,
            Simple,
            Multisig,
            Zklogin,
            Passkey,
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
                            "multisig" => Ok(GeneratedField::Multisig),
                            "zklogin" => Ok(GeneratedField::Zklogin),
                            "passkey" => Ok(GeneratedField::Passkey),
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
                        GeneratedField::Multisig => {
                            if signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("multisig"));
                            }
                            signature__ = map_.next_value::<::std::option::Option<_>>()?.map(user_signature::Signature::Multisig)
;
                        }
                        GeneratedField::Zklogin => {
                            if signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("zklogin"));
                            }
                            signature__ = map_.next_value::<::std::option::Option<_>>()?.map(user_signature::Signature::Zklogin)
;
                        }
                        GeneratedField::Passkey => {
                            if signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("passkey"));
                            }
                            signature__ = map_.next_value::<::std::option::Option<_>>()?.map(user_signature::Signature::Passkey)
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
impl serde::Serialize for ValidatorExecutionTimeObservation {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.validator.is_some() {
            len += 1;
        }
        if self.duration.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ValidatorExecutionTimeObservation", len)?;
        if let Some(v) = self.validator.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("validator", crate::utils::_serde::base64::encode(&v).as_str())?;
        }
        if let Some(v) = self.duration.as_ref() {
            struct_ser.serialize_field("duration", &crate::utils::_serde::DurationSerializer(v))?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ValidatorExecutionTimeObservation {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "validator",
            "duration",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Validator,
            Duration,
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
                            "validator" => Ok(GeneratedField::Validator),
                            "duration" => Ok(GeneratedField::Duration),
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
            type Value = ValidatorExecutionTimeObservation;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ValidatorExecutionTimeObservation")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ValidatorExecutionTimeObservation, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut validator__ = None;
                let mut duration__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Validator => {
                            if validator__.is_some() {
                                return Err(serde::de::Error::duplicate_field("validator"));
                            }
                            validator__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::BytesDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Duration => {
                            if duration__.is_some() {
                                return Err(serde::de::Error::duplicate_field("duration"));
                            }
                            duration__ = map_.next_value::<::std::option::Option<crate::utils::_serde::DurationDeserializer>>()?.map(|x| x.0.into());
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ValidatorExecutionTimeObservation {
                    validator: validator__,
                    duration: duration__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ValidatorExecutionTimeObservation", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for VariantDescriptor {
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
        if self.position.is_some() {
            len += 1;
        }
        if !self.fields.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.VariantDescriptor", len)?;
        if let Some(v) = self.name.as_ref() {
            struct_ser.serialize_field("name", v)?;
        }
        if let Some(v) = self.position.as_ref() {
            struct_ser.serialize_field("position", v)?;
        }
        if !self.fields.is_empty() {
            struct_ser.serialize_field("fields", &self.fields)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for VariantDescriptor {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "name",
            "position",
            "fields",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Name,
            Position,
            Fields,
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
                            "position" => Ok(GeneratedField::Position),
                            "fields" => Ok(GeneratedField::Fields),
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
            type Value = VariantDescriptor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.VariantDescriptor")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<VariantDescriptor, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut name__ = None;
                let mut position__ = None;
                let mut fields__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Name => {
                            if name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("name"));
                            }
                            name__ = map_.next_value()?;
                        }
                        GeneratedField::Position => {
                            if position__.is_some() {
                                return Err(serde::de::Error::duplicate_field("position"));
                            }
                            position__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Fields => {
                            if fields__.is_some() {
                                return Err(serde::de::Error::duplicate_field("fields"));
                            }
                            fields__ = Some(map_.next_value()?);
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(VariantDescriptor {
                    name: name__,
                    position: position__,
                    fields: fields__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.VariantDescriptor", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for VersionAssignment {
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
        if self.start_version.is_some() {
            len += 1;
        }
        if self.version.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.VersionAssignment", len)?;
        if let Some(v) = self.object_id.as_ref() {
            struct_ser.serialize_field("objectId", v)?;
        }
        if let Some(v) = self.start_version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("startVersion", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.version.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("version", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for VersionAssignment {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "object_id",
            "objectId",
            "start_version",
            "startVersion",
            "version",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ObjectId,
            StartVersion,
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
                            "objectId" | "object_id" => Ok(GeneratedField::ObjectId),
                            "startVersion" | "start_version" => Ok(GeneratedField::StartVersion),
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
            type Value = VersionAssignment;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.VersionAssignment")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<VersionAssignment, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut object_id__ = None;
                let mut start_version__ = None;
                let mut version__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ObjectId => {
                            if object_id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("objectId"));
                            }
                            object_id__ = map_.next_value()?;
                        }
                        GeneratedField::StartVersion => {
                            if start_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("startVersion"));
                            }
                            start_version__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
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
                Ok(VersionAssignment {
                    object_id: object_id__,
                    start_version: start_version__,
                    version: version__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.VersionAssignment", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ZkLoginAuthenticator {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.inputs.is_some() {
            len += 1;
        }
        if self.max_epoch.is_some() {
            len += 1;
        }
        if self.signature.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ZkLoginAuthenticator", len)?;
        if let Some(v) = self.inputs.as_ref() {
            struct_ser.serialize_field("inputs", v)?;
        }
        if let Some(v) = self.max_epoch.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("maxEpoch", ToString::to_string(&v).as_str())?;
        }
        if let Some(v) = self.signature.as_ref() {
            struct_ser.serialize_field("signature", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ZkLoginAuthenticator {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "inputs",
            "max_epoch",
            "maxEpoch",
            "signature",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Inputs,
            MaxEpoch,
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
                            "inputs" => Ok(GeneratedField::Inputs),
                            "maxEpoch" | "max_epoch" => Ok(GeneratedField::MaxEpoch),
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
            type Value = ZkLoginAuthenticator;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ZkLoginAuthenticator")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ZkLoginAuthenticator, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut inputs__ = None;
                let mut max_epoch__ = None;
                let mut signature__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Inputs => {
                            if inputs__.is_some() {
                                return Err(serde::de::Error::duplicate_field("inputs"));
                            }
                            inputs__ = map_.next_value()?;
                        }
                        GeneratedField::MaxEpoch => {
                            if max_epoch__.is_some() {
                                return Err(serde::de::Error::duplicate_field("maxEpoch"));
                            }
                            max_epoch__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::Signature => {
                            if signature__.is_some() {
                                return Err(serde::de::Error::duplicate_field("signature"));
                            }
                            signature__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ZkLoginAuthenticator {
                    inputs: inputs__,
                    max_epoch: max_epoch__,
                    signature: signature__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ZkLoginAuthenticator", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ZkLoginClaim {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.value.is_some() {
            len += 1;
        }
        if self.index_mod_4.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ZkLoginClaim", len)?;
        if let Some(v) = self.value.as_ref() {
            struct_ser.serialize_field("value", v)?;
        }
        if let Some(v) = self.index_mod_4.as_ref() {
            struct_ser.serialize_field("indexMod4", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ZkLoginClaim {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "value",
            "index_mod_4",
            "indexMod4",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Value,
            IndexMod4,
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
                            "value" => Ok(GeneratedField::Value),
                            "indexMod4" | "index_mod_4" => Ok(GeneratedField::IndexMod4),
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
            type Value = ZkLoginClaim;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ZkLoginClaim")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ZkLoginClaim, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut value__ = None;
                let mut index_mod_4__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Value => {
                            if value__.is_some() {
                                return Err(serde::de::Error::duplicate_field("value"));
                            }
                            value__ = map_.next_value()?;
                        }
                        GeneratedField::IndexMod4 => {
                            if index_mod_4__.is_some() {
                                return Err(serde::de::Error::duplicate_field("indexMod4"));
                            }
                            index_mod_4__ = 
                                map_.next_value::<::std::option::Option<crate::utils::_serde::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ZkLoginClaim {
                    value: value__,
                    index_mod_4: index_mod_4__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ZkLoginClaim", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ZkLoginInputs {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.proof_points.is_some() {
            len += 1;
        }
        if self.iss_base64_details.is_some() {
            len += 1;
        }
        if self.header_base64.is_some() {
            len += 1;
        }
        if self.address_seed.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ZkLoginInputs", len)?;
        if let Some(v) = self.proof_points.as_ref() {
            struct_ser.serialize_field("proofPoints", v)?;
        }
        if let Some(v) = self.iss_base64_details.as_ref() {
            struct_ser.serialize_field("issBase64Details", v)?;
        }
        if let Some(v) = self.header_base64.as_ref() {
            struct_ser.serialize_field("headerBase64", v)?;
        }
        if let Some(v) = self.address_seed.as_ref() {
            struct_ser.serialize_field("addressSeed", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ZkLoginInputs {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "proof_points",
            "proofPoints",
            "iss_base64_details",
            "issBase64Details",
            "header_base64",
            "headerBase64",
            "address_seed",
            "addressSeed",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            ProofPoints,
            IssBase64Details,
            HeaderBase64,
            AddressSeed,
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
                            "proofPoints" | "proof_points" => Ok(GeneratedField::ProofPoints),
                            "issBase64Details" | "iss_base64_details" => Ok(GeneratedField::IssBase64Details),
                            "headerBase64" | "header_base64" => Ok(GeneratedField::HeaderBase64),
                            "addressSeed" | "address_seed" => Ok(GeneratedField::AddressSeed),
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
            type Value = ZkLoginInputs;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ZkLoginInputs")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ZkLoginInputs, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut proof_points__ = None;
                let mut iss_base64_details__ = None;
                let mut header_base64__ = None;
                let mut address_seed__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::ProofPoints => {
                            if proof_points__.is_some() {
                                return Err(serde::de::Error::duplicate_field("proofPoints"));
                            }
                            proof_points__ = map_.next_value()?;
                        }
                        GeneratedField::IssBase64Details => {
                            if iss_base64_details__.is_some() {
                                return Err(serde::de::Error::duplicate_field("issBase64Details"));
                            }
                            iss_base64_details__ = map_.next_value()?;
                        }
                        GeneratedField::HeaderBase64 => {
                            if header_base64__.is_some() {
                                return Err(serde::de::Error::duplicate_field("headerBase64"));
                            }
                            header_base64__ = map_.next_value()?;
                        }
                        GeneratedField::AddressSeed => {
                            if address_seed__.is_some() {
                                return Err(serde::de::Error::duplicate_field("addressSeed"));
                            }
                            address_seed__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ZkLoginInputs {
                    proof_points: proof_points__,
                    iss_base64_details: iss_base64_details__,
                    header_base64: header_base64__,
                    address_seed: address_seed__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ZkLoginInputs", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ZkLoginProof {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.a.is_some() {
            len += 1;
        }
        if self.b.is_some() {
            len += 1;
        }
        if self.c.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ZkLoginProof", len)?;
        if let Some(v) = self.a.as_ref() {
            struct_ser.serialize_field("a", v)?;
        }
        if let Some(v) = self.b.as_ref() {
            struct_ser.serialize_field("b", v)?;
        }
        if let Some(v) = self.c.as_ref() {
            struct_ser.serialize_field("c", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ZkLoginProof {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "a",
            "b",
            "c",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            A,
            B,
            C,
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
                            "a" => Ok(GeneratedField::A),
                            "b" => Ok(GeneratedField::B),
                            "c" => Ok(GeneratedField::C),
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
            type Value = ZkLoginProof;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ZkLoginProof")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ZkLoginProof, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut a__ = None;
                let mut b__ = None;
                let mut c__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::A => {
                            if a__.is_some() {
                                return Err(serde::de::Error::duplicate_field("a"));
                            }
                            a__ = map_.next_value()?;
                        }
                        GeneratedField::B => {
                            if b__.is_some() {
                                return Err(serde::de::Error::duplicate_field("b"));
                            }
                            b__ = map_.next_value()?;
                        }
                        GeneratedField::C => {
                            if c__.is_some() {
                                return Err(serde::de::Error::duplicate_field("c"));
                            }
                            c__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ZkLoginProof {
                    a: a__,
                    b: b__,
                    c: c__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ZkLoginProof", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ZkLoginPublicIdentifier {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.iss.is_some() {
            len += 1;
        }
        if self.address_seed.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("soma.rpc.ZkLoginPublicIdentifier", len)?;
        if let Some(v) = self.iss.as_ref() {
            struct_ser.serialize_field("iss", v)?;
        }
        if let Some(v) = self.address_seed.as_ref() {
            struct_ser.serialize_field("addressSeed", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ZkLoginPublicIdentifier {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "iss",
            "address_seed",
            "addressSeed",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Iss,
            AddressSeed,
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
                            "iss" => Ok(GeneratedField::Iss),
                            "addressSeed" | "address_seed" => Ok(GeneratedField::AddressSeed),
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
            type Value = ZkLoginPublicIdentifier;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct soma.rpc.ZkLoginPublicIdentifier")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ZkLoginPublicIdentifier, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut iss__ = None;
                let mut address_seed__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Iss => {
                            if iss__.is_some() {
                                return Err(serde::de::Error::duplicate_field("iss"));
                            }
                            iss__ = map_.next_value()?;
                        }
                        GeneratedField::AddressSeed => {
                            if address_seed__.is_some() {
                                return Err(serde::de::Error::duplicate_field("addressSeed"));
                            }
                            address_seed__ = map_.next_value()?;
                        }
                        GeneratedField::__SkipField__ => {
                            let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ZkLoginPublicIdentifier {
                    iss: iss__,
                    address_seed: address_seed__,
                })
            }
        }
        deserializer.deserialize_struct("soma.rpc.ZkLoginPublicIdentifier", FIELDS, GeneratedVisitor)
    }
}
