use prost_types::FieldMask;
use std::time::Duration;
use tap::Pipe;
use tonic::Status;
use tonic::metadata::MetadataMap;
use tonic::transport::channel::ClientTlsConfig;
use types::effects::TransactionEffects;
use types::shard::Shard;
use types::transaction::Transaction;

use crate::proto::TryFromProtoError;
use crate::proto::soma::transaction_execution_service_client::TransactionExecutionServiceClient;
use crate::utils::field::FieldMaskUtil;
use crate::utils::types_conversions::SdkTypeConversionError;

pub type Result<T, E = tonic::Status> = std::result::Result<T, E>;
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Clone)]
pub struct Client {
    #[allow(unused)]
    uri: http::Uri,
    channel: tonic::transport::Channel,
    auth: AuthInterceptor,
}

impl Client {
    pub fn new<T>(uri: T) -> Result<Self>
    where
        T: TryInto<http::Uri>,
        T::Error: Into<BoxError>,
    {
        let uri = uri
            .try_into()
            .map_err(Into::into)
            .map_err(Status::from_error)?;
        let mut endpoint = tonic::transport::Endpoint::from(uri.clone());
        if uri.scheme() == Some(&http::uri::Scheme::HTTPS) {
            endpoint = endpoint
                .tls_config(ClientTlsConfig::new().with_enabled_roots())
                .map_err(Into::into)
                .map_err(Status::from_error)?;
        }
        let channel = endpoint
            .connect_timeout(Duration::from_secs(5))
            .http2_keep_alive_interval(Duration::from_secs(5))
            .connect_lazy();

        Ok(Self {
            uri,
            channel,
            auth: Default::default(),
        })
    }

    pub fn with_auth(mut self, auth: AuthInterceptor) -> Self {
        self.auth = auth;
        self
    }

    // pub fn raw_client(
    //     &self,
    // ) -> LedgerServiceClient<
    //     tonic::service::interceptor::InterceptedService<tonic::transport::Channel, AuthInterceptor>,
    // > {
    //     LedgerServiceClient::with_interceptor(self.channel.clone(), self.auth.clone())
    // }

    pub fn execution_client(
        &self,
    ) -> TransactionExecutionServiceClient<
        tonic::service::interceptor::InterceptedService<tonic::transport::Channel, AuthInterceptor>,
    > {
        TransactionExecutionServiceClient::with_interceptor(self.channel.clone(), self.auth.clone())
    }

    //      pub async fn get_latest_checkpoint(&self) -> Result<CertifiedCheckpointSummary> {
    //         self.get_checkpoint_internal(None).await
    //     }
    //  async fn get_checkpoint_internal(
    //         &self,
    //         sequence_number: Option<CheckpointSequenceNumber>,
    //     ) -> Result<CertifiedCheckpointSummary> {
    //         let mut request = proto::GetCheckpointRequest::default()
    //             .with_read_mask(FieldMask::from_paths(["summary.bcs", "signature"]));
    //         request.checkpoint_id = sequence_number.map(|sequence_number| {
    //             proto::get_checkpoint_request::CheckpointId::SequenceNumber(sequence_number)
    //         });

    //         let (metadata, checkpoint, _extentions) = self
    //             .raw_client()
    //             .get_checkpoint(request)
    //             .await?
    //             .into_parts();

    //         let checkpoint = checkpoint
    //             .checkpoint
    //             .ok_or_else(|| tonic::Status::not_found("no checkpoint returned"))?;
    //         certified_checkpoint_summary_try_from_proto(&checkpoint)
    //             .map_err(|e| status_from_error_with_metadata(e, metadata))
    //     }

    //      pub async fn get_object(&self, object_id: ObjectID) -> Result<Object> {
    //         self.get_object_internal(object_id, None).await
    //     }

    //     pub async fn get_object_with_version(
    //         &self,
    //         object_id: ObjectID,
    //         version: SequenceNumber,
    //     ) -> Result<Object> {
    //         self.get_object_internal(object_id, Some(version.value()))
    //             .await
    //     }

    //     async fn get_object_internal(
    //         &self,
    //         object_id: ObjectID,
    //         version: Option<u64>,
    //     ) -> Result<Object> {
    //         let mut request = proto::GetObjectRequest::new(&object_id.into())
    //             .with_read_mask(FieldMask::from_paths(["bcs"]));
    //         request.version = version;

    //         let (metadata, object, _extentions) =
    //             self.raw_client().get_object(request).await?.into_parts();

    //         let object = object
    //             .object
    //             .ok_or_else(|| tonic::Status::not_found("no object returned"))?;
    //         object_try_from_proto(&object).map_err(|e| status_from_error_with_metadata(e, metadata))
    //     }

    pub async fn execute_transaction(
        &self,
        transaction: &Transaction,
    ) -> Result<TransactionExecutionResponse> {
        let signatures = transaction
            .inner()
            .tx_signatures
            .iter()
            .map(|signature| {
                let message = signature.clone().into();
                message
            })
            .collect();

        let request = crate::proto::soma::ExecuteTransactionRequest::new({
            let mut tx = crate::proto::soma::Transaction::default();
            // tx.bcs = Some(
            //     crate::proto::soma::Bcs::serialize(&transaction.inner().intent_message.value)
            //         .map_err(|e| Status::from_error(e.into()))?,
            // );

            tx.kind = Some(transaction.inner().intent_message.value.kind.clone().into());
            tx.sender = Some((&transaction.inner().intent_message.value.sender()).into());
            tx.gas_payment = transaction
                .inner()
                .intent_message
                .value
                .gas_payment
                .iter()
                .map(|o| o.clone().into())
                .map(|o: crate::types::ObjectReference| o.into())
                .collect();
            tx
        })
        .with_signatures(signatures)
        .with_read_mask(FieldMask::from_paths([
            "finality",
            "transaction.effects",
            "transaction.balance_changes",
            "transaction.input_objects",
            "transaction.output_objects",
            "transaction.shard",
        ]));

        let (metadata, response, _extentions) = self
            .execution_client()
            .execute_transaction(request)
            .await?
            .into_parts();

        execute_transaction_response_try_from_proto(&response)
            .map_err(|e| status_from_error_with_metadata(e, metadata))
    }
}

#[derive(Debug)]
pub struct TransactionExecutionResponse {
    pub finality: crate::proto::soma::TransactionFinality,

    pub effects: TransactionEffects,
    pub balance_changes: Vec<crate::types::BalanceChange>,
    pub input_objects: Vec<types::object::Object>,
    pub output_objects: Vec<types::object::Object>,
    pub shard: Option<Shard>,
}

/// Attempts to parse `TransactionExecutionResponse` from the fields in `TransactionExecutionResponse`
#[allow(clippy::result_large_err)]
fn execute_transaction_response_try_from_proto(
    response: &crate::proto::soma::ExecuteTransactionResponse,
) -> Result<TransactionExecutionResponse, TryFromProtoError> {
    let finality = response
        .finality
        .clone()
        .ok_or_else(|| TryFromProtoError::missing("finality"))?;

    let executed_transaction = response
        .transaction
        .as_ref()
        .ok_or_else(|| TryFromProtoError::missing("transaction"))?;

    // First convert from proto to SDK type, then to domain type
    let effects = {
        let proto_effects = executed_transaction
            .effects
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("effects"))?;

        // Proto → SDK type
        let sdk_effects: crate::types::TransactionEffects = proto_effects.try_into()?;

        // SDK type → Domain type
        let domain_effects: types::effects::TransactionEffects = sdk_effects
            .try_into()
            .map_err(|e| TryFromProtoError::invalid("effects", e))?;

        domain_effects
    };
    // .map_err(|e| TryFromProtoError::invalid("effects", e))?;

    let balance_changes = executed_transaction
        .balance_changes
        .iter()
        .map(TryInto::try_into)
        .collect::<Result<_, _>>()?;

    let input_objects = executed_transaction
        .input_objects
        .iter()
        .map(|obj| {
            let obj: crate::types::Object = obj.try_into()?;
            obj.try_into()
                .map_err(|e| TryFromProtoError::invalid("input_objects", e))
        })
        .collect::<Result<Vec<_>, TryFromProtoError>>()
        .map_err(|e| TryFromProtoError::invalid("input_objects", e))?;

    let output_objects = executed_transaction
        .output_objects
        .iter()
        .map(|obj| {
            let obj: crate::types::Object = obj.try_into()?;
            obj.try_into()
                .map_err(|e| TryFromProtoError::invalid("output_objects", e))
        })
        .collect::<Result<Vec<_>, TryFromProtoError>>()
        .map_err(|e| TryFromProtoError::invalid("output_objects", e))?;

    let shard = executed_transaction
        .shard
        .as_ref()
        .map(|proto_shard| {
            // Proto → SDK
            let sdk_shard: crate::types::Shard = proto_shard.try_into()?;
            // SDK → Domain
            let domain_shard: types::shard::Shard = sdk_shard
                .try_into()
                .map_err(|e| TryFromProtoError::invalid("shard", e))?;
            Ok(domain_shard)
        })
        .transpose()?;

    TransactionExecutionResponse {
        finality,
        effects,
        balance_changes,
        input_objects,
        output_objects,
        shard,
    }
    .pipe(Ok)
}

fn status_from_error_with_metadata<T: Into<BoxError>>(err: T, metadata: MetadataMap) -> Status {
    let mut status = Status::from_error(err.into());
    *status.metadata_mut() = metadata;
    status
}

#[derive(Clone, Debug, Default)]
pub struct AuthInterceptor {
    auth: Option<tonic::metadata::MetadataValue<tonic::metadata::Ascii>>,
}

impl AuthInterceptor {
    /// Enable HTTP basic authentication with a username and optional password.
    pub fn basic<U, P>(username: U, password: Option<P>) -> Self
    where
        U: std::fmt::Display,
        P: std::fmt::Display,
    {
        use base64::prelude::BASE64_STANDARD;
        use base64::write::EncoderWriter;
        use std::io::Write;

        let mut buf = b"Basic ".to_vec();
        {
            let mut encoder = EncoderWriter::new(&mut buf, &BASE64_STANDARD);
            let _ = write!(encoder, "{username}:");
            if let Some(password) = password {
                let _ = write!(encoder, "{password}");
            }
        }
        let mut header = tonic::metadata::MetadataValue::try_from(buf)
            .expect("base64 is always valid HeaderValue");
        header.set_sensitive(true);

        Self { auth: Some(header) }
    }

    /// Enable HTTP bearer authentication.
    pub fn bearer<T>(token: T) -> Self
    where
        T: std::fmt::Display,
    {
        let header_value = format!("Bearer {token}");
        let mut header = tonic::metadata::MetadataValue::try_from(header_value)
            .expect("token is always valid HeaderValue");
        header.set_sensitive(true);

        Self { auth: Some(header) }
    }
}

impl tonic::service::Interceptor for AuthInterceptor {
    fn call(
        &mut self,
        mut request: tonic::Request<()>,
    ) -> std::result::Result<tonic::Request<()>, Status> {
        if let Some(auth) = self.auth.clone() {
            request
                .metadata_mut()
                .insert(http::header::AUTHORIZATION.as_str(), auth);
        }
        Ok(request)
    }
}
