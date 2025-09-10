use crate::proto::soma::*;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;
use types::crypto::SomaSignature;

//
// TransactionFee
//

impl From<types::tx_fee::TransactionFee> for TransactionFee {
    fn from(
        types::tx_fee::TransactionFee {
            base_fee,
            operation_fee,
            value_fee,
            total_fee,
            gas_object_ref,
        }: types::tx_fee::TransactionFee,
    ) -> Self {
        let mut message = Self::default();
        message.base_fee = Some(base_fee);
        message.operation_fee = Some(operation_fee);
        message.value_fee = Some(value_fee);
        message.total_fee = Some(total_fee);
        message
    }
}

impl From<types::effects::ExecutionStatus> for ExecutionStatus {
    fn from(value: types::effects::ExecutionStatus) -> Self {
        let mut message = Self::default();
        match value {
            types::effects::ExecutionStatus::Success => {
                message.success = Some(true);
            }
            types::effects::ExecutionStatus::Failure { error } => {
                let description = format!("{error:?}");
                let mut error_message = ExecutionError::from(error);

                error_message.description = Some(description);

                message.success = Some(false);
                message.error = Some(error_message);
            }
        }

        message
    }
}

impl From<types::effects::ExecutionFailureStatus> for ExecutionError {
    fn from(value: types::effects::ExecutionFailureStatus) -> Self {
        use execution_error::ErrorDetails;
        use execution_error::ExecutionErrorKind;
        use types::effects::ExecutionFailureStatus as E;

        let mut message = Self::default();

        let (kind, details) = match value {
            E::InsufficientGas => (ExecutionErrorKind::InsufficientGas, None),
            E::InvalidOwnership { object_id, .. } => (
                ExecutionErrorKind::InvalidOwnership,
                Some(object_id.to_string()),
            ),
            E::ObjectNotFound { object_id } => (
                ExecutionErrorKind::ObjectNotFound,
                Some(object_id.to_string()),
            ),
            E::InvalidObjectType { object_id, .. } => (
                ExecutionErrorKind::InvalidObjectType,
                Some(object_id.to_string()),
            ),
            E::InvalidTransactionType => (ExecutionErrorKind::InvalidTransactionType, None),
            E::InvalidArguments { reason } => (ExecutionErrorKind::InvalidArguments, Some(reason)),
            E::DuplicateValidator => (ExecutionErrorKind::DuplicateValidator, None),
            E::NotAValidator => (ExecutionErrorKind::NotAValidator, None),
            E::ValidatorAlreadyRemoved => (ExecutionErrorKind::ValidatorAlreadyRemoved, None),
            E::AdvancedToWrongEpoch => (ExecutionErrorKind::AdvancedToWrongEpoch, None),
            E::DuplicateEncoder => (ExecutionErrorKind::DuplicateEncoder, None),
            E::NotAnEncoder => (ExecutionErrorKind::NotAnEncoder, None),
            E::EncoderAlreadyRemoved => (ExecutionErrorKind::EncoderAlreadyRemoved, None),
            E::InsufficientCoinBalance => (ExecutionErrorKind::InsufficientCoinBalance, None),
            E::CoinBalanceOverflow => (ExecutionErrorKind::CoinBalanceOverflow, None),
            E::ValidatorNotFound => (ExecutionErrorKind::ValidatorNotFound, None),
            E::EncoderNotFound => (ExecutionErrorKind::EncoderNotFound, None),
            E::StakingPoolNotFound => (ExecutionErrorKind::StakingPoolNotFound, None),
            E::CannotReportOneself => (ExecutionErrorKind::CannotReportOneself, None),
            E::ReportRecordNotFound => (ExecutionErrorKind::ReportRecordNotFound, None),
            E::SomaError(e) => (ExecutionErrorKind::OtherError, Some(e.to_string())),
        };

        message.set_kind(kind);
        message
    }
}

//
// AuthorityQuorumSignInfo aka ValidatorAggregatedSignature
//

impl<const T: bool> From<types::crypto::AuthorityQuorumSignInfo<T>>
    for ValidatorAggregatedSignature
{
    fn from(value: types::crypto::AuthorityQuorumSignInfo<T>) -> Self {
        let mut message = Self::default();
        message.epoch = Some(value.epoch);
        message.signature = Some(value.signature.as_ref().to_vec().into());
        message.bitmap = value.signers_map.iter().collect();
        message
    }
}

//
// ValidatorCommittee
//

impl From<types::committee::Committee> for ValidatorCommittee {
    fn from(value: types::committee::Committee) -> Self {
        let mut message = Self::default();
        message.epoch = Some(value.epoch);
        message.members = value
            .voting_rights
            .into_iter()
            .map(|(name, weight)| {
                let mut member = ValidatorCommitteeMember::default();
                member.public_key = Some(name.0.to_vec().into());
                member.weight = Some(weight);
                member
            })
            .collect();
        message
    }
}

//
// SignatureScheme
//

impl From<types::crypto::SignatureScheme> for SignatureScheme {
    fn from(value: types::crypto::SignatureScheme) -> Self {
        use types::crypto::SignatureScheme as S;

        match value {
            S::ED25519 => Self::Ed25519,
            S::BLS12381 => Self::Bls12381,
        }
    }
}

//
// SimpleSignature
//

impl From<types::crypto::Signature> for SimpleSignature {
    fn from(value: types::crypto::Signature) -> Self {
        let scheme: SignatureScheme = value.scheme().into();
        let signature = value.signature_bytes();
        let public_key = value.public_key_bytes();

        let mut message = Self::default();
        message.scheme = Some(scheme.into());
        message.signature = Some(signature.to_vec().into());
        message.public_key = Some(public_key.to_vec().into());
        message
    }
}

//
// UserSignature
//

impl From<types::crypto::GenericSignature> for UserSignature {
    fn from(value: types::crypto::GenericSignature) -> Self {
        Self::merge_from(value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<types::crypto::GenericSignature> for UserSignature {
    fn merge(&mut self, source: types::crypto::GenericSignature, mask: &FieldMaskTree) {
        use user_signature::Signature;

        if mask.contains(Self::BCS_FIELD) {
            let mut bcs = Bcs::from(source.as_ref().to_vec());
            bcs.name = Some("UserSignatureBytes".to_owned());
            self.bcs = Some(bcs);
        }

        let scheme = match source {
            types::crypto::GenericSignature::Signature(signature) => {
                let scheme = signature.scheme().into();
                if mask.contains(Self::SIMPLE_FIELD) {
                    self.signature = Some(Signature::Simple(signature.into()));
                }
                scheme
            }
        };

        if mask.contains(Self::SCHEME_FIELD) {
            self.set_scheme(scheme);
        }
    }
}

//
// BalanceChange
//

impl From<types::balance_change::BalanceChange> for BalanceChange {
    fn from(value: types::balance_change::BalanceChange) -> Self {
        let mut message = Self::default();
        message.address = Some(value.address.to_string());
        message.amount = Some(value.amount.to_string());
        message
    }
}

impl From<types::object::Object> for Object {
    fn from(value: types::object::Object) -> Self {
        Self::merge_from_types(value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<types::object::Object> for Object {
    fn merge(&mut self, source: types::object::Object, mask: &FieldMaskTree) {
        if mask.contains(Self::BCS_FIELD.name) {
            let mut bcs = Bcs::serialize(&source).unwrap();
            bcs.name = Some("Object".to_owned());
            self.bcs = Some(bcs);
        }

        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = Some(source.digest().to_string());
        }

        if mask.contains(Self::OBJECT_ID_FIELD.name) {
            self.object_id = Some(source.id().to_string());
        }

        if mask.contains(Self::VERSION_FIELD.name) {
            self.version = Some(source.version().value());
        }

        if mask.contains(Self::OWNER_FIELD.name) {
            self.owner = Some(source.owner().to_owned().into());
        }

        if mask.contains(Self::PREVIOUS_TRANSACTION_FIELD.name) {
            self.previous_transaction = Some(source.previous_transaction.to_string());
        }

        // if mask.contains(Self::BALANCE_FIELD) {
        //     self.balance = source.as_coin_maybe().map(|coin| coin.balance.value());
        // }

        self.merge(source, mask);
    }
}

//
// ObjectReference
//

fn object_ref_to_proto(value: types::object::ObjectRef) -> ObjectReference {
    let (object_id, version, digest) = value;
    let mut message = ObjectReference::default();
    message.object_id = Some(object_id.to_string());
    message.version = Some(version.value());
    message.digest = Some(digest.to_string());
    message
}

//
// Owner
//

impl From<types::object::Owner> for Owner {
    fn from(value: types::object::Owner) -> Self {
        use owner::OwnerKind;
        use types::object::Owner as O;

        let mut message = Self::default();

        let kind = match value {
            O::AddressOwner(address) => {
                message.address = Some(address.to_string());
                OwnerKind::Address
            }
            O::Shared {
                initial_shared_version,
            } => {
                message.version = Some(initial_shared_version.value());
                OwnerKind::Shared
            }
            O::Immutable => OwnerKind::Immutable,
        };

        message.set_kind(kind);
        message
    }
}

//
// Transaction
//

impl From<types::transaction::TransactionData> for Transaction {
    fn from(value: types::transaction::TransactionData) -> Self {
        Self::merge_from(value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<types::transaction::TransactionData> for Transaction {
    fn merge(&mut self, source: types::transaction::TransactionData, mask: &FieldMaskTree) {
        if mask.contains(Self::BCS_FIELD.name) {
            let mut bcs = Bcs::serialize(&source).unwrap();
            bcs.name = Some("TransactionData".to_owned());
            self.bcs = Some(bcs);
        }

        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = Some(source.digest().to_string());
        }

        if mask.contains(Self::KIND_FIELD.name) {
            self.kind = Some(source.kind.into());
        }

        if mask.contains(Self::SENDER_FIELD.name) {
            self.sender = Some(source.sender.to_string());
        }

        if mask.contains(Self::GAS_PAYMENT_FIELD.name) {
            self.gas_payment = Some(GasPayment {
                objects: source
                    .gas_payment
                    .into_iter()
                    .map(|g| object_ref_to_proto(g))
                    .collect(),
            });
        }
    }
}

//
// TransactionKind
//

impl From<types::transaction::TransactionKind> for TransactionKind {
    fn from(value: types::transaction::TransactionKind) -> Self {
        use transaction_kind::Kind;
        use types::transaction::TransactionKind as K;

        let kind = match value {
            K::Genesis(genesis) => Kind::Genesis(GenesisTransaction {
                objects: genesis.objects.into_iter().map(Into::into).collect(),
            }),
            K::ConsensusCommitPrologue(prologue) => Kind::ConsensusCommitPrologue(prologue.into()),
            K::ChangeEpoch(change_epoch) => Kind::ChangeEpoch(change_epoch.into()),
            K::AddValidator(args) => Kind::AddValidator(args.into()),
            K::RemoveValidator(args) => Kind::RemoveValidator(args.into()),
            K::ReportValidator { reportee } => Kind::ReportValidator(ReportValidator {
                reportee: Some(reportee.to_string()),
            }),
            K::UndoReportValidator { reportee } => Kind::UndoReportValidator(UndoReportValidator {
                reportee: Some(reportee.to_string()),
            }),
            K::UpdateValidatorMetadata(args) => Kind::UpdateValidatorMetadata(args.into()),
            K::SetCommissionRate { new_rate } => Kind::SetCommissionRate(SetCommissionRate {
                new_rate: Some(new_rate),
            }),
            K::AddEncoder(args) => Kind::AddEncoder(args.into()),
            K::RemoveEncoder => Kind::RemoveEncoder(RemoveEncoder {
                encoder_pubkey_bytes: None, // TODO: Figure out how to deal with this
            }),
            K::ReportEncoder { reportee } => Kind::ReportEncoder(ReportEncoder {
                reportee: Some(reportee.to_string()),
            }),
            K::UndoReportEncoder { reportee } => Kind::UndoReportEncoder(UndoReportEncoder {
                reportee: Some(reportee.to_string()),
            }),
            K::UpdateEncoderMetadata(args) => Kind::UpdateEncoderMetadata(args.into()),
            K::SetEncoderCommissionRate { new_rate } => {
                Kind::SetEncoderCommissionRate(SetEncoderCommissionRate {
                    new_rate: Some(new_rate),
                })
            }
            K::SetEncoderBytePrice { new_price } => {
                Kind::SetEncoderBytePrice(SetEncoderBytePrice {
                    new_price: Some(new_price),
                })
            }
            K::TransferCoin {
                coin,
                amount,
                recipient,
            } => Kind::TransferCoin(TransferCoin {
                coin: Some(object_ref_to_proto(coin)),
                amount: amount,
                recipient: Some(recipient.to_string()),
            }),
            K::PayCoins {
                coins,
                amounts,
                recipients,
            } => Kind::PayCoins(PayCoins {
                coins: coins.into_iter().map(|c| object_ref_to_proto(c)).collect(),
                amounts: amounts.unwrap_or_default(),
                recipients: recipients.into_iter().map(|r| r.to_string()).collect(),
            }),
            K::TransferObjects { objects, recipient } => Kind::TransferObjects(TransferObjects {
                objects: objects
                    .into_iter()
                    .map(|o| object_ref_to_proto(o))
                    .collect(),
                recipient: Some(recipient.to_string()),
            }),
            K::AddStake {
                address,
                coin_ref,
                amount,
            } => Kind::AddStake(AddStake {
                address: Some(address.to_string()),
                coin_ref: Some(object_ref_to_proto(coin_ref)),
                amount: amount,
            }),
            K::AddStakeToEncoder {
                encoder_address,
                coin_ref,
                amount,
            } => Kind::AddStakeToEncoder(AddStakeToEncoder {
                encoder_address: Some(encoder_address.to_string()),
                coin_ref: Some(object_ref_to_proto(coin_ref)),
                amount: amount,
            }),
            K::WithdrawStake { staked_soma } => Kind::WithdrawStake(WithdrawStake {
                staked_soma: Some(object_ref_to_proto(staked_soma)),
            }),
            K::EmbedData {
                digest,
                data_size_bytes,
                coin_ref,
            } => Kind::EmbedData(EmbedData {
                digest: Some(digest.to_string()),
                data_size_bytes: Some(data_size_bytes as u32),
                coin_ref: Some(object_ref_to_proto(coin_ref)),
            }),
            K::ClaimEscrow { shard_input_ref } => Kind::ClaimEscrow(ClaimEscrow {
                shard_input_ref: Some(object_ref_to_proto(shard_input_ref)),
            }),
            K::ReportScores {
                shard_input_ref,
                scores,
                signature,
                signers,
            } => Kind::ReportScores(ReportScores {
                shard_input_ref: Some(object_ref_to_proto(shard_input_ref)),
                scores: Some(Bcs {
                    name: Some("ShardScores".to_string()),
                    value: Some(scores.into()),
                }),
                encoder_aggregate_signature: Some(Bcs {
                    name: Some("EncoderAggregateSignature".to_string()),
                    value: Some(signature.into()),
                }),
                signers: signers
                    .into_iter()
                    .map(|s| {
                        // Assuming EncoderPublicKey has a to_bytes() or similar method
                        format!("0x{}", hex::encode(s.to_bytes()))
                    })
                    .collect(),
            }),
        };

        let mut message = Self::default();
        message.kind = Some(kind);
        message
    }
}

// Implement conversions for Args types
impl From<types::transaction::AddValidatorArgs> for AddValidator {
    fn from(args: types::transaction::AddValidatorArgs) -> Self {
        Self {
            pubkey_bytes: Some(Bcs {
                name: Some("PublicKey".to_string()),
                value: Some(args.pubkey_bytes.into()),
            }),
            network_pubkey_bytes: Some(Bcs {
                name: Some("NetworkPublicKey".to_string()),
                value: Some(args.network_pubkey_bytes.into()),
            }),
            worker_pubkey_bytes: Some(Bcs {
                name: Some("WorkerPublicKey".to_string()),
                value: Some(args.worker_pubkey_bytes.into()),
            }),
            net_address: Some(Bcs {
                name: Some("NetworkAddress".to_string()),
                value: Some(args.net_address.into()),
            }),
            p2p_address: Some(Bcs {
                name: Some("P2PAddress".to_string()),
                value: Some(args.p2p_address.into()),
            }),
            primary_address: Some(Bcs {
                name: Some("PrimaryAddress".to_string()),
                value: Some(args.primary_address.into()),
            }),
            encoder_validator_address: Some(Bcs {
                name: Some("EncoderValidatorAddress".to_string()),
                value: Some(args.encoder_validator_address.into()),
            }),
        }
    }
}

impl From<types::transaction::RemoveValidatorArgs> for RemoveValidator {
    fn from(args: types::transaction::RemoveValidatorArgs) -> Self {
        Self {
            pubkey_bytes: Some(Bcs {
                name: Some("PublicKey".to_string()),
                value: Some(args.pubkey_bytes.into()),
            }),
        }
    }
}

impl From<types::transaction::UpdateValidatorMetadataArgs> for UpdateValidatorMetadata {
    fn from(args: types::transaction::UpdateValidatorMetadataArgs) -> Self {
        Self {
            next_epoch_network_address: args.next_epoch_network_address.map(|bytes| Bcs {
                name: Some("NetworkAddress".to_string()),
                value: Some(bytes.into()),
            }),
            next_epoch_p2p_address: args.next_epoch_p2p_address.map(|bytes| Bcs {
                name: Some("P2PAddress".to_string()),
                value: Some(bytes.into()),
            }),
            next_epoch_primary_address: args.next_epoch_primary_address.map(|bytes| Bcs {
                name: Some("PrimaryAddress".to_string()),
                value: Some(bytes.into()),
            }),
            next_epoch_protocol_pubkey: args.next_epoch_protocol_pubkey.map(|bytes| Bcs {
                name: Some("ProtocolPublicKey".to_string()),
                value: Some(bytes.into()),
            }),
            next_epoch_worker_pubkey: args.next_epoch_worker_pubkey.map(|bytes| Bcs {
                name: Some("WorkerPublicKey".to_string()),
                value: Some(bytes.into()),
            }),
            next_epoch_network_pubkey: args.next_epoch_network_pubkey.map(|bytes| Bcs {
                name: Some("NetworkPublicKey".to_string()),
                value: Some(bytes.into()),
            }),
        }
    }
}

impl From<types::transaction::AddEncoderArgs> for AddEncoder {
    fn from(args: types::transaction::AddEncoderArgs) -> Self {
        Self {
            encoder_pubkey_bytes: Some(Bcs {
                name: Some("EncoderPublicKey".to_string()),
                value: Some(args.encoder_pubkey_bytes.into()),
            }),
            network_pubkey_bytes: Some(Bcs {
                name: Some("NetworkPublicKey".to_string()),
                value: Some(args.network_pubkey_bytes.into()),
            }),
            internal_network_address: Some(Bcs {
                name: Some("InternalNetworkAddress".to_string()),
                value: Some(args.internal_network_address.into()),
            }),
            external_network_address: Some(Bcs {
                name: Some("ExternalNetworkAddress".to_string()),
                value: Some(args.external_network_address.into()),
            }),
            object_server_address: Some(Bcs {
                name: Some("ObjectServerAddress".to_string()),
                value: Some(args.object_server_address.into()),
            }),
        }
    }
}

impl From<types::transaction::UpdateEncoderMetadataArgs> for UpdateEncoderMetadata {
    fn from(args: types::transaction::UpdateEncoderMetadataArgs) -> Self {
        Self {
            next_epoch_external_network_address: args.next_epoch_external_network_address.map(
                |bytes| Bcs {
                    name: Some("ExternalNetworkAddress".to_string()),
                    value: Some(bytes.into()),
                },
            ),
            next_epoch_internal_network_address: args.next_epoch_internal_network_address.map(
                |bytes| Bcs {
                    name: Some("InternalNetworkAddress".to_string()),
                    value: Some(bytes.into()),
                },
            ),
            next_epoch_network_pubkey: args.next_epoch_network_pubkey.map(|bytes| Bcs {
                name: Some("NetworkPublicKey".to_string()),
                value: Some(bytes.into()),
            }),
            next_epoch_object_server_address: args.next_epoch_object_server_address.map(|bytes| {
                Bcs {
                    name: Some("ObjectServerAddress".to_string()),
                    value: Some(bytes.into()),
                }
            }),
        }
    }
}

//
// ConsensusCommitPrologue
//

impl From<types::consensus::ConsensusCommitPrologue> for ConsensusCommitPrologue {
    fn from(value: types::consensus::ConsensusCommitPrologue) -> Self {
        let mut message = Self::default();
        message.epoch = Some(value.epoch);
        message.round = Some(value.round);
        message.commit_timestamp = Some(crate::proto::timestamp_ms_to_proto(
            value.commit_timestamp_ms,
        ));
        message
    }
}

//
// GenesisTransaction
//

impl From<types::transaction::GenesisTransaction> for GenesisTransaction {
    fn from(value: types::transaction::GenesisTransaction) -> Self {
        let mut message = Self::default();
        message.objects = value.objects.into_iter().map(Into::into).collect();
        message
    }
}

//
// ChangeEpoch
//

impl From<types::transaction::ChangeEpoch> for ChangeEpoch {
    fn from(value: types::transaction::ChangeEpoch) -> Self {
        let mut message = Self::default();
        message.epoch = Some(value.epoch);
        message.epoch_start_timestamp = Some(crate::proto::timestamp_ms_to_proto(
            value.epoch_start_timestamp_ms,
        ));
        message
    }
}

//
// TransactionEffects
//

impl From<types::effects::TransactionEffects> for TransactionEffects {
    fn from(value: types::effects::TransactionEffects) -> Self {
        Self::merge_from(&value, &FieldMaskTree::new_wildcard())
    }
}
impl Merge<&types::effects::TransactionEffects> for TransactionEffects {
    fn merge(&mut self, source: &types::effects::TransactionEffects, mask: &FieldMaskTree) {
        if mask.contains(Self::STATUS_FIELD.name) {
            self.status = Some(source.status.clone().into());
        }

        if mask.contains(Self::EPOCH_FIELD.name) {
            self.epoch = Some(source.executed_epoch);
        }

        if mask.contains(Self::FEE_FIELD.name) {
            // Handle Option<TransactionFee>
            if let Some(ref fee) = source.transaction_fee {
                self.fee = Some(fee.clone().into());
            }
        }

        if mask.contains(Self::TRANSACTION_DIGEST_FIELD.name) {
            self.transaction_digest = Some(source.transaction_digest.to_string());
        }

        // Extract gas object from transaction_fee if present
        if mask.contains(Self::GAS_OBJECT_FIELD.name) {
            if let Some(ref fee) = source.transaction_fee {
                // Find the gas object in changed_objects using the gas_object_ref from fee
                let gas_object_id = fee.gas_object_ref.0;
                self.gas_object = source
                    .changed_objects
                    .iter()
                    .find(|(id, _)| *id == gas_object_id)
                    .map(|(id, change)| {
                        let mut message = ChangedObject::from(change.clone());
                        message.object_id = Some(id.to_string());
                        message
                    });
            }
        }

        if mask.contains(Self::DEPENDENCIES_FIELD.name) {
            self.dependencies = source
                .dependencies
                .iter()
                .map(ToString::to_string)
                .collect();
        }

        if mask.contains(Self::LAMPORT_VERSION_FIELD.name) {
            self.lamport_version = Some(source.version.value());
        }

        if mask.contains(Self::CHANGED_OBJECTS_FIELD.name) {
            self.changed_objects = source
                .changed_objects
                .iter()
                .map(|(id, change)| {
                    let mut message = ChangedObject::from(change.clone());
                    message.object_id = Some(id.to_string());
                    message
                })
                .collect();
        }

        // Set version for all objects that have output_digest but no output_version
        for object in self.changed_objects.iter_mut().chain(&mut self.gas_object) {
            if object.output_digest.is_some() && object.output_version.is_none() {
                object.output_version = Some(source.version.value());
            }
        }

        if mask.contains(Self::UNCHANGED_SHARED_OBJECTS_FIELD.name) {
            self.unchanged_shared_objects = source
                .unchanged_shared_objects
                .iter()
                .map(|(id, unchanged)| {
                    let mut message = UnchangedSharedObject::from(unchanged.clone());
                    message.object_id = Some(id.to_string());
                    message
                })
                .collect();
        }
    }
}

//
// ChangedObject
//

impl From<types::effects::object_change::EffectsObjectChange> for ChangedObject {
    fn from(value: types::effects::object_change::EffectsObjectChange) -> Self {
        use changed_object::InputObjectState;
        use changed_object::OutputObjectState;
        use types::effects::object_change::ObjectIn;
        use types::effects::object_change::ObjectOut;

        let mut message = Self::default();

        // Input State
        let input_state = match value.input_state {
            ObjectIn::NotExist => InputObjectState::DoesNotExist,
            ObjectIn::Exist(((version, digest), owner)) => {
                message.input_version = Some(version.value());
                message.input_digest = Some(digest.to_string());
                message.input_owner = Some(owner.into());
                InputObjectState::Exists
            }
        };
        message.set_input_state(input_state);

        // Output State
        let output_state = match value.output_state {
            ObjectOut::NotExist => OutputObjectState::DoesNotExist,
            ObjectOut::ObjectWrite((digest, owner)) => {
                message.output_digest = Some(digest.to_string());
                message.output_owner = Some(owner.into());
                OutputObjectState::ObjectWrite
            }
        };
        message.set_output_state(output_state);

        message.set_id_operation(value.id_operation.into());
        message
    }
}

//
// IdOperation
//

impl From<types::effects::object_change::IDOperation> for changed_object::IdOperation {
    fn from(value: types::effects::object_change::IDOperation) -> Self {
        use types::effects::object_change::IDOperation as I;

        match value {
            I::None => Self::None,
            I::Created => Self::Created,
            I::Deleted => Self::Deleted,
        }
    }
}

//
// UnchangedSharedObject
//

impl From<types::effects::UnchangedSharedKind> for UnchangedSharedObject {
    fn from(value: types::effects::UnchangedSharedKind) -> Self {
        use types::effects::UnchangedSharedKind as K;
        use unchanged_shared_object::UnchangedSharedObjectKind;

        let mut message = Self::default();

        let kind = match value {
            K::ReadOnlyRoot((version, digest)) => {
                message.version = Some(version.value());
                message.digest = Some(digest.to_string());
                UnchangedSharedObjectKind::ReadOnlyRoot
            }
            K::MutateDeleted(version) => {
                message.version = Some(version.value());
                UnchangedSharedObjectKind::MutatedDeleted
            }
            K::ReadDeleted(version) => {
                message.version = Some(version.value());
                UnchangedSharedObjectKind::ReadDeleted
            }
            K::Cancelled(version) => {
                message.version = Some(version.value());
                UnchangedSharedObjectKind::Canceled
            }
        };

        message.set_kind(kind);
        message
    }
}
