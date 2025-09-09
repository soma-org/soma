use super::*;
use crate::proto::TryFromProtoError;
use crate::utils::merge::Merge;

impl From<crate::types::BalanceChange> for BalanceChange {
    fn from(value: crate::types::BalanceChange) -> Self {
        Self {
            address: Some(value.address.to_string()),
            coin_type: Some(value.coin_type.to_string()),
            amount: Some(value.amount.to_string()),
        }
    }
}

impl Merge<&crate::types::BalanceChange> for BalanceChange {
    fn merge(
        &mut self,
        source: &crate::types::BalanceChange,
        mask: &crate::utils::field::FieldMaskTree,
    ) {
        if mask.contains(Self::ADDRESS_FIELD) {
            self.address = Some(source.address.to_string());
        }
        if mask.contains(Self::COIN_TYPE_FIELD) {
            self.coin_type = Some(source.coin_type.to_string());
        }
        if mask.contains(Self::AMOUNT_FIELD) {
            self.amount = Some(source.amount.to_string());
        }
    }
}

impl TryFrom<&BalanceChange> for crate::types::BalanceChange {
    type Error = TryFromProtoError;

    fn try_from(value: &BalanceChange) -> Result<Self, Self::Error> {
        Ok(Self {
            address: value
                .address()
                .parse()
                .map_err(|e| TryFromProtoError::invalid(BalanceChange::ADDRESS_FIELD, e))?,
            coin_type: value
                .coin_type()
                .parse()
                .map_err(|e| TryFromProtoError::invalid(BalanceChange::COIN_TYPE_FIELD, e))?,
            amount: value
                .amount()
                .parse()
                .map_err(|e| TryFromProtoError::invalid(BalanceChange::AMOUNT_FIELD, e))?,
        })
    }
}
