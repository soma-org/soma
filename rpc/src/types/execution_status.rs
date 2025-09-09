use serde::{Deserialize, Serialize};

#[derive(Eq, PartialEq, Clone, Debug, Deserialize, Serialize)]
pub enum ExecutionStatus {
    /// The Transaction successfully executed.
    Success,

    /// The Transaction didn't execute successfully.
    ///
    /// Failed transactions are still committed to the blockchain but any intended effects are
    /// rolled back to prior to this transaction executing with the caveat that gas objects are
    /// still smashed and gas usage is still charged.
    Failure {
        /// The error encountered during execution.
        error: ExecutionError,
        /// The command, if any, during which the error occurred.
        command: Option<u64>,
    },
}

#[derive(Eq, PartialEq, Clone, Debug, Deserialize, Serialize)]
#[non_exhaustive]
pub enum ExecutionError {
    //
    // General transaction errors
    //
    /// Insufficient Gas
    InsufficientGas,
    /// Invalid Gas Object.
    InvalidGasObject,
    /// Invariant Violation
    InvariantViolation,
    /// Attempted to used feature that is not supported yet
    FeatureNotYetSupported,
    /// Move object is larger than the maximum allowed size
    ObjectTooBig {
        object_size: u64,
        max_object_size: u64,
    },
    /// Package is larger than the maximum allowed size
    PackageTooBig {
        object_size: u64,
        max_object_size: u64,
    },
    /// Circular Object Ownership
    CircularObjectOwnership { object: Address },

    //
    // Coin errors
    //
    /// Insufficient coin balance for requested operation
    InsufficientCoinBalance,
    /// Coin balance overflowed an u64
    CoinBalanceOverflow,

    //
    // Publish/Upgrade errors
    //
    /// Publish Error, Non-zero Address.
    /// The modules in the package must have their self-addresses set to zero.
    PublishErrorNonZeroAddress,

    /// Sui Move Bytecode Verification Error.
    SuiMoveVerificationError,

    //
    // MoveVm Errors
    //
    /// Error from a non-abort instruction.
    /// Possible causes:
    ///     Arithmetic error, stack overflow, max value depth, etc."
    MovePrimitiveRuntimeError { location: Option<MoveLocation> },
    /// Move runtime abort
    MoveAbort { location: MoveLocation, code: u64 },
    /// Bytecode verification error.
    VmVerificationOrDeserializationError,
    /// MoveVm invariant violation
    VmInvariantViolation,

    //
    // Programmable Transaction Errors
    //
    /// Function not found
    FunctionNotFound,
    /// Arity mismatch for Move function.
    /// The number of arguments does not match the number of parameters
    ArityMismatch,
    /// Type arity mismatch for Move function.
    /// Mismatch between the number of actual versus expected type arguments.
    TypeArityMismatch,
    /// Non Entry Function Invoked. Move Call must start with an entry function.
    NonEntryFunctionInvoked,
    /// Invalid command argument
    CommandArgumentError {
        argument: u16,
        kind: CommandArgumentError,
    },
    /// Type argument error
    TypeArgumentError {
        /// Index of the problematic type argument
        type_argument: u16,
        kind: TypeArgumentError,
    },
    /// Unused result without the drop ability.
    UnusedValueWithoutDrop { result: u16, subresult: u16 },
    /// Invalid public Move function signature.
    /// Unsupported return type for return value
    InvalidPublicFunctionReturnType { index: u16 },
    /// Invalid Transfer Object, object does not have public transfer.
    InvalidTransferObject,

    //
    // Post-execution errors
    //
    /// Effects from the transaction are too large
    EffectsTooLarge { current_size: u64, max_size: u64 },

    /// Publish or Upgrade is missing dependency
    PublishUpgradeMissingDependency,

    /// Publish or Upgrade dependency downgrade.
    ///
    /// Indirect (transitive) dependency of published or upgraded package has been assigned an
    /// on-chain version that is less than the version required by one of the package's
    /// transitive dependencies.
    PublishUpgradeDependencyDowngrade,

    /// Invalid package upgrade
    PackageUpgradeError { kind: PackageUpgradeError },

    /// Indicates the transaction tried to write objects too large to storage
    WrittenObjectsTooLarge {
        object_size: u64,
        max_object_size: u64,
    },

    /// Certificate is on the deny list
    CertificateDenied,

    /// Sui Move Bytecode verification timed out.
    SuiMoveVerificationTimedout,

    /// The requested consensus object operation is not allowed
    ConsensusObjectOperationNotAllowed,

    /// Requested consensus object has been deleted
    InputObjectDeleted,

    /// Certificate is canceled due to congestion on consensus objects
    ExecutionCanceledDueToConsensusObjectCongestion { congested_objects: Vec<Address> },

    /// Address is denied for this coin type
    AddressDeniedForCoin { address: Address, coin_type: String },

    /// Coin type is globally paused for use
    CoinTypeGlobalPause { coin_type: String },

    /// Certificate is canceled because randomness could not be generated this epoch
    ExecutionCanceledDueToRandomnessUnavailable,

    /// Move vector element (passed to MakeMoveVec) with size {value_size} is larger \
    /// than the maximum size {max_scaled_size}. Note that this maximum is scaled based on the \
    /// type of the vector element.
    MoveVectorElemTooBig {
        value_size: u64,
        max_scaled_size: u64,
    },

    /// Move value (possibly an upgrade ticket or a dev-inspect value) with size {value_size} \
    /// is larger than the maximum size  {max_scaled_size}. Note that this maximum is scaled based \
    /// on the type of the value.
    MoveRawValueTooBig {
        value_size: u64,
        max_scaled_size: u64,
    },

    /// A valid linkage was unable to be determined for the transaction or one of its commands.
    InvalidLinkage,
}

#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CommandArgumentError {
    /// The type of the value does not match the expected type
    TypeMismatch,

    /// The argument cannot be deserialized into a value of the specified type
    InvalidBcsBytes,

    /// The argument cannot be instantiated from raw bytes
    InvalidUsageOfPureArgument,

    /// Invalid argument to private entry function.
    /// Private entry functions cannot take arguments from other Move functions.
    InvalidArgumentToPrivateEntryFunction,

    /// Out of bounds access to input or results
    IndexOutOfBounds { index: u16 },

    /// Out of bounds access to subresult
    SecondaryIndexOutOfBounds { result: u16, subresult: u16 },

    /// Invalid usage of result.
    /// Expected a single result but found either no return value or multiple.
    InvalidResultArity { result: u16 },

    /// Invalid usage of Gas coin.
    /// The Gas coin can only be used by-value with a TransferObjects command.
    InvalidGasCoinUsage,

    /// Invalid usage of move value.
    //     Mutably borrowed values require unique usage.
    //     Immutably borrowed values cannot be taken or borrowed mutably.
    //     Taken values cannot be used again.
    InvalidValueUsage,

    /// Immutable objects cannot be passed by-value.
    InvalidObjectByValue,

    /// Immutable objects cannot be passed by mutable reference, &mut.
    InvalidObjectByMutRef,

    /// consensus object operations such a wrapping, freezing, or converting to owned are not
    /// allowed.
    ConsensusObjectOperationNotAllowed,

    /// Invalid argument arity. Expected a single argument but found a result that expanded to
    /// multiple arguments.
    InvalidArgumentArity,
}

/// An error with a upgrading a package
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// package-upgrade-error = unable-to-fetch-package /
///                         not-a-package           /
///                         incompatible-upgrade    /
///                         digest-does-not-match   /
///                         unknown-upgrade-policy  /
///                         package-id-does-not-match
///
/// unable-to-fetch-package     = %x00 address
/// not-a-package               = %x01 address
/// incompatible-upgrade        = %x02
/// digest-does-not-match       = %x03 digest
/// unknown-upgrade-policy      = %x04 u8
/// package-id-does-not-match   = %x05 address address
/// ```
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PackageUpgradeError {
    /// Unable to fetch package
    UnableToFetchPackage { package_id: Address },

    /// Object is not a package
    NotAPackage { object_id: Address },

    /// Package upgrade is incompatible with previous version
    IncompatibleUpgrade,

    /// Digest in upgrade ticket and computed digest differ
    DigestDoesNotMatch { digest: Digest },

    /// Upgrade policy is not valid
    UnknownUpgradePolicy { policy: u8 },

    /// PackageId does not matach PackageId in upgrade ticket
    PackageIdDoesNotMatch {
        package_id: Address,
        ticket_id: Address,
    },
}

/// An error with a type argument
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// type-argument-error = type-not-found / constraint-not-satisfied
/// type-not-found = %x00
/// constraint-not-satisfied = %x01
/// ```
#[derive(Eq, PartialEq, Clone, Copy, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TypeArgumentError {
    /// A type was not found in the module specified
    TypeNotFound,

    /// A type provided did not match the specified constraint
    ConstraintNotSatisfied,
}
