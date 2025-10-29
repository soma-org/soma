/// Indicates a byte offset withing a resource.
pub const UPLOAD_OFFSET: &'static str = "upload-offset";

/// Indicates the size of the entire upload in bytes.
pub const UPLOAD_LENGTH: &'static str = "upload-length";

/// A comma-separated list of protocol versions supported by the server.
pub const TUS_VERSION: &'static str = "tus-version";

/// The version of the protocol used by the client or the server.
pub const TUS_RESUMABLE: &'static str = "tus-resumable";

/// A comma-separated list of the extensions supported by the server.
pub const TUS_EXTENSION: &'static str = "tus-extension";

/// Integer indicating the maximum allowed size of an entire upload in bytes.
pub const TUS_MAX_SIZE: &'static str = "tus-max-size";

/// Use this header if its environment does not support the PATCH or DELETE methods.
pub const X_HTTP_METHOD_OVERRIDE: &'static str = "x-http-method-override";

/// Use this header if its environment does not support the PATCH or DELETE methods.
pub const CONTENT_TYPE: &'static str = "content-type";

/// Use this header if its environment does not support the PATCH or DELETE methods.
//pub const UPLOAD_DEFER_LENGTH: &'static str = "upload-defer-length";

/// Use this header if its environment does not support the PATCH or DELETE methods.
pub const UPLOAD_METADATA: &'static str = "upload-metadata";

/// Use this header if its environment does not support the PATCH or DELETE methods.
pub const LOCATION: &'static str = "location";
