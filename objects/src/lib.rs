// pub mod services;
pub mod downloader;
pub mod readers;
pub mod services;
pub mod stores;

/// Cloud providers typically require a minimum multipart part size except for the last part
pub const MIN_PART_SIZE: u64 = 5 * 1024 * 1024;
/// Cloud providers typically have a max multipart part size
pub const MAX_PART_SIZE: u64 = 5 * 1024 * 1024 * 1024;

pub const CERTIFICATE_NAME: &str = "soma-objects";
