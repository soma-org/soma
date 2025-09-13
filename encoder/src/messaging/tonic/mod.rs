#![doc = include_str!("README.md")]

mod generated {
    include!(concat!(
        env!("OUT_DIR"),
        "/soma.EncoderInternalTonicService.rs"
    ));
}

pub mod external;
pub mod internal;
