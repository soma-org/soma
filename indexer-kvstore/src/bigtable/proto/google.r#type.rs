// Proto types from google.type package (minimal subset needed by BigTable)

/// Represents a whole or partial calendar date, such as a birthday.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Date {
    /// Year of the date. Must be from 1 to 9999, or 0 to specify a date without
    /// a year.
    #[prost(int32, tag = "1")]
    pub year: i32,
    /// Month of a year. Must be from 1 to 12, or 0 to specify a year without a
    /// month and day.
    #[prost(int32, tag = "2")]
    pub month: i32,
    /// Day of a month. Must be from 1 to 31 and valid for the year and month, or 0
    /// to specify a year by itself or a year and month where the day isn't
    /// significant.
    #[prost(int32, tag = "3")]
    pub day: i32,
}
