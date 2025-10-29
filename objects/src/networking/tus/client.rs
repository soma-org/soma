use axum::http::{HeaderMap, HeaderValue};
use bytes::Bytes;
use reqwest::{Client, Method, StatusCode};
use std::io::SeekFrom;
use std::str::FromStr;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt, BufReader};
use types::error::{ObjectError, ObjectResult};
use url::Url;
use uuid::Uuid;

use crate::networking::tus::headers::{
    CONTENT_TYPE, LOCATION, TUS_EXTENSION, TUS_MAX_SIZE, TUS_RESUMABLE, TUS_VERSION, UPLOAD_LENGTH,
    UPLOAD_OFFSET,
};
use crate::networking::tus::SOMA_SUPPORTED_VERSION;

const DEFAULT_CHUNK_SIZE: usize = 5 * 1024 * 1024;

pub struct TusClient {
    base_url: Url,
    client: Client,
    chunk_size: usize,
}

impl TusClient {
    pub fn new(base_url: Url, chunk_size: Option<usize>) -> Self {
        Self {
            base_url,
            client: Client::new(),
            chunk_size: chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE),
        }
    }

    pub async fn get_info(&self, uuid: Uuid) -> ObjectResult<UploadInfo> {
        let url = self
            .base_url
            .join(&format!("/uploads/{}", uuid))
            .map_err(|e| ObjectError::ParseError(e.to_string()))?;

        let resp = self
            .client
            .head(url.clone())
            .headers(default_headers())
            .send()
            .await
            .map_err(|e| ObjectError::ReqwestError(e.to_string()))?;

        if resp.status().is_client_error() {
            // TODO: return a more clear error
            return Err(ObjectError::NotFound(url.to_string()));
        }

        let headers = resp.headers();
        let offset: u64 = headers.required_typed(UPLOAD_OFFSET)?;
        let total_size: u64 = headers.required_typed(UPLOAD_LENGTH)?;

        Ok(UploadInfo {
            bytes_uploaded: offset,
            total_size,
        })
    }

    pub async fn create(&self, size: u64) -> ObjectResult<String> {
        let url = self
            .base_url
            .join("/uploads")
            .map_err(|e| ObjectError::ParseError(e.to_string()))?;

        let mut headers = default_headers();
        headers.insert(
            UPLOAD_LENGTH,
            HeaderValue::from_str(&size.to_string()).expect("file length is valid header value"),
        );

        let resp = self
            .client
            .post(url)
            .headers(headers)
            .send()
            .await
            .map_err(|e| ObjectError::ReqwestError(e.to_string()))?;

        match resp.status() {
            StatusCode::CREATED => {
                let headers = resp.headers();
                let location: String = headers.required_typed(LOCATION)?;
                Ok(location)
            }
            StatusCode::PAYLOAD_TOO_LARGE => Err(ObjectError::FileTooLarge),
            s => Err(ObjectError::UnexpectedStatusCode(s.as_u16())),
        }
    }

    pub async fn upload<R>(&self, uuid: Uuid, reader: R) -> ObjectResult<()>
    where
        R: AsyncRead + AsyncSeek + Unpin + Send,
    {
        let info = self.get_info(uuid).await?;

        let mut offset = info.bytes_uploaded;

        if offset >= info.total_size {
            return Ok(());
        }
        let url = self
            .base_url
            .join(&format!("/uploads/{}", uuid))
            .map_err(|e| ObjectError::ParseError(e.to_string()))?;

        let mut buffered_reader = BufReader::with_capacity(self.chunk_size, reader);
        buffered_reader
            .seek(SeekFrom::Start(offset))
            .await
            .map_err(|e| ObjectError::ReadError(e.to_string()))?;

        let mut buf = vec![0u8; self.chunk_size];

        loop {
            let read = buffered_reader
                .read(&mut buf)
                .await
                .map_err(|e| ObjectError::ReadError(e.to_string()))?;
            if read == 0 {
                if offset < info.total_size {
                    return Err(ObjectError::ParseError(
                        "Unexpected EOF before file length".to_string(),
                    )); // Or custom IncompleteUpload variant
                }
                break;
            }
            let chunk = &buf[..read];

            let headers = upload_headers(offset)?;
            let resp = self
                .client
                .patch(url.clone())
                .headers(headers)
                .body(Bytes::copy_from_slice(chunk))
                .send()
                .await
                .map_err(|e| ObjectError::ReqwestError(e.to_string()))?;

            match resp.status() {
                StatusCode::NO_CONTENT => {}
                StatusCode::CONFLICT => {
                    return Err(ObjectError::ParseError("Wrong upload offset".to_string()))
                } // Or custom WrongUploadOffsetError
                StatusCode::NOT_FOUND => return Err(ObjectError::NotFound(url.to_string())),
                s => return Err(ObjectError::UnexpectedStatusCode(s.as_u16())),
            }

            offset = resp.headers().required_typed(UPLOAD_OFFSET)?;
            if offset >= info.total_size {
                break;
            }
        }

        Ok(())
    }

    pub async fn get_server_info(&self) -> ObjectResult<ServerInfo> {
        let url = self
            .base_url
            .join("/uploads")
            .map_err(|e| ObjectError::ParseError(e.to_string()))?;
        let resp = self
            .client
            .request(Method::OPTIONS, url)
            .send()
            .await
            .map_err(|e| ObjectError::ReqwestError(e.to_string()))?;

        let status = resp.status();
        if !matches!(status, StatusCode::OK | StatusCode::NO_CONTENT) {
            return Err(ObjectError::UnexpectedStatusCode(status.as_u16()));
        }

        let headers = resp.headers();

        let supported_versions = headers
            .typed::<String>(TUS_VERSION)?
            .map(|s| s.split(',').map(|v| v.trim().to_owned()).collect())
            .unwrap_or_default();

        let extensions = headers
            .typed::<String>(TUS_EXTENSION)?
            .map(|s| {
                s.split(',')
                    .filter_map(|e| TusExtension::from_str(e.trim()).ok())
                    .collect()
            })
            .unwrap_or_default();

        let max_upload_size = headers.typed(TUS_MAX_SIZE)?;

        Ok(ServerInfo {
            supported_versions,
            extensions,
            max_upload_size,
        })
    }
}

/// Describes a file on the server.
#[derive(Debug)]
pub struct UploadInfo {
    /// How many bytes have been uploaded.
    pub bytes_uploaded: u64,
    /// The total size of the file.
    pub total_size: u64,
}

/// Describes the tus enabled server.
#[derive(Debug)]
pub struct ServerInfo {
    /// The different versions of the tus protocol supported by the server, ordered by preference.
    pub supported_versions: Vec<String>,
    /// The extensions to the protocol supported by the server.
    pub extensions: Vec<TusExtension>,
    /// The maximum supported total size of a file.
    pub max_upload_size: Option<u64>,
}

/// Enumerates the extensions to the tus protocol.
#[derive(Debug, PartialEq)]
pub enum TusExtension {
    /// The server supports creating files.
    Creation,
    //// The server supports setting expiration time on files and uploads.
    Expiration,
    /// The server supports verifying checksums of uploaded chunks.
    Checksum,
    /// The server supports deleting files.
    Termination,
    /// The server supports parallel uploads of a single file.
    Concatenation,
}

impl FromStr for TusExtension {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "creation" => Ok(TusExtension::Creation),
            "expiration" => Ok(TusExtension::Expiration),
            "checksum" => Ok(TusExtension::Checksum),
            "termination" => Ok(TusExtension::Termination),
            "concatenation" => Ok(TusExtension::Concatenation),
            _ => Err(()),
        }
    }
}

fn default_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        TUS_RESUMABLE,
        HeaderValue::from_static(SOMA_SUPPORTED_VERSION),
    );
    headers
}

fn upload_headers(progress: u64) -> ObjectResult<HeaderMap> {
    let mut headers = default_headers();
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/offset+octet-stream"),
    );
    headers.insert(
        UPLOAD_OFFSET,
        HeaderValue::from_str(&progress.to_string())
            .map_err(|e| ObjectError::ParseError(e.to_string()))?,
    );
    Ok(headers)
}

trait HeaderMapExtTyped {
    fn typed<T>(&self, name: &'static str) -> ObjectResult<Option<T>>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display;
    fn required_typed<T>(&self, name: &'static str) -> ObjectResult<T>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display;
}

impl HeaderMapExtTyped for reqwest::header::HeaderMap {
    fn typed<T>(&self, name: &'static str) -> ObjectResult<Option<T>>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        match self.get(name) {
            Some(value) => {
                let s = value
                    .to_str()
                    .map_err(|e| ObjectError::ParseError(e.to_string()))?;
                let parsed = s
                    .parse::<T>()
                    .map_err(|e| ObjectError::ParseError(e.to_string()))?;
                Ok(Some(parsed))
            }
            None => Ok(None),
        }
    }

    fn required_typed<T>(&self, name: &'static str) -> ObjectResult<T>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        self.typed(name)?
            .ok_or_else(|| ObjectError::MissingHeader(name.to_owned()))
    }
}
