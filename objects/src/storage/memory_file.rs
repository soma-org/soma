use futures::{pin_mut, Future};
use std::io::{self, Cursor, SeekFrom};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncSeek, AsyncWrite, ReadBuf};
use tokio::sync::Mutex;

#[derive(Clone, Debug)]
pub struct MemoryFile(Arc<Mutex<Cursor<Vec<u8>>>>);

impl MemoryFile {
    pub fn new(initial_data: Vec<u8>) -> Self {
        Self(Arc::new(Mutex::new(Cursor::new(initial_data))))
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(Arc::new(Mutex::new(Cursor::new(Vec::with_capacity(
            capacity,
        )))))
    }
}

impl Default for MemoryFile {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl AsyncRead for MemoryFile {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let lock_fut = self.0.lock();
        pin_mut!(lock_fut); // Pin the future to the stack
        let mut guard = futures::ready!(lock_fut.poll(cx));
        Pin::new(&mut *guard).poll_read(cx, buf)
    }
}

impl AsyncWrite for MemoryFile {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let lock_fut = self.0.lock();
        pin_mut!(lock_fut);
        let mut guard = futures::ready!(lock_fut.poll(cx));
        Pin::new(&mut *guard).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let lock_fut = self.0.lock();
        pin_mut!(lock_fut);
        let mut guard = futures::ready!(lock_fut.poll(cx));
        Pin::new(&mut *guard).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let lock_fut = self.0.lock();
        pin_mut!(lock_fut);
        let mut guard = futures::ready!(lock_fut.poll(cx));
        Pin::new(&mut *guard).poll_shutdown(cx)
    }
}

impl AsyncSeek for MemoryFile {
    fn start_seek(self: Pin<&mut Self>, position: SeekFrom) -> io::Result<()> {
        let mut guard = self
            .0
            .try_lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Mutex is currently locked"))?;
        Pin::new(&mut *guard).start_seek(position)
    }
    fn poll_complete(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<u64>> {
        let lock_fut = self.0.lock();
        pin_mut!(lock_fut);
        let mut guard = futures::ready!(lock_fut.poll(cx));
        Pin::new(&mut *guard).poll_complete(cx)
    }
}

// Add this module to the end of your file
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

    /// Tests basic writing and reading functionality.
    #[tokio::test]
    async fn test_write_and_read() {
        let mut file = MemoryFile::default();

        // Write a string to the file
        let write_buf = b"hello, world!";
        let bytes_written = file.write_all(write_buf).await.unwrap();
        assert_eq!(bytes_written, ()); // write_all returns () on success

        // Before reading, we must seek back to the beginning
        let new_pos = file.seek(SeekFrom::Start(0)).await.unwrap();
        assert_eq!(new_pos, 0);

        // Read the contents back into a new buffer
        let mut read_buf = Vec::new();
        file.read_to_end(&mut read_buf).await.unwrap();

        // Check if the read data matches the written data
        assert_eq!(read_buf, write_buf);
    }

    /// Tests the seeking functionality from different origins.
    #[tokio::test]
    async fn test_seek() {
        let mut file = MemoryFile::new(b"abcdefgh".to_vec());

        // 1. Seek from the start
        file.seek(SeekFrom::Start(4)).await.unwrap();
        let mut buf = [0u8; 1];
        file.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"e"); // We should be at the 5th character 'e'

        // 2. Seek from the current position
        // Current position is 5 (after reading 'e')
        file.seek(SeekFrom::Current(-2)).await.unwrap(); // Move back 2 chars
        file.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"d"); // We should be at 'd'

        // 3. Seek from the end
        file.seek(SeekFrom::End(-1)).await.unwrap(); // Go to the last character
        file.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"h"); // We should be at 'h'
    }

    /// Tests that clones access the same underlying data buffer.
    #[tokio::test]
    async fn test_cloning_and_shared_access() {
        let mut file = MemoryFile::new(b"original".to_vec());
        let mut file_clone = file.clone();

        // Write to the original handle, overwriting the start
        file.seek(SeekFrom::Start(0)).await.unwrap();
        file.write_all(b"new-").await.unwrap();

        // Read from the clone; it should see the changes made by the original
        file_clone.seek(SeekFrom::Start(0)).await.unwrap();
        let mut buf = vec![0; 8];
        file_clone.read_exact(&mut buf).await.unwrap();

        // The data should be the modified data
        assert_eq!(&buf, b"new-inal");
    }
}
