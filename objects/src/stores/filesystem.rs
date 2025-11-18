// #[async_trait]
// impl SafetensorBuffer for LocalFileSystem {
//     type Buffer = Mmap;
//     async fn safetensor_buffer(&self, path: ObjectPath) -> EvaluationResult<Self::Buffer> {
//         let fs_path = self
//             .path_to_filesystem(&path.path())
//             .map_err(|e| EvaluationError::StorageFailure(e.to_string()))?;
//         let file =
//             File::open(fs_path).map_err(|e| EvaluationError::StorageFailure(e.to_string()))?;
//         let mmap = unsafe { MmapOptions::new().map(&file) }
//             .map_err(|e| EvaluationError::StorageFailure(e.to_string()))?;
//         Ok(mmap)
//     }
// }
