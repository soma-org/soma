use std::marker::PhantomData;

use burn::{
    record::{BytesRecorder, PrecisionSettings, Recorder, RecorderError},
    tensor::backend::Backend,
};
use serde::{de::DeserializeOwned, Serialize};
use zstd::DEFAULT_COMPRESSION_LEVEL;

#[derive(Debug, Default, Clone)]
pub struct BcsRecorder<S: PrecisionSettings> {
    _settings: core::marker::PhantomData<S>,
}

impl<S: PrecisionSettings> BcsRecorder<S> {
    pub fn new() -> Self {
        Self {
            _settings: PhantomData,
        }
    }
}

impl<S: PrecisionSettings, B: Backend> BytesRecorder<B, Vec<u8>> for BcsRecorder<S> {}

impl<S: PrecisionSettings, B: Backend> Recorder<B> for BcsRecorder<S> {
    type Settings = S;
    type RecordArgs = ();
    type RecordOutput = Vec<u8>;
    type LoadArgs = Vec<u8>;

    fn save_item<I: Serialize>(
        &self,
        item: I,
        _args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError> {
        let bytes = bcs::to_bytes(&item).map_err(|e| RecorderError::Unknown(e.to_string()))?;
        zstd::encode_all(bytes.as_slice(), DEFAULT_COMPRESSION_LEVEL)
            .map_err(|e| RecorderError::Unknown(e.to_string()))
    }

    fn load_item<I: DeserializeOwned>(
        &self,
        args: &mut Self::LoadArgs,
    ) -> Result<I, RecorderError> {
        let bytes =
            zstd::decode_all(args.as_slice()).map_err(|e| RecorderError::Unknown(e.to_string()))?;
        bcs::from_bytes(&bytes).map_err(|e| RecorderError::Unknown(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        module::Module,
        nn,
        record::{FullPrecisionSettings, HalfPrecisionSettings},
    };
    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn test_can_save_and_load_bcs_format() {
        test_can_save_and_load(BcsRecorder::<FullPrecisionSettings>::default());
        test_can_save_and_load(BcsRecorder::<HalfPrecisionSettings>::default());
    }

    fn test_can_save_and_load<Recorder>(recorder: Recorder)
    where
        Recorder: BytesRecorder<TestBackend, Vec<u8>>,
    {
        let device = Default::default();
        let model1 = create_model::<TestBackend>(&device);
        let model2 = create_model::<TestBackend>(&device);
        let bytes1 = recorder.record(model1.into_record(), ()).unwrap();
        let bytes2 = recorder.record(model2.clone().into_record(), ()).unwrap();

        let model2_after = model2.load_record(recorder.load(bytes1.clone(), &device).unwrap());
        let bytes2_after = recorder.record(model2_after.into_record(), ()).unwrap();

        assert_ne!(bytes1, bytes2);
        assert_eq!(bytes1, bytes2_after);
    }

    pub fn create_model<B: Backend>(device: &B::Device) -> nn::Linear<B> {
        nn::LinearConfig::new(32, 32).with_bias(true).init(device)
    }
}
