#[cfg(test)]
pub mod generator {
    use arrgen::{constant_array, normal_array, uniform_array};

    #[cfg(test)]
    mod tests {
        use ndarray_safetensors::TensorViewWithDataBuffer;
        use safetensors::serialize;
        use std::collections::HashMap;

        use super::*;

        #[test]
        fn test_safetensor_generation() {
            let seed = 42u64;
            let shape = vec![2, 2];
            let mut tensors: HashMap<String, TensorViewWithDataBuffer> = HashMap::new();
            tensors.insert(
                "uniform_tensor".to_string(),
                TensorViewWithDataBuffer::new(&uniform_array(seed, shape.clone(), 0.0, 1.0)),
            );
            tensors.insert(
                "normal_tensor".to_string(),
                TensorViewWithDataBuffer::new(&normal_array(seed, shape.clone(), 0.0, 1.0)),
            );
            tensors.insert(
                "constant_tensor".to_string(),
                TensorViewWithDataBuffer::new(&constant_array(shape, 0.0)),
            );
            let st = serialize(tensors, &None).unwrap();
            println!("{:?}", st);
        }

        #[test]
        fn test_generate_weights() {
            let shape = vec![2, 2];
            let seed = 42;
            let min = 0.0;
            let max = 1.0;
            let weights = uniform_array(seed, shape, min, max);
            println!("{:?}", weights);
        }
    }
}
