#[cfg(test)]
pub mod generator {
    use arrgen::generate_array_rust;

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_generate_weights() {
            let shape = vec![2, 2];
            let seed = 42;
            let min = 0.0;
            let max = 1.0;
            let weights = generate_array_rust(shape, seed, min, max);
            println!("{:?}", weights);
        }
    }
}
