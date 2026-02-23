// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use rand::prelude::Distribution as _;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ClientIdSource {
    #[default]
    SocketAddr,
    XForwardedFor(usize),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Weight(f32);

impl Weight {
    pub fn new(value: f32) -> Result<Self, &'static str> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err("Weight must be between 0.0 and 1.0")
        }
    }

    pub fn one() -> Self {
        Self(1.0)
    }

    pub fn zero() -> Self {
        Self(0.0)
    }

    pub fn value(&self) -> f32 {
        self.0
    }

    pub fn is_sampled(&self) -> bool {
        let mut rng = rand::thread_rng();
        let sample = rand::distributions::Uniform::new(0.0, 1.0).sample(&mut rng);
        sample <= self.value()
    }
}

fn validate_sample_rate<'de, D>(deserializer: D) -> Result<Weight, D::Error>
where
    D: Deserializer<'de>,
{
    let value = f32::deserialize(deserializer)?;
    Weight::new(value)
        .map_err(|_| serde::de::Error::custom("spam-sample-rate must be between 0.0 and 1.0"))
}

impl PartialEq for Weight {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}
