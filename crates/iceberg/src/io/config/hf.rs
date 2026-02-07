// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! HuggingFace Hub storage configuration.
//!
//! This module provides configuration constants and types for HuggingFace Hub storage.

use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use super::StorageConfig;
use crate::Result;

/// HuggingFace repository type (`dataset`, `model`, `space`).
pub const HF_REPO_TYPE: &str = "hf.repo-type";
/// HuggingFace git revision/branch.
pub const HF_REVISION: &str = "hf.revision";
/// HuggingFace API token for private repos.
pub const HF_TOKEN: &str = "hf.token";
/// Custom HuggingFace Hub endpoint.
pub const HF_ENDPOINT: &str = "hf.endpoint";

/// HuggingFace Hub storage configuration.
///
/// This struct contains all the configuration options for accessing HuggingFace Hub storage.
/// Use the builder pattern via `HfConfig::builder()` to construct instances.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, TypedBuilder)]
pub struct HfConfig {
    /// Repository type (`dataset`, `model`, `space`). Defaults to `dataset`.
    #[builder(default, setter(strip_option, into))]
    pub repo_type: Option<String>,
    /// Git revision/branch. Defaults to `main`.
    #[builder(default, setter(strip_option, into))]
    pub revision: Option<String>,
    /// API token for private repository access.
    #[builder(default, setter(strip_option, into))]
    pub token: Option<String>,
    /// Custom HuggingFace Hub endpoint URL.
    #[builder(default, setter(strip_option, into))]
    pub endpoint: Option<String>,
}

impl TryFrom<&StorageConfig> for HfConfig {
    type Error = crate::Error;

    fn try_from(config: &StorageConfig) -> Result<Self> {
        let props = config.props();

        let mut cfg = HfConfig::default();
        if let Some(repo_type) = props.get(HF_REPO_TYPE) {
            cfg.repo_type = Some(repo_type.clone());
        }
        if let Some(revision) = props.get(HF_REVISION) {
            cfg.revision = Some(revision.clone());
        }
        if let Some(token) = props.get(HF_TOKEN) {
            cfg.token = Some(token.clone());
        }
        if let Some(endpoint) = props.get(HF_ENDPOINT) {
            cfg.endpoint = Some(endpoint.clone());
        }

        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_config_builder() {
        let config = HfConfig::builder()
            .repo_type("dataset")
            .revision("main")
            .token("hf_xxx")
            .endpoint("https://huggingface.co")
            .build();

        assert_eq!(config.repo_type.as_deref(), Some("dataset"));
        assert_eq!(config.revision.as_deref(), Some("main"));
        assert_eq!(config.token.as_deref(), Some("hf_xxx"));
        assert_eq!(config.endpoint.as_deref(), Some("https://huggingface.co"));
    }

    #[test]
    fn test_hf_config_from_storage_config() {
        let storage_config = StorageConfig::new()
            .with_prop(HF_REPO_TYPE, "model")
            .with_prop(HF_REVISION, "v1.0")
            .with_prop(HF_TOKEN, "hf_xxx");

        let hf_config = HfConfig::try_from(&storage_config).unwrap();

        assert_eq!(hf_config.repo_type.as_deref(), Some("model"));
        assert_eq!(hf_config.revision.as_deref(), Some("v1.0"));
        assert_eq!(hf_config.token.as_deref(), Some("hf_xxx"));
        assert_eq!(hf_config.endpoint, None);
    }

    #[test]
    fn test_hf_config_empty() {
        let storage_config = StorageConfig::new();

        let hf_config = HfConfig::try_from(&storage_config).unwrap();

        assert_eq!(hf_config.repo_type, None);
        assert_eq!(hf_config.revision, None);
        assert_eq!(hf_config.token, None);
        assert_eq!(hf_config.endpoint, None);
    }
}
