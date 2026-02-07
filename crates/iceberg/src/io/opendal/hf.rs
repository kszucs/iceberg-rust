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

use std::collections::HashMap;

use opendal::Operator;
use opendal::services::HuggingfaceConfig;

use crate::io::config::{HF_REPO_TYPE, HF_REVISION, HF_TOKEN};
use crate::{Error, ErrorKind, Result};

/// Parse iceberg properties to [`HuggingfaceConfig`].
pub(crate) fn hf_config_parse(mut m: HashMap<String, String>) -> Result<HuggingfaceConfig> {
    let mut cfg = HuggingfaceConfig::default();

    // Default repo_type to "dataset" for Iceberg use cases
    cfg.repo_type = Some(
        m.remove(HF_REPO_TYPE)
            .unwrap_or_else(|| "dataset".to_string()),
    );

    // Default revision to "main"
    cfg.revision = Some(
        m.remove(HF_REVISION)
            .unwrap_or_else(|| "main".to_string()),
    );

    if let Some(token) = m.remove(HF_TOKEN) {
        cfg.token = Some(token);
    }

    Ok(cfg)
}

/// Build a new OpenDAL [`Operator`] for HuggingFace and extract the relative path.
///
/// Path format: `hf://<owner>/<repo>/<relative_path>`
pub(crate) fn hf_config_build<'a>(
    cfg: &HuggingfaceConfig,
    path: &'a str,
) -> Result<(Operator, &'a str)> {
    let stripped = path.strip_prefix("hf://").ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Invalid hf url: {path}, expected hf:// prefix"),
        )
    })?;

    // Split into owner, repo, and relative path
    // e.g. "google-research-datasets/mbpp/data/train.parquet"
    //   -> owner: "google-research-datasets", repo: "mbpp", relative: "data/train.parquet"
    let parts: Vec<&str> = stripped.splitn(3, '/').collect();
    if parts.len() < 2 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Invalid hf url: {path}, expected <owner>/<repo> in path"),
        ));
    }

    let repo_id = format!("{}/{}", parts[0], parts[1]);
    let relative_path = if parts.len() > 2 { parts[2] } else { "" };

    let mut cfg = cfg.clone();
    cfg.repo_id = Some(repo_id);

    let op = Operator::from_config(cfg)
        .map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                "Failed to create HuggingFace operator",
            )
            .with_source(e)
        })?
        .finish();

    // Return a reference into the original path string for the relative portion
    let relative_start = path.len() - relative_path.len();
    Ok((op, &path[relative_start..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_config_parse_defaults() {
        let props = HashMap::new();
        let config = hf_config_parse(props).unwrap();

        assert_eq!(config.repo_type.as_deref(), Some("dataset"));
        assert_eq!(config.revision.as_deref(), Some("main"));
        assert_eq!(config.token, None);
    }

    #[test]
    fn test_hf_config_parse_with_props() {
        let mut props = HashMap::new();
        props.insert(HF_REPO_TYPE.to_string(), "model".to_string());
        props.insert(HF_REVISION.to_string(), "v1.0".to_string());
        props.insert(HF_TOKEN.to_string(), "hf_xxx".to_string());

        let config = hf_config_parse(props).unwrap();

        assert_eq!(config.repo_type.as_deref(), Some("model"));
        assert_eq!(config.revision.as_deref(), Some("v1.0"));
        assert_eq!(config.token.as_deref(), Some("hf_xxx"));
    }

    #[test]
    fn test_parse_hf_path() {
        let config = hf_config_parse(HashMap::new()).unwrap();

        let (_, relative) = hf_config_build(
            &config,
            "hf://google-research-datasets/mbpp/data/train.parquet",
        )
        .unwrap();

        assert_eq!(relative, "data/train.parquet");
    }

    #[test]
    fn test_parse_hf_path_no_relative() {
        let config = hf_config_parse(HashMap::new()).unwrap();

        let (_, relative) = hf_config_build(&config, "hf://my-org/my-dataset").unwrap();

        assert_eq!(relative, "");
    }

    #[test]
    fn test_parse_hf_path_invalid_prefix() {
        let config = hf_config_parse(HashMap::new()).unwrap();

        let result = hf_config_build(&config, "s3://bucket/key");

        assert!(result.is_err());
    }

    #[test]
    fn test_parse_hf_path_missing_repo() {
        let config = hf_config_parse(HashMap::new()).unwrap();

        let result = hf_config_build(&config, "hf://only-owner");

        assert!(result.is_err());
    }
}
