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

//! Integration tests for FileIO with HuggingFace Hub.
//!
//! These tests read from public HuggingFace datasets and require network access.
//! The path format is `hf://<owner>/<repo>/<path>` — the repo type prefix (e.g.
//! `datasets/`) must NOT be included since `repo_type` defaults to `dataset`
//! and opendal prepends it automatically in the API URL.
//!
//! Run with: `cargo test -p iceberg --features storage-hf -- --ignored hf`
#[cfg(all(test, feature = "storage-hf"))]
mod tests {
    use iceberg::io::FileIOBuilder;

    fn get_file_io() -> iceberg::io::FileIO {
        FileIOBuilder::new("hf").build().unwrap()
    }

    #[tokio::test]
    #[ignore = "requires network access to huggingface.co"]
    async fn test_hf_public_dataset_exists() {
        let file_io = get_file_io();

        let exists = file_io
            .exists("hf://stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet")
            .await
            .unwrap();
        assert!(exists);
    }

    #[tokio::test]
    #[ignore = "requires network access to huggingface.co"]
    async fn test_hf_public_dataset_read_metadata() {
        let file_io = get_file_io();

        let input = file_io
            .new_input("hf://stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet")
            .unwrap();

        let metadata = input.metadata().await.unwrap();
        assert!(metadata.size > 0);
    }

    #[tokio::test]
    #[ignore = "requires network access to huggingface.co"]
    async fn test_hf_nonexistent_path() {
        let file_io = get_file_io();

        let exists = file_io
            .exists("hf://stanfordnlp/imdb/does-not-exist.parquet")
            .await
            .unwrap();
        assert!(!exists);
    }
}
