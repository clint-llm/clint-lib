use serde::{Deserialize, Serialize};
use tap::Pipe;

use super::{Error, Result};

#[derive(Debug, Deserialize)]
enum EmbeddingObjectValue {
    #[serde(rename = "embedding")]
    Embedding,
}

#[derive(Debug, Deserialize, Serialize)]
pub enum EmbeddingModel {
    #[serde(rename = "text-embedding-ada-002")]
    TextEmbeddingAda002,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    model: EmbeddingModel,
    input: &'a str,
}

/// Generate an embedding for the given `text`.
pub async fn embed(token: &str, text: &str) -> Result<Vec<f32>> {
    reqwest::Client::new()
        .post("https://api.openai.com/v1/embeddings")
        .bearer_auth(token)
        .json(&EmbeddingRequest {
            model: EmbeddingModel::TextEmbeddingAda002,
            input: text,
        })
        .send()
        .await
        .map_err(|_| Error::InvalidEmbedding)?
        .json::<EmbeddingResponse>()
        .await
        .ok()
        .and_then(|x| x.data.into_iter().next())
        .map(|x| x.embedding)
        .ok_or(Error::InvalidEmbedding)?
        .pipe(Ok)
}
