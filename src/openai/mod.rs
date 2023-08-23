//! Interact with OpenAI's GPT models.

pub mod chat;
pub mod embed;

use serde::{Deserialize, Serialize};
use thiserror;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("chat encoding error: {0}")]
    EncodingError(std::string::FromUtf8Error),
    #[error("chat format error: {0}")]
    FormatError(serde_json::Error),
    #[error("chat function parameters error: {0}")]
    FunctionParameterError(serde_json::Error),
    #[error("chat function format error: {0}")]
    FunctionFormatError(serde_json::Error),
    #[error("network didn't return expected response")]
    NetworkError,
    #[error("failed to request chat completion: {0}")]
    InvalidChatCompletion(#[from] reqwest::Error),
    #[error("failed to get chat completion function output")]
    InvalidChatFunction,
    #[error("chat completion returned no messages")]
    EmptyChatCompletion,
    #[error("failed to request embedding")]
    InvalidEmbedding,
    #[error("failed to serailize embedding")]
    CantSerialize,
    #[error("failed to de-serailize embedding")]
    CantDeserialize,
}

type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FinishReason {
    Stop,
    Length,
}
