//! # Clint LLM
//!
//! Clint LLM is intended to be used as a web application. This library
//! compiles to WASM and exposes the `StateJs` and `DocDbJs` structures.

#![warn(missing_docs)]

use core::fmt::Debug;

use futures::future::join_all;
use hex;

mod docdb;
mod openai;
mod prompt;
mod utils;

use prompt::{
    cite::cite,
    diagnosis::{initial_diagnosis, refine_diagnosis, ResolvedDiagnosis},
    notes::{create_update_notes, Notes},
    respond::respond,
    rewrite::rewrite_message,
};
use serde::{Deserialize, Serialize};
use serde_json;
use tap::Pipe;
use wasm_bindgen::prelude::*;

use docdb::{DocDb, DocId};
use openai::chat::{ChatCompletionMessage, ChatCompletionMessageRole, ChatCompletionParts};

/// Library errors.
#[allow(missing_docs)]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Streaming didn't complete.")]
    StreamingError,
    #[error("Array contains incorrect values.")]
    ArrayError,
    #[error("OpenAI error: {0}")]
    OpenAIError(openai::Error),
    #[error(transparent)]
    DocumentDbError(docdb::Error),
    #[error(transparent)]
    PromptError(prompt::utils::Error),
    #[error("Serialization error: {0}")]
    SerdeError(serde_json::Error),
}

impl From<Error> for JsValue {
    fn from(e: Error) -> Self {
        JsValue::from_str(&e.to_string())
    }
}

type Result<T> = core::result::Result<T, Error>;

/// State for a sequence of chat message updates.
#[wasm_bindgen]
pub struct ChatMessageUpdates {
    parts: ChatCompletionParts,
}

#[wasm_bindgen]
impl ChatMessageUpdates {
    /// Get the next chat message update.
    pub async fn next(&mut self) -> Result<Option<String>> {
        self.parts
            .next()
            .await
            .map_err(Error::OpenAIError)?
            .and_then(|x| x.choices.first())
            .and_then(|x| x.message.content.as_ref().map(|y| y.to_string()))
            .pipe(Ok)
    }
}

/// Wraps a `DocDb` object for passing between Rust and JS.
#[wasm_bindgen]
pub struct DocDbJs {
    db: DocDb,
}

#[wasm_bindgen]
impl DocDbJs {
    /// Build a new `DocDb` wrapped in a `DocDbJs`.
    ///
    /// Build from the raw bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(
        origin: String,
        embeddings: &[u8],
        embeddings_pca_mapping: &[u8],
        embeddings_hash: &[u8],
        parents: &[u8],
        titles: &[u8],
        urls: &[u8],
        is_introduction: &[u8],
        is_condition: &[u8],
        is_symptoms: &[u8],
    ) -> Result<DocDbJs> {
        DocDbJs {
            db: DocDb::new(
                origin,
                embeddings,
                Some(embeddings_pca_mapping),
                embeddings_hash,
                parents,
                titles,
                urls,
                is_introduction,
                is_condition,
                is_symptoms,
            )
            .map_err(Error::DocumentDbError)?,
        }
        .pipe(Ok)
    }
}

/// The state of the conversation.
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct StateJs {
    statement: Option<String>,
    notes: Option<Notes>,
    diagnoses: Option<Vec<ResolvedDiagnosis>>,
    messages: Vec<ChatCompletionMessage>,
}

#[wasm_bindgen]
impl StateJs {
    #[wasm_bindgen(constructor)]
    /// Build a new empty conversation state.
    pub fn new() -> StateJs {
        StateJs {
            statement: None,
            notes: None,
            diagnoses: None,
            messages: Vec::new(),
        }
    }

    /// Serialize to a JSON string.
    pub fn to_string(&self) -> Result<String> {
        serde_json::to_string(&self).map_err(Error::SerdeError)
    }

    /// Deserialize from a JSON string.
    pub fn from_string(s: &str) -> Result<StateJs> {
        serde_json::from_str(s).map_err(Error::SerdeError)
    }

    /// Set the user statement.
    pub fn set_statement(&mut self, statement: Option<String>) {
        self.statement = statement;
    }

    /// Get the clinical notes as a Markdown string.
    pub fn notes_to_markdown(&self, depth: usize) -> String {
        self.notes.as_ref().map_or_else(
            || Notes::default().to_markdown(depth),
            |x| x.to_markdown(depth),
        )
    }

    /// Get the candidate diagnoses as a Markdown string.
    pub fn diagnoses_to_markdown(&self, depth: usize) -> String {
        self.diagnoses
            .as_ref()
            .map(|x| {
                x.iter()
                    .map(|x| x.to_markdown(depth))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            })
            .unwrap_or_default()
    }

    /// Add a user message to the chat history.
    pub fn add_user_message(&mut self, message: String) {
        self.messages.push(ChatCompletionMessage {
            role: ChatCompletionMessageRole::User,
            content: Some(message),
            name: None,
            function_call: None,
        });
    }

    /// Add as assistant reply to the chat history.
    pub fn add_assistant_message(&mut self, message: String) {
        self.messages.push(ChatCompletionMessage {
            role: ChatCompletionMessageRole::Assistant,
            content: Some(message),
            name: None,
            function_call: None,
        });
    }
}

/// Re-write the user's message into a medical statement.
#[wasm_bindgen]
pub async fn rewrite_message_js(message: &str, key: &str) -> Result<ChatMessageUpdates> {
    ChatMessageUpdates {
        parts: rewrite_message(message.to_string(), key.to_string(), 3)
            .await
            .map_err(Error::PromptError)?,
    }
    .pipe(Ok)
}

/// Create or update clinical notes from the statement in the notes.
#[wasm_bindgen]
pub async fn create_notes_js(state: StateJs, key: &str) -> Result<StateJs> {
    let statement = match state.statement {
        Some(x) => x,
        None => return state.pipe(Ok),
    };
    let notes = create_update_notes(statement.clone(), state.notes.as_ref(), key.to_string(), 3)
        .await
        .map_err(Error::PromptError)?;
    StateJs {
        statement: Some(statement),
        notes: Some(notes),
        ..state
    }
    .pipe(Ok)
}

/// List initial candidate diagnoses from the notes in the state.
#[wasm_bindgen]
pub async fn initial_diagnosis_js(state: StateJs, db: &DocDbJs, key: &str) -> Result<StateJs> {
    let notes = match &state.notes {
        Some(x) => x,
        None => return state.pipe(Ok),
    };
    let diagnoses = initial_diagnosis(
        notes,
        state.statement.as_deref(),
        &db.db,
        key.to_string(),
        3,
    )
    .await
    .map_err(Error::PromptError)?;
    StateJs {
        diagnoses: Some(diagnoses),
        ..state
    }
    .pipe(Ok)
}

/// Refine the reasoning for each diagnosis in the state.
#[wasm_bindgen]
pub async fn refine_diagnosis_js(state: StateJs, db: &DocDbJs, key: &str) -> Result<StateJs> {
    let mut state = state;
    let notes = match &state.notes {
        Some(x) => x,
        None => return state.pipe(Ok),
    };
    let diagnoses = match state.diagnoses.take() {
        Some(x) => x,
        None => return state.pipe(Ok),
    };
    let diagnoses = diagnoses
        .into_iter()
        .take(8)
        .map(|x| {
            refine_diagnosis(
                notes,
                x,
                state.statement.as_deref(),
                &db.db,
                key.to_string(),
                3,
            )
        })
        .pipe(join_all)
        .await
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    StateJs {
        diagnoses: Some(diagnoses),
        ..state
    }
    .pipe(Ok)
}

/// Respond to the user's message using the notes and possibly the diagnoses in
/// the state as context.
#[wasm_bindgen]
pub async fn respond_js(
    state: &StateJs,
    message: &str,
    diagnosis: bool,
    db: &DocDbJs,
    key: &str,
) -> Result<Option<ChatMessageUpdates>> {
    let notes = match &state.notes {
        Some(x) => x,
        None => return Ok(None),
    };
    ChatMessageUpdates {
        parts: respond(
            notes,
            message.to_string(),
            if diagnosis {
                state.diagnoses.as_ref()
            } else {
                None
            },
            state.statement.as_deref(),
            state.messages.clone(),
            &db.db,
            key.to_string(),
            3,
        )
        .await
        .map_err(Error::PromptError)?,
    }
    .pipe(Some)
    .pipe(Ok)
}

/// Cite documents that are relevant for a message (assistant response).
#[wasm_bindgen]
pub async fn cite_js(message: &str, db: &DocDbJs, key: &str) -> Result<String> {
    cite(message, &db.db, key.to_string(), 3)
        .await
        .map_err(Error::PromptError)?
        .excerpts
        .into_iter()
        .map(|x| {
            let mut hash: DocId = [0u8; 16];
            match hex::decode_to_slice(x.id, &mut hash) {
                Ok(_) => (),
                Err(_) => return None,
            };
            match db.db.get_url(&hash) {
                Some(url) => Some(format!("- [{}]({})", x.title, url)),
                None => None,
            }
        })
        .flatten()
        .collect::<Vec<_>>()
        .join("\n")
        .pipe(Ok)
}
