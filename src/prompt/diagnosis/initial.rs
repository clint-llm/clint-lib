use futures::future::join_all;
use serde::Serialize;
use tap::Pipe;

use super::super::notes::Notes;
use super::super::utils::{embed_for_db, quote_lines, Error, Result};
use super::super::utils::{get_excerpt, SystemInstructionsExcerpts};
use super::utils::{dedup_diagnoses, find_diagnosis_doc, CandidateDiagnoses, ResolvedDiagnosis};
use crate::docdb::DocDb;
use crate::openai::chat::{
    chat_completion_function, ChatCompletionMessage, ChatCompletionMessageRole, ChatCompletionModel,
};
use crate::prompt::utils::EmbedStructure;
use crate::{openai::chat::ChatCompletionArgs, utils::render_template};

const MESSAGE_LIST_INSTRUCTIONS: &'static str = "\
Consider the following clinical notes:

{notes}

List some plausible candidate diagnoses that are supported by the notes,
in order from most likely to least likely. \
Explain why the notes support and contradict each candidate diagnosis.\
";

#[derive(Serialize)]
struct MessageInstructions {
    notes: String,
}

impl MessageInstructions {
    fn new(notes: &Notes) -> Self {
        Self {
            notes: notes.to_markdown(0).as_str().pipe(quote_lines),
        }
    }

    fn render(&self) -> Result<String> {
        render_template(MESSAGE_LIST_INSTRUCTIONS, &self).map_err(Error::TemplateError)
    }
}

/// Come up with an initial diagnosis given the `notes`.
///
/// If a `statement` is provided, it is used to help find context documents.
pub async fn initial_diagnosis(
    notes: &Notes,
    statement: Option<&str>,
    db: &DocDb,
    key: String,
    max_retries: usize,
) -> Result<Vec<ResolvedDiagnosis>> {
    let embedding = embed_for_db(
        &EmbedStructure::new(notes, None, statement).render()?,
        db,
        &key,
    )
    .await?;
    let hashes = db.get_similar(embedding.view(), 8, None);
    let excerpts = hashes
        .iter()
        .map(|x| get_excerpt(x, db))
        .pipe(join_all)
        .await
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let args = ChatCompletionArgs::new(key.clone())
        .with_model(ChatCompletionModel::Gpt35Turbo16k)
        .with_temperature(0.0)
        .with_message(ChatCompletionMessage {
            role: ChatCompletionMessageRole::System,
            content: Some(SystemInstructionsExcerpts::new(&excerpts).render()?),
            name: None,
            function_call: None,
        })
        .with_message(ChatCompletionMessage {
            role: ChatCompletionMessageRole::User,
            content: Some(MessageInstructions::new(notes).render()?),
            name: None,
            function_call: None,
        });
    let candidates: CandidateDiagnoses = chat_completion_function(
        args,
        "list_diagnoses".to_string(),
        Some("List plausible diagnoses.".to_string()),
        max_retries,
    )
    .await
    .map_err(Error::OpenAIError)?;

    let resolved = candidates
        .diagnoses
        .iter()
        .map(|x| find_diagnosis_doc(x, db, &key))
        .pipe(join_all)
        .await
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let resolved = dedup_diagnoses(resolved);
    resolved.pipe(Ok)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn instructions_renders() {
        let instructions = MessageInstructions::new(&Notes {
            chief_complaint: "abc".to_string(),
            ..Default::default()
        })
        .render()
        .unwrap();
        assert!(instructions.contains("notes:\n\n> # Chief Complaint\n> \n> abc"));
    }
}
