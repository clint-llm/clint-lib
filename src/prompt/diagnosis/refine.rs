use futures::future::join_all;
use serde::Serialize;
use tap::Pipe;

use super::super::notes::Notes;
use super::super::utils::{embed_for_db, quote_lines, Error, Result};
use super::super::utils::{get_excerpt, SystemInstructionsExcerpts};
use super::utils::{CandidateDiagnosis, ResolvedDiagnosis};
use crate::docdb::DocDb;
use crate::openai::chat::{
    chat_completion, ChatCompletionMessage, ChatCompletionMessageRole, ChatCompletionModel,
};
use crate::prompt::utils::EmbedStructure;
use crate::{openai::chat::ChatCompletionArgs, utils::render_template};

const MESSAGE_INSTRUCTIONS: &'static str = "\
Consider the following clinical notes:

{notes}

Consider the following diagnosis:

{candidate_diagnosis}

Can you improve on the reasoning for this diagnosis given the notes? \
Correct any inaccuracies in the reasoning. \
Explain why the notes support the diagnosis. \
Explain if there discrepancies between the notes and the diagnosis. \
Look for information that is incompatible with a diagnosis. \
Take note of significant contradictions, \
basing your reasoning on physiological and biomechanical principles. \
Keep in mind that the notes might be incomplete, \
so some manifestations of the diagnosis might be missing from the notes. \
Answer in 50 words or less.\
";

#[derive(Serialize)]
struct MessageInstructions {
    notes: String,
    candidate_diagnosis: String,
}

impl MessageInstructions {
    fn new(notes: &Notes, candidate_diagnosis: &CandidateDiagnosis) -> Self {
        Self {
            notes: notes.to_markdown(0).as_str().pipe(quote_lines),
            candidate_diagnosis: candidate_diagnosis
                .to_markdown(0)
                .as_str()
                .pipe(quote_lines),
        }
    }

    fn render(&self) -> Result<String> {
        render_template(MESSAGE_INSTRUCTIONS, &self).map_err(Error::TemplateError)
    }
}

/// Refine an existing `diagnosis` by looking up relevant documents and
/// prompting the LLM to reason about the diagnosis given the `notes`.
///
/// If a `statement` is provided, it is used to help find context documents.
pub async fn refine_diagnosis(
    notes: &Notes,
    diagnosis: ResolvedDiagnosis,
    statement: Option<&str>,
    db: &DocDb,
    key: String,
    max_retries: usize,
) -> Result<ResolvedDiagnosis> {
    let embedding = embed_for_db(
        &EmbedStructure::new(notes, Some(&vec![diagnosis.clone()]), statement).render()?,
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
        .with_model(ChatCompletionModel::Gpt4o)
        .with_temperature(0.0)
        .with_message(ChatCompletionMessage {
            role: ChatCompletionMessageRole::System,
            content: Some(SystemInstructionsExcerpts::new(&excerpts).render()?),
            name: None,
            function_call: None,
        })
        .with_message(ChatCompletionMessage {
            role: ChatCompletionMessageRole::User,
            content: Some(MessageInstructions::new(notes, &diagnosis.diagnosis).render()?),
            name: None,
            function_call: None,
        });
    let refined = chat_completion(args, max_retries)
        .await
        .map_err(Error::OpenAIError)?
        .choices
        .into_iter()
        .next()
        .ok_or(Error::NetworkResponseError)?
        .message
        .content
        .ok_or(Error::NetworkResponseError)?;

    Ok(ResolvedDiagnosis {
        refined: Some(refined),
        ..diagnosis.clone()
    })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn instructions_renders() {
        let instructions = MessageInstructions::new(
            &Notes {
                chief_complaint: "abc".to_string(),
                ..Default::default()
            },
            &CandidateDiagnosis {
                name: "bcd".to_string(),
                reasoning_for: String::new(),
                reasoning_against: String::new(),
            },
        )
        .render()
        .unwrap();
        assert!(instructions.contains("notes:\n\n> # Chief Complaint\n> \n> abc"));
        assert!(instructions.contains("diagnosis:\n\n> # bcd"));
    }
}
