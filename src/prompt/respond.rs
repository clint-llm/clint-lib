use futures::future::join_all;
use serde::Serialize;
use tap::Pipe;

use super::diagnosis::ResolvedDiagnosis;
use super::notes::Notes;
use super::utils::{
    embed_for_db, get_excerpt, quote_lines, EmbedStructure, Error, Result,
    SystemInstructionsExcerpts,
};
use crate::docdb::DocDb;
use crate::openai::chat::{
    ChatCompletionArgs, ChatCompletionMessage, ChatCompletionMessageRole, ChatCompletionModel,
    ChatCompletionParts,
};
use crate::utils::render_template;

const MESSAGE_INSTRUCTIONS: &'static str = "\
My message is:

{message}

You have recorded the following clinical notes about me:

{notes}

Please respond to the my message using plain English. \
You can ask me questions to gather more information for your notes. \
Don't ask questions that have already been answered or can be answered from the notes. \
Don't repeat what was already said in a prior message.\
";

#[derive(Serialize)]
struct MessageInstructions {
    pub notes: String,
    pub message: String,
}

impl MessageInstructions {
    fn render(&self) -> Result<String> {
        render_template(MESSAGE_INSTRUCTIONS, &self).map_err(Error::TemplateError)
    }
}

impl MessageInstructions {
    fn new(notes: &Notes, message: &str) -> Self {
        Self {
            notes: notes.to_markdown(0).pipe(|x| quote_lines(x.as_str())),
            message: message.pipe(quote_lines),
        }
    }
}

const MESSAGE_INSTRUCTIONS_DIAGNOSIS: &'static str = "\
My message is:

{message}

You have recorded the following clinical notes about me:

{notes}

You have arrived at the following differential diagnosis:

{diagnosis}

Please respond to the my message using plain English. \
You can ask me questions to gather more information for your notes and to narrow the diagnosis. \
Don't ask questions that have already been answered or can be answered from the notes. \
Please also explain any plausible diagnoses. \
Don't repeat what was already said in a prior message.\
";

#[derive(Serialize)]
struct MessageInstructionsDiagnosis {
    pub notes: String,
    pub diagnosis: String,
    pub message: String,
}

impl MessageInstructionsDiagnosis {
    fn render(&self) -> Result<String> {
        render_template(MESSAGE_INSTRUCTIONS_DIAGNOSIS, &self).map_err(Error::TemplateError)
    }
}

impl MessageInstructionsDiagnosis {
    fn new(notes: &Notes, diagnoses: &Vec<ResolvedDiagnosis>, message: &str) -> Self {
        Self {
            notes: notes.to_markdown(0).pipe(|x| quote_lines(x.as_str())),
            diagnosis: diagnoses
                .into_iter()
                .map(|x| x.diagnosis.to_markdown(0))
                .collect::<Vec<_>>()
                .join("\n\n")
                .pipe(|x| quote_lines(x.as_str())),
            message: message.pipe(quote_lines),
        }
    }
}

/// Respond to the user's `message`.
///
/// If a `diagnoses` is provided, the response include a description of the
/// more plausible diagnoses. If a `statement` is provided, it is used to help
/// find context documents.
pub async fn respond(
    notes: &Notes,
    message: String,
    diagnoses: Option<&Vec<ResolvedDiagnosis>>,
    statement: Option<&str>,
    messages: Vec<ChatCompletionMessage>,
    db: &DocDb,
    key: String,
    max_retries: usize,
) -> Result<ChatCompletionParts> {
    let embedding = embed_for_db(
        &EmbedStructure::new(notes, diagnoses, statement).render()?,
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

    ChatCompletionParts::new(
        ChatCompletionArgs::new(key)
            .with_model(ChatCompletionModel::Gpt4o)
            .with_temperature(0.0)
            .with_message(ChatCompletionMessage {
                role: ChatCompletionMessageRole::System,
                content: Some(SystemInstructionsExcerpts::new(&excerpts).render()?),
                name: None,
                function_call: None,
            })
            .with_messages(messages)
            .with_message(ChatCompletionMessage {
                role: ChatCompletionMessageRole::User,
                content: Some(if let Some(diagnoses) = diagnoses {
                    MessageInstructionsDiagnosis::new(notes, diagnoses, &message).render()?
                } else {
                    MessageInstructions::new(notes, &message).render()?
                }),
                name: None,
                function_call: None,
            }),
        max_retries,
    )
    .await
    .map_err(Error::OpenAIError)
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
            "bcd",
        )
        .render()
        .unwrap();
        assert!(instructions.contains("message is:\n\n> bcd"));
        assert!(instructions.contains("notes about me:\n\n> # Chief Complaint\n> \n> abc"));
    }
}
