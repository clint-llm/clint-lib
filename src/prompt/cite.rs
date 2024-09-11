use futures::future::join_all;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tap::Pipe;

use super::utils::{embed_for_db, get_excerpt, quote_lines, Error, Result, SYSTEM_IDENTITY};
use crate::docdb::DocDb;
use crate::openai::chat::{
    chat_completion_function, ChatCompletionArgs, ChatCompletionMessage, ChatCompletionMessageRole,
    ChatCompletionModel,
};
use crate::utils::render_template;

#[derive(Debug, Default, JsonSchema, Deserialize)]
pub struct CiteExcerpt {
    #[schemars(description = "The ID must contain only hex characters.")]
    pub id: String,
    #[schemars(description = "The excerpt title.")]
    pub title: String,
}

#[derive(Debug, Default, JsonSchema, Deserialize)]
pub struct CiteDocuments {
    #[schemars(description = "The excerpts to cite.")]
    pub excerpts: Vec<CiteExcerpt>,
}

const MESSAGE_INSTRUCTIONS: &'static str = "\
Consider the following document excerpts and their IDs:

{excerpts}

Consider the following message: 

{message}

Provide select the most relevant documents excerpts to cite. \
Cite only excerpts that are related to the contents of the above message. \
Don't cite any excerpts if none are related to the message, \
Include the excerpt's Markdown title and ID. \
The ID can be found in the link `<id:...>`).\
";

#[derive(Serialize)]
struct MessageInstructions {
    pub excerpts: String,
    pub message: String,
}

impl MessageInstructions {
    fn render(&self) -> Result<String> {
        render_template(MESSAGE_INSTRUCTIONS, &self).map_err(Error::TemplateError)
    }
}

impl MessageInstructions {
    fn new(message: &str, excerpts: Vec<String>) -> Self {
        Self {
            excerpts: excerpts
                .iter()
                .map(|x| quote_lines(x.as_str()))
                .collect::<Vec<String>>()
                .join("\n\n"),
            message: message.pipe(quote_lines),
        }
    }
}

pub async fn cite(
    message: &str,
    db: &DocDb,
    key: String,
    max_retries: usize,
) -> Result<CiteDocuments> {
    let embedding = embed_for_db(message, db, &key).await?;
    let hashes = db.get_similar(embedding.view(), 8, None);
    let excerpts = hashes
        .iter()
        .map(|x| get_excerpt(x, db))
        .pipe(join_all)
        .await
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    chat_completion_function(
        ChatCompletionArgs::new(key)
            .with_model(ChatCompletionModel::Gpt4o)
            .with_temperature(0.0)
            .with_message(ChatCompletionMessage {
                role: ChatCompletionMessageRole::System,
                content: Some(SYSTEM_IDENTITY.to_string()),
                name: None,
                function_call: None,
            })
            .with_message(ChatCompletionMessage {
                role: ChatCompletionMessageRole::User,
                content: Some(MessageInstructions::new(message, excerpts).render()?),
                name: None,
                function_call: None,
            }),
        "list_document_ids".to_string(),
        Some("List document IDs.".to_string()),
        max_retries,
    )
    .await
    .map_err(Error::OpenAIError)
}
