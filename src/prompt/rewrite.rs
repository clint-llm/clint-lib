use serde::Serialize;

use super::utils::SYSTEM_IDENTITY;
use super::utils::{quote_lines, Error, Result};
use crate::openai::chat::{
    ChatCompletionArgs, ChatCompletionMessage, ChatCompletionMessageRole, ChatCompletionParts,
};
use crate::utils::render_template;

const MESSAGE_INSTRUCTIONS: &'static str = "\
Rewrite the following statement using precise medical terminology, \
referring to the patient in the 3rd person. \
Don't assume the statement is complete or accurate, \
so be sure to include symptoms or systems that could be related to the statement.

Statement:

{query}\
";

#[derive(Serialize)]
struct MessageInstructions {
    pub query: String,
}

impl MessageInstructions {
    fn render(&self) -> Result<String> {
        render_template(MESSAGE_INSTRUCTIONS, &self).map_err(Error::TemplateError)
    }
}

impl MessageInstructions {
    fn new(query: &str) -> Self {
        Self {
            query: quote_lines(query),
        }
    }
}

/// Rewrite a user's `message` in the 3rd person using precise medical terminology.
pub async fn rewrite_message(
    message: String,
    key: String,
    max_retries: usize,
) -> Result<ChatCompletionParts> {
    ChatCompletionParts::new(
        ChatCompletionArgs::new(key)
            .with_temperature(0.0)
            .with_message(ChatCompletionMessage {
                role: ChatCompletionMessageRole::System,
                content: Some(SYSTEM_IDENTITY.to_string()),
                name: None,
                function_call: None,
            })
            .with_message(ChatCompletionMessage {
                role: ChatCompletionMessageRole::User,
                content: Some(MessageInstructions::new(&message).render()?),
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
        let instructions = MessageInstructions::new("abc").render().unwrap();
        assert!(instructions.contains("Statement:\n\n> abc"));
    }
}
