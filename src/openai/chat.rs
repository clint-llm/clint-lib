use async_sse::{decode as sse_decode, Decoder, Event};
use bytes::Bytes;
use futures::stream::{IntoAsyncRead, StreamExt};
use futures::{Stream, TryStreamExt};
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::time::Duration;
use tap::Pipe;

use super::{Error, FinishReason, Result};

#[derive(Debug, Serialize, Deserialize)]
enum ChatCompletionObjectValue {
    #[serde(rename = "chat.completion")]
    ChatCompletion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChatCompletionMessageRole {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "function")]
    Function,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionCallArg {
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionArg {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: ChatCompletionMessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

#[derive(Debug, PartialEq, Deserialize)]
pub struct ChatCompletionChoice {
    pub message: ChatCompletionMessage,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, PartialEq, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Deserialize)]
struct FunctionCallUpdate {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessageUpdate {
    role: Option<ChatCompletionMessageRole>,
    content: Option<String>,
    name: Option<String>,
    function_call: Option<FunctionCallUpdate>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChoiceUpdate {
    delta: ChatCompletionMessageUpdate,
    finish_reason: Option<FinishReason>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponseUpdate {
    choices: Vec<ChatCompletionChoiceUpdate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatCompletionModel {
    #[serde(rename = "gpt-4")]
    Gpt4,
    #[serde(rename = "gpt-3.5-turbo")]
    Gpt35Turbo,
    #[serde(rename = "gpt-3.5-turbo-16k")]
    Gpt35Turbo16k,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: ChatCompletionModel,
    messages: Vec<ChatCompletionMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    functions: Option<Vec<FunctionArg>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<FunctionCallArg>,
}

#[derive(Debug, Clone)]
pub struct ChatCompletionArgs {
    pub key: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub model: ChatCompletionModel,
    pub max_tokens: Option<u16>,
    pub temperature: Option<f32>,
    pub functions: Option<Vec<FunctionArg>>,
    pub function_call: Option<FunctionCallArg>,
}

impl ChatCompletionArgs {
    pub fn new(key: String) -> Self {
        Self {
            key,
            messages: Vec::new(),
            model: ChatCompletionModel::Gpt35Turbo,
            max_tokens: None,
            temperature: None,
            functions: None,
            function_call: None,
        }
    }

    pub fn with_model(mut self, model: ChatCompletionModel) -> Self {
        self.model = model;
        self
    }

    pub fn with_message(mut self, message: ChatCompletionMessage) -> Self {
        self.messages.push(message);
        self
    }

    pub fn with_messages(mut self, mut messages: Vec<ChatCompletionMessage>) -> Self {
        self.messages.append(&mut messages);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_no_functions(mut self) -> Self {
        self.functions = None;
        self
    }

    pub fn with_function(mut self, function: FunctionArg) -> Self {
        if let Some(functions) = &mut self.functions {
            functions.push(function);
        } else {
            self.functions = Some(vec![function]);
        }
        self
    }

    pub fn with_function_call(mut self, function_call: FunctionCallArg) -> Self {
        self.function_call = Some(function_call);
        self
    }
}

/// Request a chat completion.
pub async fn chat_completion(
    args: ChatCompletionArgs,
    max_retries: usize,
) -> Result<ChatCompletionResponse> {
    let mut n_retried: usize = 0;
    loop {
        match reqwest::Client::new()
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(args.key.clone())
            .json(&ChatCompletionRequest {
                model: args.model.clone(),
                messages: args.messages.clone(),
                max_tokens: args.max_tokens,
                temperature: args.temperature,
                stream: Some(false),
                functions: args.functions.clone(),
                function_call: args.function_call.clone(),
            })
            .send()
            .await
        {
            Ok(response) => {
                return response
                    .json::<ChatCompletionResponse>()
                    .await
                    .map_err(Error::InvalidChatCompletion)?
                    .pipe(Ok);
            }
            Err(err) => {
                if err.status().is_some_and(|x| x.is_server_error()) && n_retried < max_retries {
                    std::thread::sleep(Duration::from_secs(2.0f64.powi(n_retried as i32) as u64));
                    n_retried += 1;
                    continue;
                } else {
                    return Err(Error::NetworkError);
                }
            }
        }
    }
}

/// Request a chat completion whose output is a JSON object of type `T`.
///
/// Uses the _function calling_ feature to get the LLM to output a JSON object
/// conforming to the schema of `T`.
pub async fn chat_completion_function<T>(
    args: ChatCompletionArgs,
    name: String,
    description: Option<String>,
    max_retries: usize,
) -> Result<T>
where
    T: DeserializeOwned + JsonSchema,
{
    let parameters = serde_json::to_value(schema_for!(T)).map_err(Error::FunctionParameterError)?;
    let mut n_retried = 0;
    loop {
        let args = args
            .clone()
            .with_no_functions()
            .with_function(FunctionArg {
                name: name.clone(),
                description: description.clone(),
                parameters: parameters.clone(),
            })
            .with_function_call(FunctionCallArg { name: name.clone() });
        let args = if n_retried > 0 {
            args.with_temperature(0.5)
        } else {
            args
        };
        let response = chat_completion(args, max_retries).await?;
        let message = response
            .choices
            .into_iter()
            .next()
            .ok_or(Error::EmptyChatCompletion)?
            .message;
        let function_call = message
            .function_call
            .clone()
            .ok_or(Error::EmptyChatCompletion)?;
        match serde_json::from_str::<T>(&function_call.arguments) {
            Ok(result) => return Ok(result),
            Err(err) => {
                if n_retried < max_retries {
                    n_retried += 1;
                    continue;
                } else {
                    return Err(Error::FunctionFormatError(err));
                }
            }
        }
    }
}

/// Update chat compleiton response the streamed bytes.
///
/// <https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb>
fn update_response(response: &mut ChatCompletionResponse, data: &[u8]) -> Result<bool> {
    let mut updated = false;
    let data = String::from_utf8(data.to_vec()).map_err(Error::EncodingError)?;
    if data == "[DONE]" || data.trim().is_empty() {
        return Ok(updated);
    }
    let update: ChatCompletionResponseUpdate =
        serde_json::from_str(&data).map_err(Error::FormatError)?;
    if let Some(ChatCompletionChoiceUpdate {
        delta,
        finish_reason,
    }) = update.choices.into_iter().next()
    {
        match response.choices.first_mut() {
            None => {
                response.choices.push(ChatCompletionChoice {
                    message: ChatCompletionMessage {
                        role: delta.role.unwrap_or(ChatCompletionMessageRole::Assistant),
                        content: delta.content.unwrap_or(String::new()).pipe(Some),
                        name: delta.name,
                        function_call: delta.function_call.map(|x| FunctionCall {
                            name: x.name.unwrap_or(String::new()),
                            arguments: x.arguments.unwrap_or(String::new()),
                        }),
                    },
                    finish_reason: None,
                });
                updated = true;
            }
            Some(previous) => {
                if let Some(content) = delta.content {
                    if let Some(previous_content) = previous.message.content.as_mut() {
                        previous_content.push_str(&content);
                    } else {
                        previous.message.content = Some(content);
                    }
                    updated = true;
                }
                if let Some(role) = delta.role {
                    previous.message.role = role;
                    updated = true;
                }
                if let Some(name) = delta.name {
                    if let Some(previous_name) = previous.message.name.as_mut() {
                        previous_name.push_str(&name);
                    } else {
                        previous.message.name = Some(name);
                    };
                    updated = true;
                }
                if let Some(function_call) = delta.function_call {
                    if let Some(previous_function_call) = previous.message.function_call.as_mut() {
                        if let Some(name) = function_call.name {
                            previous_function_call.name.push_str(&name);
                        }
                        if let Some(arguments) = function_call.arguments {
                            previous_function_call.arguments.push_str(&arguments);
                        }
                    } else {
                        previous.message.function_call = Some(FunctionCall {
                            name: function_call.name.unwrap_or(String::new()),
                            arguments: function_call.arguments.unwrap_or(String::new()),
                        });
                    }
                    updated = true;
                }
                if let Some(finish_reason) = finish_reason {
                    previous.finish_reason = Some(finish_reason);
                    updated = true;
                }
            }
        }
    }
    Ok(updated)
}

type ReqwestStreamItem = std::result::Result<Bytes, reqwest::Error>;
type BoxedIoStream = Pin<Box<dyn Stream<Item = std::result::Result<Bytes, std::io::Error>>>>;
type Events = Decoder<IntoAsyncRead<BoxedIoStream>>;

/// Streaming chat completion response.
pub struct ChatCompletionParts {
    events: Events,
    response: ChatCompletionResponse,
}

impl ChatCompletionParts {
    async fn new_stream(
        args: ChatCompletionArgs,
        max_retries: usize,
    ) -> Result<impl Stream<Item = ReqwestStreamItem>> {
        let mut n_retried = 0;
        loop {
            match reqwest::Client::new()
                .post("https://api.openai.com/v1/chat/completions")
                .bearer_auth(args.key.clone())
                .json(&ChatCompletionRequest {
                    model: args.model.clone(),
                    messages: args.messages.clone(),
                    max_tokens: args.max_tokens,
                    temperature: args.temperature,
                    stream: Some(true),
                    functions: args.functions.clone(),
                    function_call: args.function_call.clone(),
                })
                .send()
                .await
            {
                Ok(response) => {
                    return response.bytes_stream().pipe(Ok);
                }
                Err(err) => {
                    if err.status().is_some_and(|x| x.is_server_error()) && n_retried < max_retries
                    {
                        std::thread::sleep(Duration::from_secs(
                            2.0f64.powi(n_retried as i32) as u64
                        ));
                        n_retried += 1;
                        continue;
                    } else {
                        return Err(Error::NetworkError);
                    }
                }
            }
        }
    }

    pub async fn new(args: ChatCompletionArgs, max_retries: usize) -> Result<ChatCompletionParts> {
        // TODO: map into error types that can be handled
        let stream: BoxedIoStream = Self::new_stream(args, max_retries)
            .await?
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidData))
            .boxed_local();
        let reader = stream.into_async_read();
        let events = sse_decode(reader);
        ChatCompletionParts {
            events,
            response: ChatCompletionResponse {
                choices: Vec::new(),
            },
        }
        .pipe(Ok)
    }

    /// Update the response from the stream.
    ///
    /// Returns `None` when the stream is done.
    pub async fn next(&mut self) -> Result<Option<&ChatCompletionResponse>> {
        loop {
            let event = match self.events.next().await {
                Some(event) => event,
                // return None to stop iteration
                None => break Ok(None),
            };
            match event {
                Ok(Event::Message(message)) => {
                    match update_response(&mut self.response, message.data()) {
                        Ok(false) => continue,
                        Ok(true) => break Ok(Some(&self.response)),
                        Err(_) => break Err(Error::NetworkError),
                    }
                }
                Ok(Event::Retry(_)) => continue,
                Err(_) => {
                    // TODO: retry
                    break Err(Error::NetworkError);
                }
            };
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn updates_empty_response() {
        let mut response = ChatCompletionResponse {
            choices: Vec::new(),
        };
        let data = r#"{"choices":[{"delta":{"role":"assistant"}}]}"#.as_bytes();
        assert!(update_response(&mut response, data).unwrap());
        assert_eq!(
            response,
            ChatCompletionResponse {
                choices: vec![ChatCompletionChoice {
                    message: ChatCompletionMessage {
                        role: ChatCompletionMessageRole::Assistant,
                        content: Some(String::new()),
                        name: None,
                        function_call: None,
                    },
                    finish_reason: None,
                }],
            }
        );
    }

    #[test]
    fn updates_response_content() {
        let mut response = ChatCompletionResponse {
            choices: vec![ChatCompletionChoice {
                message: ChatCompletionMessage {
                    role: ChatCompletionMessageRole::Assistant,
                    content: Some("abc".to_string()),
                    name: None,
                    function_call: None,
                },
                finish_reason: None,
            }],
        };
        let data = r#"{"choices":[{"delta":{"content":"def"}}]}"#.as_bytes();
        assert!(update_response(&mut response, data).unwrap());
        assert_eq!(
            response,
            ChatCompletionResponse {
                choices: vec![ChatCompletionChoice {
                    message: ChatCompletionMessage {
                        role: ChatCompletionMessageRole::Assistant,
                        content: Some("abcdef".to_string()),
                        name: None,
                        function_call: None,
                    },
                    finish_reason: None,
                }],
            }
        )
    }

    #[test]
    fn updates_response_function_call() {
        let mut response = ChatCompletionResponse {
            choices: vec![ChatCompletionChoice {
                message: ChatCompletionMessage {
                    role: ChatCompletionMessageRole::Assistant,
                    content: Some(String::new()),
                    name: None,
                    function_call: None,
                },
                finish_reason: None,
            }],
        };
        let data = r#"{"choices":[{"delta":{"function_call":{"name":"abc"}}}]}"#.as_bytes();
        assert!(update_response(&mut response, data).unwrap());
        assert_eq!(
            response,
            ChatCompletionResponse {
                choices: vec![ChatCompletionChoice {
                    message: ChatCompletionMessage {
                        role: ChatCompletionMessageRole::Assistant,
                        content: Some(String::new()),
                        name: None,
                        function_call: Some(FunctionCall {
                            name: "abc".to_string(),
                            arguments: String::new(),
                        }),
                    },
                    finish_reason: None,
                }],
            }
        )
    }
}
