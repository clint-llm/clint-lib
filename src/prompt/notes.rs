use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tap::Pipe;

use super::utils::{quote_lines, Error, Result, SystemInstructionsExcerpts};
use crate::openai::chat::{
    chat_completion_function, ChatCompletionMessage, ChatCompletionMessageRole,
};
use crate::{openai::chat::ChatCompletionArgs, utils::render_template};

#[derive(Debug, Default, JsonSchema, Serialize, Deserialize)]
pub struct Notes {
    #[schemars(description = "The patient's Chief Complaint")]
    pub chief_complaint: String,
    #[schemars(description = "History of Present Illness")]
    pub history_of_present_illness: String,
    #[schemars(description = "The patient's medical history")]
    pub patient_history: String,
}

const NOTES_MARKDOWN: &'static str = "\
{depth}# Chief Complaint

{chief_complaint}

{depth}# History of Present Illness

{history_of_present_illness}

{depth}# Patient History

{patient_history}
";

#[derive(Serialize)]
struct NotesMarkdown<'a> {
    depth: &'a str,
    chief_complaint: &'a str,
    history_of_present_illness: &'a str,
    patient_history: &'a str,
}

impl<'a> NotesMarkdown<'a> {
    fn render(&self) -> Result<String> {
        render_template(NOTES_MARKDOWN, &self).map_err(Error::TemplateError)
    }
}

impl Notes {
    pub fn to_markdown(&self, depth: usize) -> String {
        let depth = "#".repeat(depth);
        NotesMarkdown {
            depth: &depth,
            chief_complaint: &self.chief_complaint,
            history_of_present_illness: &self.history_of_present_illness,
            patient_history: &self.patient_history,
        }
        .render()
        .unwrap()
    }
}

const INFORMATION_NOTES: &'static str = "\
# Structure of Clinical Notes

Clinical notes must contain the following sections.

## Chief Complaint

The _Chief Complaint_ is the reason the patient is seeking a clinical consultation.

## History of Present Illness

The _History of Present Illness_ is the elaboration of the patient's chief complaint. \
Include information related to the chief complaint such as: \
onset, \
location, \
duration, \
characterization, \
alleviating and aggravating factors, \
radiation, \
temporal factor, \
severity.

## Patient History

The _Patient History_ is the patient's relevant medical history. \
Include information about the patient but not strictly related to the chief complaint such as: \
current or past medical conditions, \
surgical history, \
family history, \
etc.\
";

const MESSAGE_INSTRUCTIONS_NOTES: &'static str = "\
You have recorded the following patient notes:

{current_notes}

Update your notes by adding information from the following patient statement. \
When the patient's statement includes symptoms, \
the explanation could be incorrect or incomplete, \
so include any plausible interpretations of the patient's symptoms in your notes. \
Include only information that belongs in clinical notes. \
Be sure to follow the complete structure of clinical notes, \
including empty sections if you lack information. \
Don't discard any information from your current notes.

Be on the lookout for information that isn't plausible from a physiological or biochemical perspective. \
If you find any, take note of the contradiction.

Patient statement:

{statement}\
";

#[derive(Serialize)]
struct MessageInstructionsNotes {
    current_notes: String,
    statement: String,
}

impl MessageInstructionsNotes {
    fn new(statement: &str, current_notes: &Notes) -> Self {
        Self {
            current_notes: current_notes.to_markdown(0).as_str().pipe(quote_lines),
            statement: quote_lines(statement),
        }
    }

    fn render(&self) -> Result<String> {
        render_template(MESSAGE_INSTRUCTIONS_NOTES, &self).map_err(Error::TemplateError)
    }
}

const MESSAGE_INSTRUCTIONS: &'static str = "\
Start writing clinical notes with information from the following patient statement. \
When the patient's statement includes symptoms, \
the explanation could be incorrect or incomplete, \
so include any plausible interpretations of the patient's symptoms in your notes. \
Include only information that belongs in clinical notes. \
Be sure to follow the complete structure of clinical notes, \
including empty sections if you lack information, \
and capture the patient's chief complaint.

Be on the lookout for information that isn't plausible from a physiological or biochemical perspective. \
If you find any, take note of the contradiction.

Patient statement:

{statement}\
";

#[derive(Serialize)]
struct MessageInstructions {
    statement: String,
}

impl MessageInstructions {
    fn new(statement: &str) -> Self {
        Self {
            statement: quote_lines(statement),
        }
    }

    fn render(&self) -> Result<String> {
        render_template(MESSAGE_INSTRUCTIONS, &self).map_err(Error::TemplateError)
    }
}

/// Create or update the clinical notes `current_notes` with the patient
/// `statement`.
pub async fn create_update_notes(
    statement: String,
    current_notes: Option<&Notes>,
    key: String,
    max_retries: usize,
) -> Result<Notes> {
    let instructions = if let Some(current_notes) = current_notes {
        MessageInstructionsNotes::new(&statement, current_notes).render()?
    } else {
        MessageInstructions::new(&statement).render()?
    };
    let args = ChatCompletionArgs::new(key)
        .with_temperature(0.0)
        .with_message(ChatCompletionMessage {
            role: ChatCompletionMessageRole::System,
            content: Some(
                SystemInstructionsExcerpts::new(&vec![INFORMATION_NOTES.to_string()]).render()?,
            ),
            name: None,
            function_call: None,
        })
        .with_message(ChatCompletionMessage {
            role: ChatCompletionMessageRole::User,
            content: Some(instructions),
            name: None,
            function_call: None,
        });
    chat_completion_function(
        args,
        "record_notes".to_string(),
        Some("Record patient notes.".to_string()),
        max_retries,
    )
    .await
    .map_err(Error::OpenAIError)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn notes_renders_markdown() {
        let notes_md = Notes {
            chief_complaint: "Patient has a headache.".to_string(),
            history_of_present_illness: String::new(),
            patient_history: String::new(),
        }
        .to_markdown(0);
        assert!(notes_md.starts_with("# "));
        assert!(notes_md.contains("Patient has a headache."));
    }

    #[test]
    fn notes_renders_markdown_with_depth() {
        let notes_md = Notes {
            chief_complaint: "Patient has a headache.".to_string(),
            history_of_present_illness: String::new(),
            patient_history: String::new(),
        }
        .to_markdown(2);
        assert!(notes_md.starts_with("### "));
    }

    #[test]
    fn instructions_renders_with_notes() {
        let instructions = MessageInstructionsNotes::new(
            "abc",
            &Notes {
                chief_complaint: "abc".to_string(),
                ..Default::default()
            },
        )
        .render()
        .unwrap();
        assert!(instructions.contains("patient notes:\n\n> "));
        assert!(instructions.contains("Patient statement:\n\n> abc"));
    }

    #[test]
    fn instructions_renders_without_notes() {
        let instructions = MessageInstructions::new("abc").render().unwrap();
        assert!(instructions.contains("Patient statement:\n\n> abc"));
    }
}
