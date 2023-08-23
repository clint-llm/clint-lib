use std::convert::TryFrom;

use ndarray::Array1;
use noisy_float::prelude::N32;
use serde::Serialize;
use tap::Pipe;
use thiserror;

use crate::docdb::{DocDb, DocId};
use crate::openai::embed::embed;
use crate::utils::render_template;

use super::diagnosis::ResolvedDiagnosis;
use super::notes::Notes;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    TemplateError(#[from] crate::utils::Error),
    #[error(transparent)]
    OpenAIError(#[from] crate::openai::Error),
    #[error("error parsing network response")]
    NetworkResponseError,
    #[error("embedding error")]
    EmbeddingError,
}

pub type Result<T> = core::result::Result<T, Error>;

pub const SYSTEM_IDENTITY: &'static str = "\
Act as an expert clinician with extensive knowledge of medical topics: \
anatomy, \
embryology, \
histology, \
physiology, \
pathology, \
microbiology, \
immunology, \
biochemistry, \
and other related fields.

You are assessing an outpatient.\
";

const SYSTEM_INSTRUCTIONS_EXCERPTS: &'static str = "\
{system_identity}

You can refer to the following document excerpts:

{excerpts}\
";

#[derive(Serialize)]
pub struct SystemInstructionsExcerpts {
    system_identity: &'static str,
    excerpts: String,
}

impl SystemInstructionsExcerpts {
    pub fn new(excerpts: &Vec<String>) -> Self {
        Self {
            system_identity: SYSTEM_IDENTITY,
            excerpts: excerpts
                .iter()
                .map(|x| quote_lines(x.as_str()))
                .collect::<Vec<String>>()
                .join("\n\n"),
        }
    }

    pub fn render(&self) -> Result<String> {
        render_template(SYSTEM_INSTRUCTIONS_EXCERPTS, &self).map_err(Error::TemplateError)
    }
}

const EMBED_STRUCTURE: &'static str = "\
# Clinical Notes

{notes}\
{{if diagnoses}}

# Differential Diagnosis

{diagnoses}\
{{endif}}\
{{if statement}}

# Patient Statement

{statement}\
{{endif}}\
";

#[derive(Serialize)]
pub struct EmbedStructure {
    notes: String,
    diagnoses: String,
    statement: String,
}

impl EmbedStructure {
    pub fn new(
        notes: &Notes,
        diagnoses: Option<&Vec<ResolvedDiagnosis>>,
        statement: Option<&str>,
    ) -> Self {
        Self {
            notes: notes.to_markdown(1),
            diagnoses: match diagnoses {
                Some(x) => x
                    .iter()
                    .map(|x| x.diagnosis.to_markdown(1))
                    .collect::<Vec<_>>()
                    .join("\n\n"),
                None => String::new(),
            },
            statement: quote_lines(&statement.map(|x| x.to_owned()).unwrap_or_default()),
        }
    }

    pub fn render(&self) -> Result<String> {
        render_template(EMBED_STRUCTURE, &self).map_err(Error::TemplateError)
    }
}

pub fn quote_lines(content: &str) -> String {
    content
        .lines()
        .map(|l| format!("> {}", l))
        .collect::<Vec<String>>()
        .join("\n")
}

pub async fn get_excerpt(hash: &DocId, db: &DocDb) -> Option<String> {
    let document = match db.get_document(&hash).await {
        Ok(document) => document,
        Err(_) => return None,
    };
    let mut titles: Vec<&str> = vec![];
    let mut hash_for_title = Some(hash);
    loop {
        match hash_for_title {
            Some(hash) => {
                if let Some(title) = db.get_title(hash) {
                    titles.push(title);
                }
                hash_for_title = db.get_parent(hash);
            }
            None => break,
        }
    }
    if !titles.is_empty() {
        format!(
            "# {}\n\n{}\n\n<id:{}>",
            titles.into_iter().rev().collect::<Vec<_>>().join(" > "),
            document.trim(),
            hex::encode(hash)
        )
        .pipe(Some)
    } else {
        format!("{}\n\n<id:{}>", document.trim(), hex::encode(hash)).pipe(Some)
    }
}

pub async fn embed_for_db(text: &str, db: &DocDb, key: &str) -> Result<Array1<N32>> {
    let embedding = embed(&key, text)
        .await?
        .into_iter()
        .map(|x| N32::try_from(x))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| Error::EmbeddingError)?;
    let embedding =
        Array1::from_shape_vec((embedding.len(),), embedding).map_err(|_| Error::EmbeddingError)?;
    db.get_pca_mapped(embedding.view()).to_owned().pipe(Ok)
}

#[cfg(test)]
mod test {
    #[test]
    fn quotes_lines() {
        assert_eq!(
            super::quote_lines("foo\nbar\n\nbaz\n"),
            "> foo\n> bar\n> \n> baz"
        );
    }
}
