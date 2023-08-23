use itertools::Itertools;
use std::collections::HashSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tap::Pipe;

use super::super::utils::embed_for_db;
use crate::docdb::{DocDb, DocId};

#[derive(Debug, Clone, Default, JsonSchema, Deserialize, Serialize)]
pub struct CandidateDiagnosis {
    #[schemars(description = "Name of the diagnosis disease or condition.")]
    pub name: String,
    #[schemars(
        description = "What in the patient notes support this diagnosis? 30 words or less."
    )]
    pub reasoning_for: String,
    #[schemars(
        description = "What in the patient notes that contradict this diagnosis? 30 words or less."
    )]
    pub reasoning_against: String,
}

impl CandidateDiagnosis {
    pub fn to_markdown(&self, depth: usize) -> String {
        let depth = "#".repeat(depth);
        let title = format!("{}# {}", depth, &self.name);
        let mut parts: Vec<&str> = vec![&title];
        if !self.reasoning_for.is_empty() {
            parts.push(&self.reasoning_for)
        }
        if !self.reasoning_against.is_empty() {
            parts.push(&self.reasoning_against)
        }
        return parts.join("\n\n");
    }
}

#[derive(Debug, Default, JsonSchema, Deserialize)]
pub struct CandidateDiagnoses {
    #[schemars(description = "Plausible diagnoses.")]
    pub diagnoses: Vec<CandidateDiagnosis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedDiagnosis {
    pub doc_hash: DocId,
    pub diagnosis: CandidateDiagnosis,
    pub refined: Option<String>,
}

impl ResolvedDiagnosis {
    pub fn to_markdown(&self, depth: usize) -> String {
        match &self.refined {
            Some(refined) => {
                let depth = "#".repeat(depth);
                format!("{}# {}\n\n{}", depth, &self.diagnosis.name, refined)
            }
            None => self.diagnosis.to_markdown(depth),
        }
    }
}

pub async fn find_diagnosis_doc(
    candidate_diagnosis: &CandidateDiagnosis,
    db: &DocDb,
    key: &str,
) -> Option<ResolvedDiagnosis> {
    let embedding = embed_for_db(candidate_diagnosis.to_markdown(0).as_str(), db, key)
        .await
        .ok()?;
    let filter = db
        .get_is_introduction()
        .union(db.get_is_symptoms())
        .map(|x| x.clone())
        .collect::<HashSet<_>>()
        .pipe(Some);
    let hashes = db.get_similar(embedding.view(), 8, filter.as_ref());
    let hashes_count = hashes
        .into_iter()
        .map(|x| {
            let mut hash = &x;
            loop {
                if db.get_is_diagnosis().contains(hash) {
                    return Some(hash.to_owned());
                }
                hash = if let Some(hash) = db.get_parent(hash) {
                    hash
                } else {
                    return None;
                };
            }
        })
        .flatten()
        .counts();
    let mut hashes_count = hashes_count.into_iter().collect::<Vec<_>>();
    // `y.cmp(x)` for descending order
    hashes_count.sort_by(|(_, x), (_, y)| y.cmp(x));
    let (hash, _) = hashes_count.first()?;
    let name = db.get_title(hash)?.to_string();
    Some(ResolvedDiagnosis {
        doc_hash: hash.to_owned(),
        diagnosis: CandidateDiagnosis {
            name,
            reasoning_for: candidate_diagnosis.reasoning_for.clone(),
            reasoning_against: candidate_diagnosis.reasoning_against.clone(),
        },
        refined: None,
    })
}

pub fn dedup_diagnoses(diagnoses: Vec<ResolvedDiagnosis>) -> Vec<ResolvedDiagnosis> {
    let mut seen: HashSet<DocId> = HashSet::new();
    let mut deduped: Vec<ResolvedDiagnosis> = Vec::new();
    for diagnosis in diagnoses {
        if seen.contains(&diagnosis.doc_hash) {
            continue;
        }
        seen.insert(diagnosis.doc_hash.clone());
        deduped.push(diagnosis);
    }
    deduped
}
