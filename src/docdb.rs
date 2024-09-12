//! An in-memory document database with vector embeddings lookup.

use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::io;

use ndarray::{Array2, ArrayView1, CowArray, Ix1};
use noisy_float::prelude::{n32, N32};
use npyz::NpyFile;
use tap::Pipe;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("array data shape is invalid")]
    ArrayShape,
    #[error("array format is invalid: {0}")]
    ArrayRaeding(io::Error),
    #[error("ID format is invalid: {0}")]
    Id(hex::FromHexError),
    #[error("array values must not be NaN")]
    NotNan,
    #[error("record format is invalid: {0}")]
    Record(&'static str),
    #[error("document not available: {0}")]
    DocumentNotAvailable(#[from] reqwest::Error),
}

type Result<T> = core::result::Result<T, Error>;

pub type DocId = [u8; 16];

fn decode_doc_id(data: &[u8]) -> Result<DocId> {
    let mut id = [0u8; 16];
    hex::decode_to_slice(data, &mut id[..]).map_err(Error::Id)?;
    Ok(id)
}

/// The document database data.
#[derive(Debug, Default)]
pub struct DocDb {
    origin: String,
    embeddings: Array2<N32>,
    embeddings_pca_mapping: Option<Array2<N32>>,
    embeddings_id: Vec<DocId>,
    parents: HashMap<DocId, DocId>,
    titles: HashMap<DocId, String>,
    urls: HashMap<DocId, String>,
    is_introduction: HashSet<DocId>,
    is_condition: HashSet<DocId>,
    is_symptoms: HashSet<DocId>,
}

fn array2_from_npy<T: npyz::Deserialize>(npy_data: NpyFile<&[u8]>) -> Result<Array2<T>> {
    use ndarray::ShapeBuilder;
    let shape = match npy_data.shape()[..] {
        [i1, i2] => [i1 as usize, i2 as usize],
        _ => Err(Error::ArrayShape)?,
    };
    let true_shape = shape.set_f(npy_data.order() == npyz::Order::Fortran);
    ndarray::Array2::from_shape_vec(true_shape, npy_data.into_vec::<T>().unwrap())
        .map_err(|_| Error::ArrayShape)
}

impl DocDb {
    /// Build a new database with the provided resources.
    ///
    /// The resources are bytes for the embeddings and metadata. Each document
    /// is represented by a [`DocId`]. The document contents aren't stored in
    /// the database, but are fetched from the URL.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        origin: String,
        embeddings: &[u8],
        embeddings_pca_mapping: Option<&[u8]>,
        embeddings_id: &[u8],
        parents: &[u8],
        titles: &[u8],
        urls: &[u8],
        is_introduction: &[u8],
        is_condition: &[u8],
        is_symptoms: &[u8],
    ) -> Result<DocDb> {
        let embeddings: Array2<f32> =
            array2_from_npy(NpyFile::new(embeddings).map_err(Error::ArrayRaeding)?)?;
        let embeddings: Array2<N32> = if embeddings.iter().any(|x| x.is_nan()) {
            return Err(Error::NotNan);
        } else {
            // NOTE: asserts the values are non NaN only in debug builds
            embeddings.mapv(n32)
        };

        let embeddings_pca_mapping: Option<Array2<N32>> =
            if let Some(embeddings_pca_mapping) = embeddings_pca_mapping {
                let embeddings_pca_mapping: Array2<f32> = array2_from_npy(
                    NpyFile::new(embeddings_pca_mapping).map_err(Error::ArrayRaeding)?,
                )?;
                if embeddings_pca_mapping.iter().any(|x| x.is_nan()) {
                    return Err(Error::NotNan);
                } else {
                    // NOTE: asserts the values are non NaN only in debug builds
                    embeddings_pca_mapping.mapv(n32).pipe(Some)
                }
            } else {
                None
            };

        let embeddings_id: Vec<DocId> = embeddings_id
            .split(|&x| x == 0x0a)
            .filter(|x| !x.is_empty())
            .map(decode_doc_id)
            .collect::<Result<Vec<_>>>()?;

        if embeddings_id.len() != embeddings.shape()[0] {
            return Err(Error::ArrayShape);
        }

        let parents: HashMap<DocId, DocId> = parents
            .split(|&x| x == 0x0a)
            .filter(|x| !x.is_empty())
            .map(|x| {
                x.splitn(2, |&x| x == 0x09)
                    .collect::<Vec<&[u8]>>()
                    .pipe(<[&[u8]; 2]>::try_from)
                    .map_err(|_| Error::Record("parent line lacks two columns"))
            })
            .map(|x| match x {
                Ok([id, parent]) => Ok((decode_doc_id(id)?, decode_doc_id(parent)?)),
                Err(x) => Err(x),
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let titles: HashMap<DocId, String> = titles
            .split(|&x| x == 0x0a)
            .filter(|x| !x.is_empty())
            .map(|x| {
                x.splitn(2, |&x| x == 0x09)
                    .collect::<Vec<&[u8]>>()
                    .pipe(<[&[u8]; 2]>::try_from)
                    .map_err(|_| Error::Record("title line lacks two columns"))
            })
            .map(|x| match x {
                Ok([id, title]) => Ok((
                    decode_doc_id(id)?,
                    String::from_utf8(title.to_vec())
                        .map_err(|_| Error::Record("title line isn't a valid string"))?,
                )),
                Err(x) => Err(x),
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let urls: HashMap<DocId, String> = urls
            .split(|&x| x == 0x0a)
            .filter(|x| !x.is_empty())
            .map(|x| {
                x.splitn(2, |&x| x == 0x09)
                    .collect::<Vec<&[u8]>>()
                    .pipe(<[&[u8]; 2]>::try_from)
                    .map_err(|_| Error::Record("url line lacks two columns"))
            })
            .map(|x| match x {
                Ok([id, title]) => Ok((
                    decode_doc_id(id)?,
                    String::from_utf8(title.to_vec())
                        .map_err(|_| Error::Record("url line isn't a valid string"))?,
                )),
                Err(x) => Err(x),
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let is_introduction: HashSet<DocId> = is_introduction
            .split(|&x| x == 0x0a)
            .filter(|x| !x.is_empty())
            .map(decode_doc_id)
            .collect::<Result<HashSet<_>>>()?;

        let is_condition: HashSet<DocId> = is_condition
            .split(|&x| x == 0x0a)
            .filter(|x| !x.is_empty())
            .map(decode_doc_id)
            .collect::<Result<HashSet<_>>>()?;

        let is_symptoms: HashSet<DocId> = is_symptoms
            .split(|&x| x == 0x0a)
            .filter(|x| !x.is_empty())
            .map(decode_doc_id)
            .collect::<Result<HashSet<_>>>()?;

        Ok(DocDb {
            origin,
            embeddings,
            embeddings_pca_mapping,
            embeddings_id,
            parents,
            titles,
            urls,
            is_introduction,
            is_condition,
            is_symptoms,
        })
    }

    /// Get up to `n` IDs for the documents with embeddings most similar to
    /// `query`.
    ///
    /// If `filter` is provided, only documents with IDs in `filter` are
    /// considered.
    pub fn get_similar(
        &self,
        query: ArrayView1<N32>,
        n: usize,
        filter: Option<&HashSet<DocId>>,
    ) -> Vec<DocId> {
        let mut similarities = self
            .embeddings
            .dot(&query)
            .into_iter()
            .zip(&self.embeddings_id)
            .filter(|(_, x)| match filter {
                Some(filter) => filter.contains(*x),
                None => true,
            })
            .collect::<Vec<_>>();
        // `y.cmp(x)` for descending order
        similarities.sort_by(|(x, _), (y, _)| y.cmp(x));
        similarities
            .into_iter()
            .take(n)
            .map(|(_, x)| x.to_owned())
            .collect()
    }

    /// Get the PCA-mapped version of the embedding `query`.
    pub fn get_pca_mapped<'a>(&self, query: ArrayView1<'a, N32>) -> CowArray<'a, N32, Ix1> {
        if let Some(mapping) = &self.embeddings_pca_mapping {
            CowArray::from(query.dot(mapping))
        } else {
            CowArray::from(query)
        }
    }

    /// Get the contents of the document with `id` by making a request to
    /// the document's URL.
    pub async fn get_document(&self, id: &DocId) -> Result<String> {
        let id = hex::encode(id);
        let path = id
            .chars()
            .take(3)
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join("/");
        let url = format!("{}/db/documents/{}/{}.md", self.origin, path, id);
        let response = reqwest::get(&url)
            .await
            .map_err(Error::DocumentNotAvailable)?;
        response.text().await.unwrap().pipe(Ok)
    }

    /// Get the title of the document with `id`.
    pub fn get_title(&self, id: &DocId) -> Option<&str> {
        self.titles.get(id).map(|x| x.as_str())
    }

    /// Get the url of the document with `id`.
    pub fn get_url(&self, id: &DocId) -> Option<&str> {
        self.urls.get(id).map(|x| x.as_str())
    }

    /// Get the parent `id` of the document with `id`.
    pub fn get_parent(&self, id: &DocId) -> Option<&DocId> {
        self.parents.get(id)
    }

    /// Does the document with `id` describe a condition?
    pub fn get_is_diagnosis(&self) -> &HashSet<DocId> {
        &self.is_condition
    }

    /// Is the document with `id` a section about symptoms for a condition?
    pub fn get_is_symptoms(&self) -> &HashSet<DocId> {
        &self.is_symptoms
    }

    /// Is the document with `id` an introduction section?
    pub fn get_is_introduction(&self) -> &HashSet<DocId> {
        &self.is_introduction
    }
}

#[cfg(test)]
mod test {
    use ndarray::{array, Array1};
    use noisy_float::prelude::n32;

    use super::*;

    #[test]
    fn document_db_gets_similar() {
        let query: Array1<N32> = array![1.0, 0.0].mapv(n32);
        let expected: Vec<DocId> = vec![[0x02; 16], [0x03; 16]];
        let actual = DocDb {
            embeddings: array![[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].mapv(n32),
            embeddings_id: vec![[0x01; 16], [0x02; 16], [0x03; 16]],
            ..Default::default()
        }
        .get_similar(query.view(), expected.len(), None);
        assert_eq!(expected, actual);
    }

    /// DocumentDb gets most similar IDs that are filtered.
    #[test]
    fn document_db_gets_similar_filtered() {
        let query: Array1<N32> = array![1.0, 0.0].mapv(n32);
        let filter: HashSet<DocId> = vec![[0x01; 16], [0x2; 16]].into_iter().collect();
        let expected: Vec<DocId> = vec![[0x02; 16], [0x01; 16]];
        let actual = DocDb {
            embeddings: array![[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].mapv(n32),
            embeddings_id: vec![[0x01; 16], [0x02; 16], [0x03; 16]],
            ..Default::default()
        }
        .get_similar(query.view(), expected.len(), Some(&filter));
        assert_eq!(expected, actual);
    }

    #[test]
    fn document_db_gets_pca_mapped() {
        let query: Array1<N32> = array![1.0, 0.0, 2.0].mapv(n32);
        let expected: Array1<N32> = array![0.0, 1.0].mapv(n32);
        let actual = DocDb {
            embeddings_pca_mapping: Some(array![[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]].mapv(n32)),
            ..Default::default()
        }
        .get_pca_mapped(query.view());
        assert_eq!(expected, actual);
    }

    #[test]
    fn document_db_gets_pca_mapped_no_mapping() {
        let query: Array1<N32> = array![1.0, 0.0].mapv(n32);
        let expected: Array1<N32> = array![1.0, 0.0].mapv(n32);
        let actual = DocDb::default().get_pca_mapped(query.view());
        assert_eq!(expected, actual);
    }
}
