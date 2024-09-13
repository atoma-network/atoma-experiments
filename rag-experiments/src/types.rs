use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct TextToEmbed {
    /// Query id
    pub query_id: String,
    /// The index name for Pinecone storage
    pub index_name: String,
    /// The content of the document to embed
    pub content: String,
    /// The source of the document
    pub source: Option<String>,
    /// The author of the document
    pub author: Option<String>,
    /// The page of the document
    pub page: Option<u16>,
    /// The date in which the document was published
    pub date: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub result: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryInput {
    pub index_name: String,
    pub query_text: String,
    pub top_k: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateIndexInput {
    pub index_name: String,
    pub dimension: i32,
    pub metric: Option<MetricOptions>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum MetricOptions {
    Cosine,
    Euclidean,
    Dotproduct,
}
