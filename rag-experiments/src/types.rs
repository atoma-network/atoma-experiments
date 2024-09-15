use serde::{Deserialize, Serialize};

/// Represents a text document to be embedded
#[derive(Debug, Deserialize, Serialize)]
pub struct TextToEmbed {
    /// Unique identifier for the query
    pub query_id: String,
    /// The name of the index in Pinecone storage
    pub index_name: String,
    /// The actual text content to be embedded
    pub content: String,
    /// Optional source of the document
    pub source: Option<String>,
    /// Optional author of the document
    pub author: Option<String>,
    /// Optional page number of the document
    pub page: Option<u16>,
    /// Optional publication date of the document
    pub date: Option<String>,
}

/// Input parameters for querying the index
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryInput {
    /// The name of the index to query
    pub index_name: String,
    /// The text to search for in the index
    pub query_text: String,
    /// Optional number of top results to return
    pub top_k: Option<u32>,
}

/// Represents a single query response item
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QueryResponse {
    /// Similarity score of the result
    pub score: f32,
    /// Vector representation of the text
    pub embedding: Vec<f32>,
    /// The actual text content of the result
    pub text: String,
}

/// Input parameters for creating a new index
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateIndexInput {
    /// The name of the index to create
    pub index_name: String,
    /// The dimensionality of the vectors in the index
    pub dimension: i32,
    /// Optional similarity metric to use for the index
    pub metric: Option<MetricOptions>,
}

/// Available similarity metrics for index creation
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum MetricOptions {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance
    Euclidean,
    /// Dot product
    Dotproduct,
}
