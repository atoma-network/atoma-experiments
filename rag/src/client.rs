use std::collections::BTreeMap;

use anyhow::Result;
use pinecone_sdk::{
    models::{Cloud, DeletionProtection, Kind, Metadata, Metric, Value, Vector, WaitPolicy},
    pinecone::{PineconeClient, PineconeClientConfig},
};
use reqwest::Client;
use serde_json::json;
use tracing::{debug, error, info, info_span, instrument, Span};

use crate::types::QueryResponse;

const CURRENT_NAME_SPACE: &str = "atoma-alpha";

/// A client for managing embeddings and interacting with Pinecone vector database.
///
/// This struct provides methods for creating embeddings, storing them in Pinecone,
/// creating indexes, and querying the vector database.
pub struct EmbeddingClient {
    /// Counter for generating unique IDs for stored embeddings.
    pub counter: usize,
    /// HTTP client for making requests to the embedding service.
    pub embedding_client: Client,
    /// Client for interacting with the Pinecone API.
    pub pinecone_client: PineconeClient,
    /// Host address of the embedding service.
    pub host: String,
    /// Port number of the embedding service.
    pub port: u16,
    /// Tracing span for logging and debugging.
    pub span: Span,
}

impl EmbeddingClient {
    /// Constructor
    pub async fn new(host: String, port: u16) -> Result<Self> {
        let span = info_span!("embedding_client");
        let cloned_span = span.clone();
        let _enter = span.enter();
        let pinecone_api_key = std::env::var("PINECONE_API_KEY").expect("PINECONE_API_KEY not set");
        let config = PineconeClientConfig {
            api_key: Some(pinecone_api_key),
            ..Default::default()
        };
        let pinecone_client = match config.client() {
            Ok(client) => client,
            Err(e) => {
                error!("Failed to create Pinecone client: {}", e);
                return Err(anyhow::anyhow!("Failed to create Pinecone client: {}", e));
            }
        };
        match pinecone_client.list_indexes().await {
            Ok(indexes) => {
                info!("Client indexes: {:?}", indexes);
                indexes
            }
            Err(e) => {
                error!("Failed to list indexes: {}", e);
                return Err(anyhow::anyhow!("Failed to list indexes: {}", e));
            }
        };
        Ok(Self {
            counter: 0,
            embedding_client: Client::new(),
            pinecone_client,
            host,
            port,
            span: cloned_span,
        })
    }

    /// Creates an embedding for the given text using the embedding service.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to be embedded.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of 32-bit floating-point numbers
    /// representing the embedding if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The HTTP request to the embedding service fails.
    /// - The response cannot be parsed as a vector of f32 values.
    #[instrument(skip_all)]
    pub async fn create_embedding(&self, text: String) -> Result<Vec<f32>> {
        let _enter = self.span.enter();
        let input = json!({ "input": text });
        info!("Posting to embedding client");
        let response = match self
            .embedding_client
            .post(format!("http://{}:{}/embed", self.host, self.port))
            .json(&input)
            .send()
            .await
        {
            Ok(res) => res,
            Err(e) => {
                error!("Error posting to embedding client: {:?}", e);
                return Err(anyhow::anyhow!(
                    "Error posting to embedding client: {:?}",
                    e
                ));
            }
        };
        debug!("Response: {:?} for text = {}", response, text);
        let embedding = match response.json::<Vec<f32>>().await {
            Ok(embedding) => embedding,
            Err(e) => {
                error!("Error parsing embedding: {:?}", e);
                return Err(anyhow::anyhow!("Error parsing embedding: {:?}", e));
            }
        };
        info!("Embedding: {:?}", embedding);
        Ok(embedding)
    }

    /// Stores an embedding in the specified Pinecone index.
    ///
    /// # Arguments
    ///
    /// * `original_text` - The original text associated with the embedding.
    /// * `embedding` - The vector representation of the text to be stored.
    /// * `index_name` - The name of the Pinecone index to store the embedding in.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the embedding is successfully stored, or an `Err` if an error occurs.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The Pinecone index cannot be retrieved.
    /// - The upsert operation to the Pinecone index fails.
    ///
    /// # Notes
    ///
    /// This method increments an internal counter to generate unique IDs for each stored embedding.
    /// The embedding is stored with metadata containing the original text.
    #[instrument(skip_all)]
    pub async fn store_embedding(
        &mut self,
        original_text: String,
        embedding: Vec<f32>,
        index_name: &str,
    ) -> Result<()> {
        let _enter = self.span.enter();
        info!("Storing embedding");
        let mut index = self.pinecone_client.index(index_name).await?;
        let metadata: Metadata = Metadata {
            fields: BTreeMap::from_iter(vec![(
                "text".to_string(),
                Value {
                    kind: Some(Kind::StringValue(original_text)),
                },
            )]),
        };
        let vector = Vector {
            id: format!("{}", self.counter),
            values: embedding,
            sparse_values: None,
            metadata: Some(metadata),
        };
        match index.upsert(&[vector], &CURRENT_NAME_SPACE.into()).await {
            Ok(result) => {
                info!(
                    "Response successful, with insertions: {:?}",
                    result.upserted_count
                );
                self.counter += 1;
                Ok(())
            }
            Err(e) => {
                error!("Error storing embedding: {:?}", e);
                Err(anyhow::anyhow!("Error storing embedding: {:?}", e))
            }
        }
    }

    /// Creates a new serverless index in Pinecone.
    ///
    /// # Arguments
    ///
    /// * `index_name` - The name of the index to create.
    /// * `dimension` - The dimension of the vectors to be stored in the index.
    /// * `metric` - Optional similarity metric to use. Defaults to Cosine similarity if not provided.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the index is successfully created, or an `Err` if an error occurs.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The Pinecone API request fails.
    /// - There's an issue with creating the serverless index.
    ///
    /// # Notes
    ///
    /// - The index is created in the AWS us-east-1 region.
    /// - Deletion protection is enabled for the created index.
    /// - The function uses a no-wait policy, meaning it returns immediately after initiating index creation.
    #[instrument(skip_all)]
    pub async fn create_index(
        &mut self,
        index_name: &str,
        dimension: i32,
        metric: Option<Metric>,
    ) -> Result<()> {
        let _enter = self.span.enter();
        info!("Creating index");
        let region = "us-east-1";
        let metric = metric.unwrap_or(Metric::Cosine);
        match self
            .pinecone_client
            .create_serverless_index(
                index_name,
                dimension,
                metric,
                Cloud::Aws,
                region,
                DeletionProtection::Enabled,
                WaitPolicy::NoWait,
            )
            .await
        {
            Ok(result) => {
                info!("Index created: {:?}", result);
                Ok(())
            }
            Err(e) => {
                error!("Error creating index: {:?}", e);
                Err(anyhow::anyhow!("Error creating index: {:?}", e))
            }
        }
    }

    /// Queries the Pinecone index with a given input and returns the most similar results.
    ///
    /// # Arguments
    ///
    /// * `query` - The input text to query against the index.
    /// * `index_name` - The name of the Pinecone index to query.
    /// * `top_k` - Optional number of top results to return. Defaults to 10 if not specified.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of `QueryResponse` structs if successful.
    /// Each `QueryResponse` contains the similarity score, embedding vector, and original text.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The Pinecone index cannot be retrieved.
    /// - Creating an embedding for the query fails.
    /// - Querying the Pinecone index fails.
    /// - The metadata in the response doesn't contain the expected text field.
    ///
    /// # Panics
    ///
    /// This function will panic if the metadata in the response doesn't contain the expected text field.
    #[instrument(skip_all)]
    pub async fn query(
        &self,
        query: String,
        index_name: &str,
        top_k: Option<u32>,
    ) -> Result<Vec<QueryResponse>> {
        let _enter = self.span.enter();
        info!("Retrieving index");
        let mut index = match self.pinecone_client.index(index_name).await {
            Ok(index) => index,
            Err(e) => {
                error!("Error retrieving index: {:?}", e);
                return Err(anyhow::anyhow!("Error retrieving index: {:?}", e));
            }
        };
        let top_k = top_k.unwrap_or(10);
        let query_vector = match self.create_embedding(query).await {
            Ok(embedding) => embedding,
            Err(e) => {
                error!("Error creating embedding: {:?}", e);
                return Err(anyhow::anyhow!("Error creating embedding: {:?}", e));
            }
        };
        let response = match index
            .query_by_value(
                query_vector,
                None,
                top_k,
                &CURRENT_NAME_SPACE.into(),
                None,
                None,
                Some(true),
            )
            .await
        {
            Ok(response) => response,
            Err(e) => {
                error!("Error querying index: {:?}", e);
                return Err(anyhow::anyhow!("Error querying index: {:?}", e));
            }
        };
        let query_response = response
            .matches
            .iter()
            .map(|match_| {
                let text = match match_.metadata.as_ref().unwrap().fields.get("text") {
                    Some(Value {
                        kind: Some(Kind::StringValue(text)),
                        ..
                    }) => text.to_string(),
                    _ => panic!("No text found in metadata"),
                };
                QueryResponse {
                    score: match_.score,
                    embedding: match_.values.clone(),
                    text,
                }
            })
            .collect::<Vec<_>>();
        Ok(query_response)
    }
}
