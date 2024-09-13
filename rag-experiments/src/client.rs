use std::collections::BTreeMap;

use anyhow::Result;
use pinecone_sdk::{
    models::{Cloud, DeletionProtection, Kind, Metadata, Metric, Value, Vector, WaitPolicy},
    pinecone::{PineconeClient, PineconeClientConfig},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{error, info, info_span, instrument, Span};

const CURRENT_NAME_SPACE: &str = "atoma-alpha";

pub struct EmbeddingClient {
    pub counter: usize,
    pub embedding_client: Client,
    pub pinecone_client: PineconeClient,
    pub host: String,
    pub port: u16,
    pub span: Span,
}

impl EmbeddingClient {
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
        let indexes = match pinecone_client.list_indexes().await {
            Ok(indexes) => {
                info!("Client indexes: {:?}", indexes);
                indexes
            }
            Err(e) => {
                error!("Failed to list indexes: {}", e);
                return Err(anyhow::anyhow!("Failed to list indexes: {}", e));
            }
        };
        info!("Client indexes: {:?}", indexes);
        Ok(Self {
            counter: 0,
            embedding_client: Client::new(),
            pinecone_client,
            host,
            port,
            span: cloned_span,
        })
    }

    #[instrument(skip_all)]
    pub async fn create_embedding(&self, text: String) -> Result<Vec<f32>> {
        let _enter = self.span.enter();
        info!("Posting to embedding client");
        let response = self
            .embedding_client
            .post(format!("http://{}:{}/embed", self.host, self.port))
            .json(&text)
            .send()
            .await?;
        let embedding = response.json::<Vec<f32>>().await?;
        info!("Embedding: {:?}", embedding);
        Ok(embedding)
    }

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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QueryResponse {
    pub score: f32,
    pub embedding: Vec<f32>,
    pub text: String,
}
