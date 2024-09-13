use crate::{
    client::{EmbeddingClient, QueryResponse},
    types::{CreateIndexInput, MetricOptions, QueryInput, TextToEmbed},
};
use anyhow::Error;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use pinecone_sdk::models::Metric;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::error;

#[derive(Clone)]
pub struct AppState {
    embedding_client: Arc<Mutex<EmbeddingClient>>,
}

impl AppState {
    pub fn new(client: EmbeddingClient) -> Self {
        AppState {
            embedding_client: Arc::new(Mutex::new(client)),
        }
    }
}

pub async fn start(host: &str, port: u16, client: EmbeddingClient) -> Result<(), Error> {
    let app_state = AppState::new(client);
    let router = Router::new()
        .route("/create_index", post(create_index))
        .route("/embed", post(embed))
        .route("/query", get(query))
        .with_state(app_state);

    let ip: IpAddr = host
        .parse()
        .map_err(|_| Error::msg("Invalid host address"))?;
    let addr = SocketAddr::new(ip, port);
    axum_server::bind(addr)
        .serve(router.into_make_service())
        .await?;
    Ok(())
}

pub async fn embed(
    State(app_state): State<AppState>,
    Json(input): Json<TextToEmbed>,
) -> Result<Json<()>, (StatusCode, String)> {
    let mut embedding_client = app_state.embedding_client.lock().await;
    let embedding = match embedding_client
        .create_embedding(
            serde_json::to_string(&input)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
        )
        .await
    {
        Ok(embedding) => embedding,
        Err(e) => {
            error!("Error creating embedding: {}", e);
            return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
    };
    let original_text = serde_json::to_string(&input)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    match embedding_client
        .store_embedding(original_text, embedding, &input.index_name)
        .await
    {
        Ok(_) => Ok(Json(())),
        Err(e) => {
            error!("Error storing embedding: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

pub async fn query(
    State(app_state): State<AppState>,
    Json(input): Json<QueryInput>,
) -> Result<Json<Vec<QueryResponse>>, (StatusCode, String)> {
    let QueryInput {
        index_name,
        query_text,
        top_k,
    } = input;
    let embedding_client = app_state.embedding_client.lock().await;
    let query_response = match embedding_client.query(query_text, &index_name, top_k).await {
        Ok(query_response) => query_response,
        Err(e) => {
            error!("Error querying: {}", e);
            return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
    };
    Ok(Json(query_response))
}

pub async fn create_index(
    State(app_state): State<AppState>,
    Json(input): Json<CreateIndexInput>,
) -> Result<(), (StatusCode, String)> {
    let CreateIndexInput {
        index_name,
        dimension,
        metric,
    } = input;
    let metric = metric.map(|m| match m {
        MetricOptions::Cosine => Metric::Cosine,
        MetricOptions::Euclidean => Metric::Euclidean,
        MetricOptions::Dotproduct => Metric::Dotproduct,
    });
    let mut embedding_client = app_state.embedding_client.lock().await;
    embedding_client
        .create_index(&index_name, dimension, metric)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(())
}
