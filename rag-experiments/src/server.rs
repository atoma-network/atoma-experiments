use crate::{
    client::EmbeddingClient,
    types::{CreateIndexInput, MetricOptions, QueryInput, QueryResponse, TextToEmbed},
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
use tracing::{error, info, info_span, instrument};

/// Represents the shared state of the application.
///
/// This struct holds the shared resources that need to be accessible
/// across different request handlers in the server.
#[derive(Clone)]
pub struct AppState {
    /// The embedding client wrapped in an Arc<Mutex> for thread-safe access.
    ///
    /// This allows multiple handlers to access and modify the embedding client
    /// concurrently without causing data races.
    embedding_client: Arc<Mutex<EmbeddingClient>>,
}

impl AppState {
    /// Constructor
    pub fn new(client: EmbeddingClient) -> Self {
        AppState {
            embedding_client: Arc::new(Mutex::new(client)),
        }
    }
}

/// Starts the server with the given configuration and embedding client.
///
/// # Arguments
///
/// * `host` - A string slice that holds the host address to bind the server to.
/// * `port` - The port number to bind the server to.
/// * `client` - An instance of `EmbeddingClient` to be used for embedding operations.
///
/// # Returns
///
/// Returns `Ok(())` if the server starts successfully, or an `Error` if there's a problem.
///
/// # Errors
///
/// This function will return an error if:
/// - The host address is invalid and cannot be parsed.
/// - The server fails to bind to the specified address and port.
/// - There's an error while serving the application.
#[instrument(skip_all)]
pub async fn start(host: &str, port: u16, client: EmbeddingClient) -> Result<(), Error> {
    let span = info_span!("start-server");
    let _enter = span.enter();
    info!("Starting server on {}:{}", host, port);
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
    match axum_server::bind(addr)
        .serve(router.into_make_service())
        .await
    {
        Ok(_) => {
            info!("Server started successfully");
            Ok(())
        }
        Err(e) => {
            error!("Error starting server: {}", e);
            Err(e.into())
        }
    }
}

/// Handles the embedding of text and storing it in the specified index.
///
/// This function takes text input, creates an embedding for it, and stores
/// the embedding along with the original text in the specified index.
///
/// # Arguments
///
/// * `app_state` - The shared application state containing the embedding client.
/// * `input` - The input data containing the text to embed and the index name.
///
/// # Returns
///
/// Returns `Ok(Json(()))` if the embedding is successfully created and stored,
/// or an error with an appropriate status code and message if any step fails.
///
/// # Errors
///
/// This function will return an error if:
/// - There's an issue creating the embedding.
/// - There's a problem serializing the input data.
/// - Storing the embedding in the index fails.
#[instrument(skip_all)]
pub async fn embed(
    State(app_state): State<AppState>,
    Json(input): Json<TextToEmbed>,
) -> Result<Json<()>, (StatusCode, String)> {
    let span = info_span!("embed");
    let _enter = span.enter();
    info!("Embedding text, for query with id: {}", input.query_id);
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

/// Handles querying the vector database for similar embeddings.
///
/// This function takes a query input, performs a similarity search in the specified index,
/// and returns the top-k most similar results.
///
/// # Arguments
///
/// * `app_state` - The shared application state containing the embedding client.
/// * `input` - The query input containing the index name, query text, and number of results to return.
///
/// # Returns
///
/// Returns `Ok(Json(Vec<QueryResponse>))` if the query is successful, where `QueryResponse`
/// contains the matched documents and their similarity scores.
///
/// # Errors
///
/// Returns a `(StatusCode, String)` error tuple if:
/// - There's an issue accessing the embedding client.
/// - The query operation fails in the vector database.
///
/// # Example
///
/// ```
/// let query_input = QueryInput {
///     index_name: "my_index".to_string(),
///     query_text: "Sample query".to_string(),
///     top_k: 5,
/// };
/// let result = query(State(app_state), Json(query_input)).await;
/// ```
#[instrument(skip_all)]
pub async fn query(
    State(app_state): State<AppState>,
    Json(input): Json<QueryInput>,
) -> Result<Json<Vec<QueryResponse>>, (StatusCode, String)> {
    let span = info_span!("query");
    let _enter = span.enter();
    info!("Querying index: {}", input.index_name);
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

/// Handles the creation of a new index in the vector database.
///
/// This function takes the index creation input, processes it, and creates a new index
/// in the underlying vector database using the embedding client.
///
/// # Arguments
///
/// * `app_state` - The shared application state containing the embedding client.
/// * `input` - The input data for creating the index, including name, dimension, and metric.
///
/// # Returns
///
/// Returns `Ok(())` if the index is successfully created, or an error with an appropriate
/// status code and message if the creation fails.
///
/// # Errors
///
/// This function will return an error if:
/// - There's an issue accessing the embedding client.
/// - The index creation operation fails in the vector database.
///
/// # Example
///
/// ```
/// let create_index_input = CreateIndexInput {
///     index_name: "my_new_index".to_string(),
///     dimension: 768,
///     metric: Some(MetricOptions::Cosine),
/// };
/// let result = create_index(State(app_state), Json(create_index_input)).await;
/// ```
#[instrument(skip_all)]
pub async fn create_index(
    State(app_state): State<AppState>,
    Json(input): Json<CreateIndexInput>,
) -> Result<(), (StatusCode, String)> {
    let span = info_span!("create_index");
    let _enter = span.enter();
    info!("Creating index: {}", input.index_name);
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
