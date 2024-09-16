use anyhow::Result;
use dotenv::dotenv;
use rag::{client::EmbeddingClient, server::start};
use std::env;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    dotenv().expect("Failed to load .env file");

    // Get host and port from environment variables or use defaults
    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8081);
    let embedding_host = env::var("EMBEDDING_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let embedding_port = env::var("EMBEDDING_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    // Initialize your EmbeddingClient here
    // For example:
    // let client = EmbeddingClient::new(/* parameters */);

    info!("Starting server on {}:{}", host, port);

    let client = EmbeddingClient::new(embedding_host, embedding_port).await?;
    // Start the server
    start(&host, port, client).await?;

    Ok(())
}
