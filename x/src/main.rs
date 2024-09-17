use anyhow::Result;
use dotenv::dotenv;
use rag::{client::EmbeddingClient, server::start, types::TextToEmbed};
use reqwest::Client;
use std::{
    env,
    hash::{DefaultHasher, Hash, Hasher},
};
use tracing::{error, info, info_span};
use x::note_tweet::parse_note_tweets;

const INDEX_NAME: &str = "atoma-alpha-mistral";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let span = info_span!("x");
    let _enter = span.enter();

    dotenv().expect("Failed to load .env file");
    let username = env::var("USERNAME").expect("USERNAME not set");
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

    let pinecone_api_key = env::var("PINECONE_API_KEY").unwrap();
    let pinecone_host = env::var("PINECONE_HOST").unwrap();
    let note_tweets =
        parse_note_tweets(&env::var("NOTE_TWEET_FILE").expect("NOTE_TWEET_FILE not set"))
            .expect("Failed to parse note tweets json file");

    let host_clone = host.clone();
    let _join_handle = tokio::spawn(async move {
        let client = EmbeddingClient::new(
            embedding_host,
            embedding_port,
            pinecone_api_key,
            pinecone_host,
        )
        .await?;
        start(&host_clone, port, client).await?;
        Ok::<_, anyhow::Error>(())
    });

    let client = Client::new();
    for note_tweet in note_tweets {
        let mut default_hasher = DefaultHasher::new();
        note_tweet.hash(&mut default_hasher);
        let query_id = default_hasher.finish().to_string();
        let text_to_embed = TextToEmbed {
            query_id: query_id.clone(),
            index_name: INDEX_NAME.to_string(),
            content: note_tweet.core.text,
            topic: None,
            description: None,
            source: Some("x".to_string()),
            author: Some(username.clone()),
            page: None,
            date: Some(note_tweet.created_at),
        };

        match client
            .post(format!("http://{}:{}/embed", host, port))
            .json(&text_to_embed)
            .send()
            .await
        {
            Ok(response) => {
                info!("Succesfully embedded result: {:?}", response);
            }
            Err(e) => {
                error!("Error: {:?}", e);
                panic!("Failed to successfully embed the tweet data for query_id: {}, with error: {:?}", query_id, e);
            }
        }
    }

    Ok(())
}
