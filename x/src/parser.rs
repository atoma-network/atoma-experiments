use std::hash::{DefaultHasher, Hash, Hasher};

use anyhow::Result;
use rag::types::TextToEmbed;

use crate::{note_tweet::types::NoteTweet, tweets::types::Tweet};

pub fn parse_tweet_data_to_embed(
    author: String,
    index_name: String,
    note_tweets: Vec<NoteTweet>,
    tweets: Vec<Tweet>,
) -> Result<Vec<TextToEmbed>> {
    let mut text_to_embeds = vec![];
    for note_tweet in note_tweets {
        println!("\n\nNOTE_TWEET: {}", note_tweet.core.text);
        let _tweet = tweets
            .iter()
            .find(|t| {
                let text = t.full_text.split('…').next().unwrap();
                println!("TWEET: {}\n\n", text.get(0..10).unwrap());
                note_tweet.core.text.contains(&text.get(0..10).unwrap())}
            ).expect("Failed ot extract tweet from node tweet");
        let mut default_hasher = DefaultHasher::new();
        note_tweet.hash(&mut default_hasher);
        text_to_embeds.push(TextToEmbed {
            query_id: default_hasher.finish().to_string(),
            index_name: index_name.clone(),
            content: note_tweet.core.text,
            topic: "".to_string(),
            description: None,
            source: Some("x".to_string()),
            author: Some(author.clone()),
            page: None,
            date: Some(note_tweet.created_at),
        });
    }
    Ok(text_to_embeds)
}

#[cfg(test)]
mod tests {
    use crate::{note_tweet::parse_note_tweets, tweets::parse_tweets};

    use super::*;

    #[test]
    fn test_parse_tweet_data_to_embed() {
        dotenv::dotenv().unwrap();
        let note_tweets = parse_note_tweets(&std::env::var("NOTE_TWEET_FILE").unwrap()).unwrap();
        let tweets = parse_tweets(&std::env::var("TWEETS_FILE").unwrap()).unwrap();
        let text_to_embeds = parse_tweet_data_to_embed(
            "Twen1Ack".to_string(),
            "test".to_string(),
            note_tweets,
            tweets,
        )
        .unwrap();
        println!("{:?}", text_to_embeds);
    }
}
