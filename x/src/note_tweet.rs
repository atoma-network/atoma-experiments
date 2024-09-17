use std::io::Read;
use types::{NoteTweet, NoteTweetContainer};

use anyhow::Result;

/// Parses note tweets from a given file.
///
/// This function reads a file containing note tweet data in a specific JSON format,
/// processes it, and returns a vector of `NoteTweet` objects.
///
/// # Arguments
///
/// * `file_path` - A string slice that holds the path to the file containing note tweet data.
///
/// # Returns
///
/// * `Result<Vec<NoteTweet>>` - A Result containing a vector of `NoteTweet` objects if successful,
///   or an error if any step of the parsing process fails.
///
/// # Errors
///
/// This function will return an error if:
/// * The file cannot be opened or read.
/// * The JSON content is malformed or cannot be parsed.
///
/// # Example
///
/// ```
/// use your_crate_name::parse_note_tweets;
///
/// let note_tweets = parse_note_tweets("path/to/note_tweets.json").expect("Failed to parse note tweets");
/// println!("Parsed {} note tweets", note_tweets.len());
/// ```
pub fn parse_note_tweets(file_path: &str) -> Result<Vec<NoteTweet>> {
    let file = std::fs::File::open(file_path)?;
    let reader = std::io::BufReader::new(file);

    let mut content = String::new();
    std::io::Read::read_to_string(&mut reader.take(u64::MAX), &mut content)?;

    // Remove the "window.YTD.note_tweet.part0 = " prefix
    let json_content = content.trim_start_matches("window.YTD.note_tweet.part0 = ");
    let containers: Vec<NoteTweetContainer> = serde_json::from_str(json_content)?;
    let note_tweets: Vec<NoteTweet> = containers.into_iter().map(|c| c.note_tweet).collect();

    Ok(note_tweets)
}

pub mod types {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Hash, Serialize, Deserialize)]
    pub struct NoteTweetContainer {
        #[serde(rename = "noteTweet")]
        pub note_tweet: NoteTweet,
    }

    #[derive(Debug, Hash, Serialize, Deserialize)]
    pub struct NoteTweet {
        #[serde(rename = "noteTweetId")]
        pub note_tweet_id: String,
        #[serde(rename = "updatedAt")]
        pub updated_at: String,
        lifecycle: Lifecycle,
        #[serde(rename = "createdAt")]
        pub created_at: String,
        pub core: Core,
    }

    #[derive(Debug, Hash, Serialize, Deserialize)]
    pub struct Lifecycle {
        pub value: String,
        pub name: String,
        #[serde(rename = "originalName")]
        pub original_name: String,
        pub annotations: serde_json::Value,
    }

    #[derive(Debug, Hash, Serialize, Deserialize)]
    pub struct Url {
        #[serde(rename = "expandedUrl")]
        pub expanded_url: String,
        #[serde(rename = "toIndex")]
        pub to_index: String,
        #[serde(rename = "shortUrl")]
        pub short_url: String,
        #[serde(rename = "displayUrl")]
        pub display_url: String,
        #[serde(rename = "fromIndex")]
        pub from_index: String,
    }

    #[derive(Debug, Hash, Serialize, Deserialize)]
    pub struct Mention {
        #[serde(rename = "screenName")]
        pub screen_name: String,
        #[serde(rename = "fromIndex")]
        pub from_index: String,
        #[serde(rename = "toIndex")]
        pub to_index: String,
    }

    #[derive(Debug, Hash, Serialize, Deserialize)]
    pub struct StyleTag {
        #[serde(rename = "styleTypes")]
        pub style_types: Vec<StyleType>,
        #[serde(rename = "fromIndex")]
        pub from_index: String,
        #[serde(rename = "toIndex")]
        pub to_index: String,
    }

    #[derive(Debug, Hash, Serialize, Deserialize)]
    pub struct StyleType {
        pub value: String,
        pub name: String,
        #[serde(rename = "originalName")]
        pub original_name: String,
        pub annotations: serde_json::Value,
    }

    #[derive(Debug, Hash, Serialize, Deserialize)]
    pub struct Core {
        pub styletags: Option<Vec<StyleTag>>,
        pub urls: Vec<Url>,
        pub text: String,
        pub mentions: Vec<Mention>,
        pub cashtags: Vec<String>,
        pub hashtags: Vec<String>,
    }
}
