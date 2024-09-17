use anyhow::Result;
use std::fs::File;
use std::io::{BufReader, Read};
use types::{Tweet, TweetContainer};

pub fn parse_tweets(file_path: &str) -> Result<Vec<Tweet>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut content = String::new();
    std::io::Read::read_to_string(&mut reader.take(u64::MAX), &mut content)?;

    let json_content = content.trim_start_matches("window.YTD.tweets.part0 = ");

    let containers: Vec<TweetContainer> = serde_json::from_str(json_content)?;

    let tweets: Vec<Tweet> = containers.into_iter().map(|c| c.tweet).collect();

    Ok(tweets)
}

pub mod types {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct TweetContainer {
        pub tweet: Tweet,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Tweet {
        pub edit_info: EditInfo,
        pub retweeted: bool,
        pub source: String,
        pub entities: Entities,
        pub display_text_range: Vec<String>,
        pub favorite_count: String,
        pub id_str: String,
        pub truncated: bool,
        pub retweet_count: String,
        pub id: String,
        #[serde(default)]
        pub possibly_sensitive: bool,
        pub created_at: String,
        pub favorited: bool,
        pub full_text: String,
        pub lang: String,
        #[serde(default)]
        pub in_reply_to_status_id_str: Option<String>,
        #[serde(default)]
        pub in_reply_to_user_id: Option<String>,
        #[serde(default)]
        pub in_reply_to_status_id: Option<String>,
        #[serde(default)]
        pub in_reply_to_screen_name: Option<String>,
        #[serde(default)]
        pub in_reply_to_user_id_str: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct EditInfo {
        pub edit: Option<Edit>,
        pub initial: Option<EditControlInitial>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Edit {
        #[serde(rename = "initialTweetId")]
        pub initial_tweet_id: String,
        #[serde(rename = "editControlInitial")]
        pub edit_control_initial: EditControlInitial,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct EditControlInitial {
        #[serde(rename = "editTweetIds")]
        pub edit_tweet_ids: Vec<String>,
        #[serde(rename = "editableUntil")]
        pub editable_until: String,
        #[serde(rename = "editsRemaining")]
        pub edits_remaining: String,
        #[serde(rename = "isEditEligible")]
        pub is_edit_eligible: bool,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct HashTag {
        pub text: String,
        pub indices: Vec<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Entities {
        pub hashtags: Vec<HashTag>,
        pub symbols: Vec<Symbol>,
        pub user_mentions: Vec<UserMention>,
        pub urls: Vec<Url>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Symbol {
        text: String,
        indices: Vec<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct UserMention {
        pub name: String,
        pub screen_name: String,
        pub indices: Vec<String>,
        pub id_str: String,
        pub id: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Url {
        pub url: String,
        pub expanded_url: String,
        pub display_url: String,
        pub indices: Vec<String>,
    }
}
