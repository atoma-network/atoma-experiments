use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;
use unicode_segmentation::UnicodeSegmentation;

// pub struct Corpus {
//     pub original_text: String,
//     pub token_count_per_chunk: usize,
//     pub chunks: Vec<String>,
//     pub in_context_of: Vec<String>,
// }

// impl Corpus {
//     pub fn new(original_text: String) -> Self {
//         let chunks = original_text
//             .split("\n")
//             .map(|chunk| chunk.to_string())
//             .collect();
//         let in_context_of = Vec::new();
//         Self {
//             original_text,
//             chunks,
//             in_context_of: Vec::with_capacity(chunks.len()),
//         }
//     }
// }

/// Defines the criteria for splitting text into chunks.
pub enum SplitCriteria {
    /// Splits the text at the end of each sentence.
    EndOfSentence,
    /// Splits the text at paragraph breaks.
    Paragraph,
    /// Splits the text based on a maximum token count.
    ///
    /// # Arguments
    ///
    /// * `usize` - The maximum number of tokens allowed per chunk.
    TokenCount(usize),
}

impl SplitCriteria {
    /// Splits the given text into chunks based on the specified criteria.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to be split into chunks.
    /// * `tokenizer` - An optional reference to a `Tokenizer` used for token-based splitting.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a `Vec<String>` of text chunks if successful,
    /// or an `Error` if the splitting process fails.
    ///
    /// # Behavior
    ///
    /// The method splits the text differently based on the `SplitCriteria`:
    ///
    /// - `EndOfSentence`: Splits at the end of each sentence.
    /// - `Paragraph`: Splits at paragraph breaks (empty lines).
    /// - `TokenCount(max_tokens)`: Splits based on a maximum token count per chunk.
    ///
    /// For `TokenCount`, a tokenizer must be provided. If a single sentence exceeds
    /// the maximum token count, it will be split into smaller chunks.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tokenization fails when using `TokenCount` criteria.
    /// - No tokenizer is provided for `TokenCount` criteria.
    ///
    /// # Examples
    ///
    /// ```
    /// use your_crate::{SplitCriteria, Tokenizer};
    ///
    /// let text = "This is a sample text. It has multiple sentences. And paragraphs.\n\nThis is a new paragraph.";
    /// let criteria = SplitCriteria::EndOfSentence;
    /// let chunks = criteria.split(text, None).unwrap();
    /// assert_eq!(chunks.len(), 4);
    ///
    /// let criteria = SplitCriteria::Paragraph;
    /// let chunks = criteria.split(text, None).unwrap();
    /// assert_eq!(chunks.len(), 2);
    ///
    /// let tokenizer = Tokenizer::new(); // Initialize your tokenizer
    /// let criteria = SplitCriteria::TokenCount(10);
    /// let chunks = criteria.split(text, Some(&tokenizer)).unwrap();
    /// // Number of chunks will depend on the tokenizer and max_tokens
    /// ```
    pub fn split(&self, text: &str, tokenizer: Option<&Tokenizer>) -> Result<Vec<String>> {
        let mut chunks = Vec::new();

        // Split the text into paragraphs first
        let paragraphs: Vec<&str> = text.split("\n\n").collect();

        for paragraph in paragraphs {
            match self {
                SplitCriteria::EndOfSentence => {
                    for sentence in paragraph.unicode_sentences() {
                        chunks.push(sentence.trim().to_string());
                    }
                }
                SplitCriteria::Paragraph => {
                    chunks.push(paragraph.trim().to_string());
                }
                SplitCriteria::TokenCount(max_tokens) => {
                    if let Some(tokenizer) = tokenizer {
                        let mut current_chunk_tokens = Vec::new();
                        let mut current_tokens = 0;

                        for sentence in paragraph.unicode_sentences() {
                            // Split the sentence into words
                            let words: Vec<&str> = sentence.unicode_words().collect();
                            let mut first_word_in_sentence = true;

                            for word in words {
                                let word_to_encode = if first_word_in_sentence {
                                    word
                                } else {
                                    // Include a leading space
                                    &format!(" {}", word)
                                };

                                // Tokenize the word
                                let encoding =
                                    tokenizer.encode(word_to_encode, false).map_err(|e| {
                                        anyhow!(
                                            "Failed to encode word: '{}', with error: {}",
                                            word_to_encode,
                                            e
                                        )
                                    })?;
                                let word_tokens = encoding.get_ids();
                                let word_token_count = word_tokens.len();

                                // Check if adding the word exceeds max_tokens
                                if current_tokens + word_token_count > *max_tokens {
                                    if !current_chunk_tokens.is_empty() {
                                        // Decode current chunk tokens into text and add to chunks
                                        let chunk_text = tokenizer
                                            .decode(&current_chunk_tokens, true)
                                            .map_err(|e| {
                                                anyhow!(
                                                    "Failed to decode tokens: {:?}, with error: {}",
                                                    current_chunk_tokens,
                                                    e
                                                )
                                            })?;
                                        chunks.push(chunk_text.trim().to_string());
                                        current_chunk_tokens.clear();
                                        current_tokens = 0;
                                    }

                                    if word_token_count > *max_tokens {
                                        // Word is longer than max_tokens, place it in a chunk by itself
                                        current_chunk_tokens.extend_from_slice(word_tokens);

                                        // Decode current chunk tokens into text and add to chunks
                                        let chunk_text = tokenizer
                                            .decode(&current_chunk_tokens, true)
                                            .map_err(|e| {
                                                anyhow!(
                                                    "Failed to decode tokens: {:?}, with error: {}",
                                                    current_chunk_tokens,
                                                    e
                                                )
                                            })?;
                                        chunks.push(chunk_text.trim().to_string());
                                        current_chunk_tokens.clear();
                                        current_tokens = 0;
                                    } else {
                                        // Start a new chunk with the current word
                                        current_chunk_tokens.extend_from_slice(word_tokens);
                                        current_tokens += word_token_count;
                                    }
                                } else {
                                    // Add the word's tokens to the current chunk
                                    current_chunk_tokens.extend_from_slice(word_tokens);
                                    current_tokens += word_token_count;
                                }

                                first_word_in_sentence = false;
                            }
                        }

                        // Add any remaining tokens in current_chunk_tokens to chunks
                        if !current_chunk_tokens.is_empty() {
                            let chunk_text = tokenizer
                                .decode(&current_chunk_tokens, true)
                                .map_err(|e| {
                                anyhow!(
                                    "Failed to decode tokens: {:?}, with error: {}",
                                    current_chunk_tokens,
                                    e
                                )
                            })?;
                            chunks.push(chunk_text.trim().to_string());
                        }
                    } else {
                        return Err(anyhow!(
                            "No tokenizer provided for maximum token count splitting"
                        ));
                    }
                }
            }
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
    use serial_test::serial;

    // Helper function to create a simple tokenizer for testing
    fn create_test_tokenizer() -> Tokenizer {
        let model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string();
        let revision = "main".to_string();
        let api = ApiBuilder::new()
            .with_cache_dir("./cache/".into())
            .build()
            .expect("Failed to create the HF API");

        println!("loading the model weights from {model_id}");
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api
            .get("tokenizer.json")
            .expect("Failed to get tokenizer.json");
        Tokenizer::from_file(tokenizer_filename).expect("Failed to load the tokenizer")
    }

    #[test]
    fn test_split_end_of_sentence() {
        let text = "This is a test. It has three sentences. Last one here.";
        let criteria = SplitCriteria::EndOfSentence;
        let chunks = criteria.split(text, None).unwrap();
        assert_eq!(
            chunks,
            vec![
                "This is a test.",
                "It has three sentences.",
                "Last one here."
            ]
        );
    }

    #[test]
    fn test_split_paragraph() {
        let text = "This is paragraph one.\nStill paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three.";
        let criteria = SplitCriteria::Paragraph;
        let chunks = criteria.split(text, None).unwrap();
        assert_eq!(
            chunks,
            vec![
                "This is paragraph one.\nStill paragraph one.",
                "This is paragraph two.",
                "This is paragraph three."
            ]
        );
    }

    #[test]
    #[serial]
    fn test_split_token_count() {
        let text =
            "This is a long sentence that will be split into multiple chunks based on token count.";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::TokenCount(5);
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();
        assert!(chunks.len() > 1);
        for chunk in chunks.iter() {
            let tokens = tokenizer.encode(chunk.clone(), false).unwrap();
            assert!(tokens.get_ids().len() <= 5);
        }
        println!("chunks: {:?}", chunks);
        std::fs::remove_dir_all("./cache/").expect("Failed to remove cache directory");
    }

    #[test]
    #[serial]
    fn test_split_token_count_long_word() {
        let text = "Supercalifragilisticexpialidocious is a very long word.";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::TokenCount(3);
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();
        println!("chunks: {:?}", chunks);
        assert!(chunks.len() > 1);
        assert!(chunks[0].contains("Supercalifragilisticexpialidocious"));
        std::fs::remove_dir_all("./cache/").expect("Failed to remove cache directory");
    }

    #[test]
    fn test_split_token_count_no_tokenizer() {
        let text = "This should fail.";
        let criteria = SplitCriteria::TokenCount(5);
        let result = criteria.split(text, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_split_empty_text() {
        let text = "";
        let criteria = SplitCriteria::EndOfSentence;
        let chunks = criteria.split(text, None).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_split_unicode() {
        let text = "こんにちは。世界。";
        let criteria = SplitCriteria::EndOfSentence;
        let chunks = criteria.split(text, None).unwrap();
        assert_eq!(chunks, vec!["こんにちは。", "世界。"]);
    }
}
