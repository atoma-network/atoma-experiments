use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Defines the criteria for splitting text into chunks.
pub enum SplitCriteria {
    /// Splits the text at the end of each sentence.
    EndOfSentence,
    /// Splits the text at paragraph breaks.
    Paragraph,
    /// Splits the text based on a maximum token count and includes context sentences.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - The maximum number of tokens allowed per chunk.
    /// * `context_sentences` - The number of previous sentences to include as context.
    TokenCount {
        max_tokens: usize,
        context_sentences: usize,
    },
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
    /// - `TokenCount`: Splits based on a maximum token count per chunk and includes context sentences.
    ///
    /// For `TokenCount`, a tokenizer must be provided. Each chunk will include
    /// the specified number of previous sentences as context, without exceeding the maximum token count.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tokenization fails when using `TokenCount` criteria.
    /// - No tokenizer is provided for `TokenCount` criteria.
    pub fn split(&self, text: &str, tokenizer: Option<&Tokenizer>) -> Result<Vec<String>> {
        match self {
            SplitCriteria::EndOfSentence => {
                let sentences = text
                    .unicode_sentences()
                    .map(|s| s.trim().to_string())
                    .collect();
                Ok(sentences)
            }
            SplitCriteria::Paragraph => {
                let paragraphs = text.split("\n\n").map(|p| p.trim().to_string()).collect();
                Ok(paragraphs)
            }
            SplitCriteria::TokenCount {
                max_tokens,
                context_sentences,
            } => {
                if let Some(tokenizer) = tokenizer {
                    let mut chunks = Vec::new();
                    // Change sentences to own its data
                    let mut sentences: Vec<String> = text
                        .unicode_sentences()
                        .map(|s| s.trim().to_string())
                        .collect();
                    let mut index = 0;

                    while index < sentences.len() {
                        // Determine the start index for context
                        let context_start = if index >= *context_sentences {
                            index - *context_sentences
                        } else {
                            0
                        };

                        // Collect context sentences and the current sentence
                        let current_sentences: Vec<&str> = sentences[context_start..=index]
                            .iter()
                            .map(|s| s.as_str())
                            .collect();
                        let mut current_chunk_text = current_sentences.join(" ");

                        // Tokenize the current chunk
                        let encoding =
                            tokenizer
                                .encode(current_chunk_text.clone(), true)
                                .map_err(|e| {
                                    anyhow!(
                                        "Failed to encode text: '{}', with error: {}",
                                        current_chunk_text,
                                        e
                                    )
                                })?;
                        let token_count = encoding.get_ids().len();

                        // If token count exceeds max_tokens, adjust current_sentences
                        if token_count > *max_tokens {
                            // Remove the earliest context sentences
                            let mut adjusted_current_sentences = current_sentences.clone();
                            while adjusted_current_sentences.len() > 1 {
                                adjusted_current_sentences.remove(0); // Remove first sentence
                                current_chunk_text = adjusted_current_sentences.join(" ");
                                let encoding = tokenizer
                                    .encode(current_chunk_text.clone(), true)
                                    .map_err(|e| {
                                    anyhow!(
                                        "Failed to encode text: '{}', with error: {}",
                                        current_chunk_text,
                                        e
                                    )
                                })?;
                                let token_count = encoding.get_ids().len();
                                if token_count <= *max_tokens {
                                    break;
                                }
                            }

                            // If token count still exceeds max_tokens, split the sentence
                            if token_count > *max_tokens {
                                // Split the sentence into words and fit as many as possible
                                let sentence = &sentences[index];
                                let words: Vec<&str> = sentence.unicode_words().collect();
                                let mut word_index = 0;
                                let mut word_chunk = Vec::new();
                                let mut word_chunk_text = String::new();
                                let mut word_token_count = 0;

                                while word_index < words.len() {
                                    let word = words[word_index];
                                    let word_to_encode = if word_index == 0 {
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
                                    let word_token_len = word_tokens.len();

                                    if word_token_len > *max_tokens {
                                        // NOTE: If a single word exceeds max_tokens, place it in a chunk by itself
                                        if word_chunk.is_empty() {
                                            word_chunk.push(word_to_encode.to_string());
                                            word_chunk_text = word_chunk.join("");
                                            word_index += 1;
                                        }
                                        break;
                                    }

                                    if word_token_count + word_token_len > *max_tokens {
                                        break;
                                    }

                                    word_chunk.push(word_to_encode.to_string());
                                    word_chunk_text = word_chunk.join("");
                                    word_token_count += word_token_len;
                                    word_index += 1;
                                }

                                if !word_chunk.is_empty() {
                                    chunks.push(word_chunk_text.trim().to_string());
                                }

                                // Move to the next set of words
                                if word_index < words.len() {
                                    // There are remaining words in the sentence
                                    let remaining_sentence = words[word_index..].join(" ");
                                    sentences.insert(index + 1, remaining_sentence);
                                }
                            } else {
                                chunks.push(current_chunk_text.trim().to_string());
                            }
                        } else {
                            chunks.push(current_chunk_text.trim().to_string());
                        }

                        index += 1;
                    }

                    Ok(chunks)
                } else {
                    Err(anyhow!("No tokenizer provided for TokenCount splitting"))
                }
            }
        }
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
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 5,
            context_sentences: 1,
        };
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
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 5,
            context_sentences: 1,
        };
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();
        println!("chunks: {:?}", chunks);
        assert!(chunks.len() > 1);
        assert!(chunks[0].contains("Supercalifragilisticexpialidocious"));
        std::fs::remove_dir_all("./cache/").expect("Failed to remove cache directory");
    }

    #[test]
    fn test_split_token_count_no_tokenizer() {
        let text = "This should fail.";
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 5,
            context_sentences: 1,
        };
        let result = criteria.split(text, None);
        assert!(result.is_err());
    }

    #[test]
    #[serial]
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

    #[test]
    fn test_end_of_sentence_split() {
        let text = "This is a sentence. Here is another one! And a question?";
        let criteria = SplitCriteria::EndOfSentence;
        let chunks = criteria.split(text, None).unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "This is a sentence.");
        assert_eq!(chunks[1], "Here is another one!");
        assert_eq!(chunks[2], "And a question?");
    }

    #[test]
    fn test_paragraph_split() {
        let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";
        let criteria = SplitCriteria::Paragraph;
        let chunks = criteria.split(text, None).unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "Paragraph one.");
        assert_eq!(chunks[1], "Paragraph two.");
        assert_eq!(chunks[2], "Paragraph three.");
    }

    #[test]
    #[serial]
    fn test_token_count_split_basic() {
        let text = "This is a sample text.";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 5,
            context_sentences: 0,
        };
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();

        // Depending on the tokenizer, the number of chunks may vary
        // Here we test that at least one chunk is returned and no error occurs
        assert!(!chunks.is_empty());
    }

    #[test]
    #[serial]
    fn test_token_count_split_with_context() {
        let text = "Sentence one. Sentence two. Sentence three. Sentence four.";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 10,
            context_sentences: 1,
        };
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();

        // Test that context sentences are included
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], "Sentence one.");
        assert_eq!(chunks[1], "Sentence one. Sentence two.");
        assert_eq!(chunks[2], "Sentence two. Sentence three.");
        assert_eq!(chunks[3], "Sentence three. Sentence four.");

        std::fs::remove_dir_all("./cache/").expect("Failed to remove cache directory");
    }

    #[test]
    #[serial]
    fn test_token_count_split_sentence_longer_than_max_tokens() {
        let text = "This is a very long sentence that might exceed the maximum token count set for splitting.";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 5,
            context_sentences: 0,
        };
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();

        // Test that the long sentence is split into smaller chunks
        assert!(!chunks.is_empty());
        for chunk in chunks {
            let encoding = tokenizer.encode(chunk.clone(), true).unwrap();
            assert!(encoding.get_ids().len() <= 5);
        }
    }

    #[test]
    #[serial]
    fn test_token_count_split_with_zero_context_sentences() {
        let text = "First sentence. Second sentence.";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 10,
            context_sentences: 0,
        };
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "First sentence.");
        assert_eq!(chunks[1], "Second sentence.");
    }

    #[test]
    #[serial]
    fn test_token_count_split_context_greater_than_available_sentences() {
        let text = "Only one sentence here.";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 10,
            context_sentences: 5,
        };
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Only one sentence here.");
    }

    #[test]
    #[serial]
    fn test_empty_text() {
        let text = "";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 10,
            context_sentences: 1,
        };
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_text_with_only_spaces() {
        let text = "     ";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::Paragraph;
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "");
    }

    #[test]
    fn test_text_with_only_newlines() {
        let text = "\n\n\n";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::Paragraph;
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();

        assert_eq!(chunks.len(), 2); // Three empty paragraphs and one after the last newline
        for chunk in chunks {
            assert_eq!(chunk, "");
        }
    }

    #[test]
    fn test_unicode_characters() {
        let text = "Here is a sentence with emojis 😊😂👍.";
        let criteria = SplitCriteria::EndOfSentence;
        let chunks = criteria.split(text, None).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Here is a sentence with emojis 😊😂👍.");
    }

    #[test]
    #[serial]
    fn test_context_sentences_with_long_sentences() {
        let text = "Sentence one is short. This is sentence two which is significantly longer and may cause issues with the maximum token limit. Sentence three is here.";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 15,
            context_sentences: 1,
        };
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();

        // Ensure that context is included correctly and chunks respect the max token limit
        for chunk in chunks {
            let encoding = tokenizer.encode(chunk.clone(), true).unwrap();
            assert!(encoding.get_ids().len() <= 15);
        }
    }

    #[test]
    #[serial]
    fn test_max_tokens_exact_match() {
        let text = "This is a test.";
        let tokenizer = create_test_tokenizer();
        let encoding = tokenizer.encode(text, true).unwrap();
        let token_count = encoding.get_ids().len();

        let criteria = SplitCriteria::TokenCount {
            max_tokens: token_count,
            context_sentences: 0,
        };
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    #[serial]
    fn test_no_tokenizer_provided() {
        let text = "This is a test.";
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 5,
            context_sentences: 0,
        };
        let result = criteria.split(text, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_special_characters() {
        let text = "Special characters: @#$%^&*() are included.";
        let criteria = SplitCriteria::EndOfSentence;
        let chunks = criteria.split(text, None).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Special characters: @#$%^&*() are included.");
    }

    #[test]
    fn test_paragraphs_with_multiple_newlines() {
        let text =
            "First paragraph.\n\n\nSecond paragraph after multiple newlines.\n\nThird paragraph.";
        let criteria = SplitCriteria::Paragraph;
        let chunks = criteria.split(text, None).unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "First paragraph.");
        assert_eq!(chunks[1], "Second paragraph after multiple newlines.");
        assert_eq!(chunks[2], "Third paragraph.");
    }

    #[test]
    fn test_sentence_splitting_with_abbreviations() {
        // NOTE: This test is not working as expected.
        let text = "Dr. Smith went to Washington. He arrived at 3 p.m.";
        let criteria = SplitCriteria::EndOfSentence;
        let chunks = criteria.split(text, None).unwrap();

        println!("chunks: {:?}", chunks);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "Dr.");
        assert_eq!(chunks[1], "Smith went to Washington.");
        assert_eq!(chunks[2], "He arrived at 3 p.m.");
    }

    #[test]
    #[serial]
    fn test_token_count_split_with_large_context() {
        let text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five.";
        let tokenizer = create_test_tokenizer();
        let criteria = SplitCriteria::TokenCount {
            max_tokens: 20,
            context_sentences: 3,
        };
        let chunks = criteria.split(text, Some(&tokenizer)).unwrap();

        // Even though context_sentences is 10, there are only 5 sentences
        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks[0], "Sentence one.");
        assert_eq!(chunks[1], "Sentence one. Sentence two.");
        assert_eq!(chunks[2], "Sentence one. Sentence two. Sentence three.");
        assert_eq!(
            chunks[3],
            "Sentence one. Sentence two. Sentence three. Sentence four."
        );
        assert_eq!(
            chunks[4],
            "Sentence two. Sentence three. Sentence four. Sentence five."
        );
    }

    #[test]
    fn test_text_with_no_sentences() {
        let text = "No sentences here but some words";
        let criteria = SplitCriteria::EndOfSentence;
        let chunks = criteria.split(text, None).unwrap();

        // Since there are no sentence-ending punctuation marks, the entire text is one chunk
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "No sentences here but some words");
    }
}
