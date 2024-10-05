# Atoma-RAG

This is a simple RAG server built using Rust, Axum, the text-embeddings-inference library from HuggingFace (see [repo](https://github.com/huggingface/text-embeddings-inference)) and Pinecone for vector storage.

## Start the server

To start the RAG server, you will need to have Rust and Cargo installed (see [Rust installation guide](https://www.rust-lang.org/tools/install)). You are also required
to have a Pinecone account setup, including an API key and a host vector database. Once you have those, you must fill in a `.env` file, following the `.env.example` example file.

The embedding port and host correspond to the text-embeddings-inference server, which listens to new embeddings requests in the background.

Once you have setup your `.env` file, you can run the server with the following command:

```bash
RUST_LOG=info cargo run --release
```

The server will start and listen for requests on the port you have specified in the `.env` file (in variable `PORT`). 

## Usage

Once the server is running, you can send it requests using the `POST` method and JSON body to the `/search` endpoint.

Example request to embed a new text chunk (assuming the server is running locally on port 8081):

```bash
curl -X POST http://localhost:8081/embed \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "unique_query_id",
    "index_name": "your_index_name",
    "content": "This is the text content you want to embed",
    "topic": "Optional topic",
    "description": "Optional description",
    "source": "Optional source",
    "author": "Optional author",
    "page": 1,
    "date": "2023-04-14"
  }'
```

Example request to query the index (assuming the server is running locally on port 8081):

```bash
curl -X POST http://localhost:8081/query \
  -H "Content-Type: application/json" \
  -d '{
    "index_name": "your_index_name",
    "query_text": "This is the text you want to search for",
    "top_k": 5,
    "score_threshold": 0.5
  }'
```

## Docker

Alternatively, you can build a Docker image for the RAG server, run the following command:

```bash
docker build -t atoma-rag .
```

To run the Docker container, use the following command:

```bash
docker run -p 8081:8081 atoma-rag
```

