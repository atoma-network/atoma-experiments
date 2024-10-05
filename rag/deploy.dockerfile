# Use the official Rust image as a parent image
FROM rust:1.70 as builder

# Set the working directory in the container
WORKDIR /usr/src/rag

# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml Cargo.lock ./

# Copy the source code
COPY src ./src

# Build the application
RUN cargo build --release

# Start a new stage for a smaller final image
FROM debian:bullseye-slim

# Install OpenSSL - required for some Rust dependencies
RUN apt-get update && apt-get install -y openssl ca-certificates libssl-dev pkg-config && rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /usr/src/rag/target/release/rag /usr/local/bin/rag

# Set the startup command
CMD ["rag"]

# Expose the port the app runs on
EXPOSE 8081
