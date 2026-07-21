FROM rust:1.97.1-slim-bookworm

# Install system dependencies used by Cargo's git dependency fetching and common build scripts.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Add the Rust components used by CI, local review, and the dev container post-create step.
RUN rustup component add rustfmt clippy llvm-tools-preview

# Create a non-root user for VS Code and grant ownership of the Rust toolchain.
RUN useradd -m -s /bin/bash -u 1000 vscode \
    && chown -R vscode:vscode /usr/local/cargo /usr/local/rustup

USER vscode
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
WORKDIR /workspace
