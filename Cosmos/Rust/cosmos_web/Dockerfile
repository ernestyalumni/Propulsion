# syntax=docker/dockerfile:1
# Cosmos Integrator Lab in one image: compile cosmos_web to WebAssembly, build
# the TypeScript viewer, serve dist/ with nginx. The build context is the
# repository root, because cosmos_numerical depends on Stunticons/wildrider;
# Dockerfile.dockerignore narrows it to those two directories:
#
#   docker build -f Cosmos/Rust/cosmos_web/Dockerfile -t cosmos-integrator-lab .
#   docker run --rm -p 4173:80 cosmos-integrator-lab      # http://localhost:4173
#
# `--target dist --output type=local,dest=site` exports only the static site
# (used by the GitHub Pages workflow).

FROM rust:1.97.1-slim-bookworm AS wasm
RUN rustup target add wasm32-unknown-unknown
# wasm-bindgen-cli must equal the crate's exact pin. Reading the pin from the
# manifest alone keeps this slow layer cached across source edits.
COPY Cosmos/Rust/cosmos_web/Cargo.toml /pin/Cargo.toml
RUN cargo install wasm-bindgen-cli --locked --version \
      "$(sed -n 's/^wasm-bindgen = "=\([0-9.]*\)"/\1/p' /pin/Cargo.toml)"
COPY Stunticons/wildrider /src/Stunticons/wildrider
COPY Cosmos/Rust /src/Cosmos/Rust
WORKDIR /src/Cosmos/Rust
RUN cargo build --profile web --target wasm32-unknown-unknown -p cosmos_web \
 && wasm-bindgen --target web --out-dir /generated \
      target/wasm32-unknown-unknown/web/cosmos_web.wasm

FROM node:24.18.1-bookworm-slim AS web
WORKDIR /web
COPY Cosmos/Rust/cosmos_web/web/package.json Cosmos/Rust/cosmos_web/web/package-lock.json ./
RUN npm ci
COPY Cosmos/Rust/cosmos_web/web/ ./
COPY --from=wasm /generated src/generated
RUN npm run build:web

FROM scratch AS dist
COPY --from=web /web/dist/ /

FROM nginx:1.30.1-alpine AS serve
COPY --from=web /web/dist/ /usr/share/nginx/html/
