#!/usr/bin/env bash
# Compile cosmos_web to WebAssembly and generate its JS/TS bindings into
# src/generated/. Requires the wasm32-unknown-unknown target and
# wasm-bindgen-cli at the version pinned in cosmos_web/Cargo.toml.
set -euo pipefail
web_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rust_root="$(cd "${web_directory}/../.." && pwd)"
pinned="$(sed -n 's/^wasm-bindgen = "=\([0-9.]*\)"/\1/p' "${rust_root}/cosmos_web/Cargo.toml")"
installed="$(wasm-bindgen --version | awk '{print $2}')"
if [[ "${pinned}" != "${installed}" ]]; then
  echo "wasm-bindgen-cli ${installed} does not match the crate's ${pinned}:" >&2
  echo "  cargo install wasm-bindgen-cli --version ${pinned} --locked" >&2
  exit 1
fi
cargo build --manifest-path "${rust_root}/Cargo.toml" --profile web \
  --target wasm32-unknown-unknown -p cosmos_web
wasm-bindgen --target web --out-dir "${web_directory}/src/generated" \
  "${rust_root}/target/wasm32-unknown-unknown/web/cosmos_web.wasm"
