#!/usr/bin/env bash
# Fetch Hairer's DOPRI5 and DOP853 Fortran codes into a build directory that
# is not under version control, and verify their SHA-256 digests.
#
# The codes are "free subject to the licence agreement" on
# http://www.unige.ch/~hairer/software.html; they are fetched, never vendored.
#
# usage: fetch_hairer_codes.sh [--from LOCAL_DIR] [BUILD_DIR]
#   --from LOCAL_DIR  copy dopri5.f and dop853.f from LOCAL_DIR instead of
#                     downloading (the digests are still checked)
#   BUILD_DIR         default: Cosmos/Rust/target/hairer_reference
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rust_root="$(cd "${script_directory}/../.." && pwd)"
source_directory=""
if [[ "${1:-}" == "--from" ]]; then
  source_directory="$2"
  shift 2
fi
build_directory="${1:-${rust_root}/target/hairer_reference}"
base_url="http://www.unige.ch/~hairer/prog/nonstiff"

declare -A expected_sha256=(
  [dopri5.f]=e4fbbe080508134f743d9934e76e45f49f03a320b7b024273757a7c278804c95
  [dop853.f]=8bfc0a954a38af1a458f5ab8025fb09c923373a936f52ff61e32f1a15c8f298a
)

mkdir -p "${build_directory}"
for file in dopri5.f dop853.f; do
  destination="${build_directory}/${file}"
  if [[ -n "${source_directory}" ]]; then
    cp "${source_directory}/${file}" "${destination}"
  elif [[ ! -f "${destination}" ]]; then
    curl -sSfL --max-time 60 -o "${destination}" "${base_url}/${file}"
  fi
  actual="$(sha256sum "${destination}" | cut -d' ' -f1)"
  if [[ "${actual}" != "${expected_sha256[$file]}" ]]; then
    echo "ERROR: ${file} sha256 ${actual} != expected ${expected_sha256[$file]}" >&2
    echo "Hairer may have published a new version; inspect it before updating the digest." >&2
    rm -f "${destination}"
    exit 1
  fi
  echo "ok ${file} ${actual}"
done
