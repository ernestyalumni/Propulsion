#!/usr/bin/env bash
# Build and open the Cosmos Integrator Lab in the default browser.
#
#   ./run-lab.sh                 local toolchain if complete, otherwise Docker
#   ./run-lab.sh --docker        always Docker (needs only Docker)
#   ./run-lab.sh --local         always the local toolchain
#   ./run-lab.sh --port 8080     serve on another port (default 4173)
#   ./run-lab.sh --no-open       do not open a browser
#   ./run-lab.sh --install-launcher
#                                Linux: add "Cosmos Integrator Lab" to the
#                                desktop's application menu
#
# The local toolchain is cargo with the wasm32-unknown-unknown target,
# wasm-bindgen-cli at the version pinned in Cargo.toml, and Node >= 20.19.
# Ctrl-C stops the server. Works with macOS's bash 3.2.
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
web_directory="${lab_directory}/web"
repository_root="$(cd "${lab_directory}/../../.." && pwd)"
image="cosmos-integrator-lab"

mode="auto"
port=4173
open_browser=1
hold=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker) mode="docker" ;;
    --local) mode="local" ;;
    --port) port="$2"; shift ;;
    --no-open) open_browser=0 ;;
    --hold) hold=1 ;;  # launchers: keep the terminal open after an exit
    --install-launcher) mode="install-launcher" ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

server_pid=""
container=""
cleanup()
{
  local status=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${server_pid}" ]]; then kill "${server_pid}" 2>/dev/null || true; fi
  if [[ -n "${container}" ]]; then docker stop "${container}" >/dev/null 2>&1 || true; fi
  if [[ "${hold}" == 1 ]]; then
    echo
    read -r -p "Lab stopped. Press Enter to close this window. " _ || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

pinned_bindgen()
{
  sed -n 's/^wasm-bindgen = "=\([0-9.]*\)"/\1/p' "${lab_directory}/Cargo.toml"
}

# Prints what the local toolchain lacks; prints nothing when it is complete.
local_toolchain_gaps()
{
  command -v cargo >/dev/null || echo "cargo (https://rustup.rs)"
  if command -v rustup >/dev/null \
    && ! rustup target list --installed | grep -x wasm32-unknown-unknown >/dev/null; then
    echo "the wasm target: rustup target add wasm32-unknown-unknown"
  fi
  local pin installed=""
  pin="$(pinned_bindgen)"
  if command -v wasm-bindgen >/dev/null; then
    installed="$(wasm-bindgen --version | awk '{print $2}')"
  fi
  if [[ "${installed}" != "${pin}" ]]; then
    echo "wasm-bindgen-cli ${pin}: cargo install wasm-bindgen-cli --version ${pin} --locked"
  fi
  if ! command -v node >/dev/null \
    || ! node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a===22&&b>=12)||(a===20&&b>=19)?0:1)'; then
    echo "Node.js >= 20.19 (https://nodejs.org)"
  fi
}

docker_ready()
{
  command -v docker >/dev/null && docker info >/dev/null 2>&1
}

open_url()
{
  local url="$1"
  [[ "${open_browser}" == 1 ]] || return 0
  if [[ "$(uname -s)" == "Darwin" ]]; then
    open "${url}"
  elif command -v xdg-open >/dev/null; then
    xdg-open "${url}" >/dev/null 2>&1 &
  else
    echo "Open ${url} in a browser."
  fi
}

wait_for()
{
  local url="$1" attempt
  for attempt in $(seq 1 100); do
    if curl -fsS -o /dev/null "${url}" 2>/dev/null; then return 0; fi
    sleep 0.2
  done
  echo "the server did not answer at ${url}" >&2
  return 1
}

run_local()
{
  echo "==> Building with the local toolchain"
  cd "${web_directory}"
  if [[ ! -d node_modules || package-lock.json -nt node_modules ]]; then
    npm ci
  fi
  npm run build
  npx vite preview --host 127.0.0.1 --port "${port}" --strictPort &
  server_pid=$!
}

run_docker()
{
  echo "==> Building the ${image} image (the first build takes a few minutes)"
  docker build -f "${lab_directory}/Dockerfile" -t "${image}" "${repository_root}"
  container="$(docker run -d --rm -p "127.0.0.1:${port}:80" "${image}")"
  docker logs -f "${container}" &
  server_pid=$!
}

# A freedesktop.org menu entry with this checkout's absolute path. Inside the
# double-quoted Exec argument the spec reserves " ` $ \, each written \\x.
install_launcher()
{
  local applications="${XDG_DATA_HOME:-${HOME}/.local/share}/applications"
  local entry="${applications}/cosmos-integrator-lab.desktop"
  local escaped
  escaped="$(printf '%s' "${lab_directory}/run-lab.sh" | sed 's/[\\"`$]/\\\\&/g')"
  mkdir -p "${applications}"
  cat > "${entry}" <<EOF
[Desktop Entry]
Type=Application
Name=Cosmos Integrator Lab
Comment=Rust integrators compiled to WebAssembly, rendered live
Exec="${escaped}" --hold
Terminal=true
Icon=applications-science
Categories=Education;
EOF
  echo "Installed ${entry}"
  echo "Open Activities (or the application menu) and search for Cosmos Integrator Lab."
}

case "${mode}" in
  install-launcher)
    if [[ "$(uname -s)" == "Darwin" ]]; then
      echo "On macOS, double-click \"Cosmos Integrator Lab.command\" in Finder instead." >&2
      exit 1
    fi
    install_launcher
    exit 0 ;;
  local)
    gaps="$(local_toolchain_gaps)"
    if [[ -n "${gaps}" ]]; then
      echo "The local toolchain is missing:" >&2
      echo "${gaps}" | sed 's/^/  - /' >&2
      exit 1
    fi
    run_local ;;
  docker)
    if ! docker_ready; then echo "Docker is not installed or not running." >&2; exit 1; fi
    run_docker ;;
  auto)
    gaps="$(local_toolchain_gaps)"
    if [[ -z "${gaps}" ]]; then
      run_local
    elif docker_ready; then
      run_docker
    else
      echo "Neither route is available. Either start Docker, or install:" >&2
      echo "${gaps}" | sed 's/^/  - /' >&2
      exit 1
    fi ;;
esac

url="http://127.0.0.1:${port}/"
wait_for "${url}"
echo
echo "==> Cosmos Integrator Lab: ${url}   (WebGL 2 fallback: ${url}?webgl)"
echo "    Ctrl-C stops the server."
open_url "${url}"
wait "${server_pid}"
