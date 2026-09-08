#!/usr/bin/env bash
# Repository-context inference stays on the explicitly configured local endpoint.
set -euo pipefail
pdd_task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pdd_task_bin="${PDD_EXECUTABLE:-/home/propdev/.openclaw/workspace/workspace2/repos/PromptDrivenDevelopment/pdd/.venv/bin/pdd}"
if [[ ! -x "$pdd_task_bin" ]]; then
    echo 'Set PDD_EXECUTABLE to an executable from Ernest’s local-LLM-capable PDD fork.' >&2
    exit 2
fi
# Do not inherit an unrelated endpoint, disabled routing, or provider credentials.
unset PDD_LOCAL_LLM_CONFIG_JSON PDD_LOCAL_LLM_BASE_URL PDD_LOCAL_LLM_MODEL
unset PDD_LOCAL_LLM_MAX_TOKENS PDD_LOCAL_LLM_TIMEOUT
unset OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY
unset DEEPSEEK_API_KEY OPENROUTER_API_KEY PDD_LOCAL_LLM_API_KEY
export PDD_LOCAL_LLM_CONFIG="$pdd_task_dir/.pdd/local_llm.json"
export PDD_LOCAL_LLM_ENABLED=1 PDD_LOCAL_ONLY=1 PDD_FORCE_LOCAL=1
export LITELLM_LOCAL_MODEL_COST_MAP=true PYTHONDONTWRITEBYTECODE=1
cd -- "$pdd_task_dir"
exec "$pdd_task_bin" --no-core-dump "$@"
