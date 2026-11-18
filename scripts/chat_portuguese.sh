#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${COGNICORE_MODEL:-${project_dir}/models/conversa-pt-fluent-v3}"
cli_path="${COGNICORE_CLI:-${project_dir}/build/cognicore}"
if [[ ! -x "$cli_path" || ! -f "$model_dir/codebook.bin" ]]; then
    echo "Binário ou modelo ausente. Consulte docs/MODEL_CONVERSA_PT.md." >&2
    exit 1
fi
cd -- "$project_dir"
exec "$cli_path" chat --model "$model_dir" "$@"
