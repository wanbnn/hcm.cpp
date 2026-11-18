#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${COGNICORE_MODEL:-${project_dir}/models/conversa-pt-fluent-v3}"
cli_path="${COGNICORE_CLI:-${project_dir}/build/cognicore}"
# A user-supplied --model must win over the default and must not be duplicated.
args=("$@")
has_model=false
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--model" && $((i + 1)) -lt ${#args[@]} ]]; then
        model_dir="${args[i + 1]}"
        has_model=true
    fi
done
if [[ ! -x "$cli_path" ]]; then
    echo "Binário ausente. Consulte docs/MODEL_CONVERSA_PT.md." >&2
    exit 1
fi
if ! $has_model; then
    args=(--model "$model_dir" "${args[@]}")
fi
if [[ ! -f "$model_dir/codebook.bin" ]]; then
    echo "Modelo ausente: $model_dir. Consulte docs/MODEL_CONVERSA_PT.md." >&2
    exit 1
fi
cd -- "$project_dir"
exec "$cli_path" chat "${args[@]}"
