#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias runner; canonical runner is run_financial_q_and_a.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${FINSIGHTAI_CONDA_ENV_NAME:-finsightai-notebooks}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda command not found. Install Miniconda/Anaconda and run setup_conda_env.sh." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
python3 "${SCRIPT_DIR}/financial_q_and_a.py" "$@"
