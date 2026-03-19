#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
ENV_NAME="${FINSIGHTAI_CONDA_ENV_NAME:-finsightai-notebooks}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda command not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

if [[ -s "${REQ_FILE}" ]]; then
  python -m pip install -r "${REQ_FILE}"
fi

echo "Conda environment ready: ${ENV_NAME}"
