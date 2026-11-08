#!/usr/bin/env bash
# Persistent storage, one frozen FP16 3B model, and strictly batch-one live calls.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PA3_ROOT:?Set PA3_ROOT to persistent storage, e.g. /workspace/pa3-task3}"
mkdir -p "$PA3_ROOT"
PA3_ROOT="$(cd "$PA3_ROOT" && pwd)"
export PA3_ROOT
export HF_HOME="${PA3_HF_HOME:-$PA3_ROOT/cache/huggingface}"
export XDG_CACHE_HOME="$PA3_ROOT/cache"
export MPLCONFIGDIR="$PA3_ROOT/cache/matplotlib"
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
VENV_DIR="$PA3_ROOT/venv-task3"
command="${1:-run}"
if [[ "$command" == setup ]]; then
    python="${PA3_PYTHON:-python3.11}"
    "$python" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13), "Use CPython 3.10–3.12"'
    "$python" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade 'pip==25.1.1'
    "$VENV_DIR/bin/python" -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
    "$VENV_DIR/bin/python" -m pip install -r "$REPO_DIR/requirements-task3.txt"
    "$VENV_DIR/bin/python" -m pip freeze > "$PA3_ROOT/environment-freeze.txt"
    echo '[setup] Ready. Run bash scripts/cloud-task3.sh auth, then smoke.'
    exit 0
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo 'Run bash scripts/cloud-task3.sh setup first.' >&2
    exit 1
fi
shift || true
exec "$VENV_DIR/bin/python" -m atml.task3 --root "$PA3_ROOT" "$command" "$@"
