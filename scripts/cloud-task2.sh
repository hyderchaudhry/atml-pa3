#!/usr/bin/env bash
# Run from a clone on Vast/RunPod; PA3_ROOT must point to persistent storage.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PA3_ROOT:?Set PA3_ROOT to a persistent directory, e.g. /workspace/pa3-task2}"
export PA3_ROOT
mkdir -p "$PA3_ROOT"
PA3_ROOT="$(cd "$PA3_ROOT" && pwd)"
export PA3_ROOT
export HF_HOME="$PA3_ROOT/cache/huggingface"
export MPLCONFIGDIR="$PA3_ROOT/cache/matplotlib"
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
VENV_DIR="$PA3_ROOT/venv-task2"
command="${1:-run}"
if [[ "$command" == setup ]]; then
    python="${PA3_PYTHON:-python3.11}"
    "$python" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13), "Use CPython 3.10–3.12"'
    "$python" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade 'pip==25.1.1'
    # CUDA 12.4 wheels include the user-space runtime; the host needs a compatible driver.
    "$VENV_DIR/bin/python" -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
    "$VENV_DIR/bin/python" -m pip install -r "$REPO_DIR/requirements-task2.txt"
    "$VENV_DIR/bin/python" -m pip freeze > "$PA3_ROOT/environment-freeze.txt"
    echo "[setup] Ready. Run: bash scripts/cloud-task2.sh auth"
    exit 0
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo 'Run bash scripts/cloud-task2.sh setup first.' >&2
    exit 1
fi
shift || true
exec "$VENV_DIR/bin/python" -m atml.task2 --root "$PA3_ROOT" "$command" "$@"
