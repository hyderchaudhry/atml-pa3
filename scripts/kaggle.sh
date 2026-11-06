#!/usr/bin/env bash
# Run from a cloned repository; no notebook file is needed.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_DIR="${PA3_SCRATCH_DIR:-/kaggle/tmp/atml-pa3}"
OUTPUT_DIR="${PA3_OUTPUT_DIR:-/kaggle/working/pa3}"
VENV_DIR="$SCRATCH_DIR/venv"
export HF_HOME="$SCRATCH_DIR/huggingface"
export MPLCONFIGDIR="$SCRATCH_DIR/matplotlib"
export XDG_CACHE_HOME="$SCRATCH_DIR/cache"
export PIP_NO_CACHE_DIR=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$SCRATCH_DIR/tmp"

if [[ $# -eq 0 || "$1" == "help" || "$1" == "--help" ]]; then
    cat <<'HELP'
Usage: bash scripts/kaggle.sh COMMAND [arguments]
  setup                       Install an isolated CUDA environment in scratch space
  doctor                      Check CUDA, dependencies, paths, and storage
  run                         Automatically restore and run remaining Task 1 steps
  next                        Run and back up just the next required step
  status                      Show saved progress and the next required step
  export                      Package evidence into a downloadable ZIP
  restore                     Restore shared artifacts/evidence from PA3_BACKUP_REPO
  train-stage METHOD          Resume through the next required recovery checkpoint
  backup                      Back up artifacts/evidence and evict verified checkpoints
  COMMAND ...                 Run an atml-task1 command, then back up and free checkpoints

Set HF_TOKEN and PA3_BACKUP_REPO; the question YAML downloads automatically.
Set PA3_OUTPUT_DIR or PA3_SCRATCH_DIR to override Kaggle defaults.
HELP
    exit 0
fi

COMMAND="$1"
shift
if [[ "$COMMAND" == "setup" ]]; then
    source "$REPO_DIR/scripts/kaggle_python.sh"
    create_kaggle_venv
    "$VENV_DIR/bin/python" --version
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
    "$VENV_DIR/bin/python" -m pip install 'torch==2.6.0' \
        --index-url "${PA3_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
    "$VENV_DIR/bin/python" -m pip install -e "$REPO_DIR"
    "$VENV_DIR/bin/python" -m pip check
    echo 'Setup complete. Set your values from docs/kaggle.md, then: bash scripts/kaggle.sh run'
    exit 0
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo 'Run bash scripts/kaggle.sh setup in this session first.' >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR" "$TMPDIR"
cd "$REPO_DIR"
CORE=("$VENV_DIR/bin/python" -m atml.task1 --artifacts "$OUTPUT_DIR/artifacts/task1" --evidence "$OUTPUT_DIR/evidence/task1")

case "$COMMAND" in
    doctor|protocol)
        exec "${CORE[@]}" "$COMMAND" "$@"
        ;;
    run|next|status|export)
        if [[ "$COMMAND" != "export" ]]; then
            : "${PA3_BACKUP_REPO:?Set PA3_BACKUP_REPO=your-user/pa3-task1.}"
        fi
        exec "$VENV_DIR/bin/python" -m atml.task1.workflow "$COMMAND" \
            --artifacts "$OUTPUT_DIR/artifacts/task1" --evidence "$OUTPUT_DIR/evidence/task1" "$@"
        ;;
esac

: "${PA3_BACKUP_REPO:?Set PA3_BACKUP_REPO=your-user/pa3-task1 (a private Hugging Face dataset).}"
case "$COMMAND" in
    restore)
        exec "${CORE[@]}" restore --repo "$PA3_BACKUP_REPO" "$@"
        ;;
    backup)
        exec "${CORE[@]}" backup --repo "$PA3_BACKUP_REPO" --offload "$@"
        ;;
    train-stage)
        if [[ $# -ne 1 ]]; then
            echo 'Usage: bash scripts/kaggle.sh train-stage full_ft|l2sp|lora4|lora16|oplora16' >&2
            exit 1
        fi
        "${CORE[@]}" train --method "$1" --resume-latest --stop-after next \
            --checkpoint-repo "$PA3_BACKUP_REPO"
        ;;
    *)
        "${CORE[@]}" "$COMMAND" "$@"
        ;;
esac
# An upload failure returns a nonzero status and leaves local checkpoints intact.
"${CORE[@]}" backup --repo "$PA3_BACKUP_REPO" --offload
