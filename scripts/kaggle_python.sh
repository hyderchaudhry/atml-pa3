#!/usr/bin/env bash
# Sourced by kaggle.sh. Keep interpreter setup separate from experiment commands.

compatible_python() {
    command -v "$1" >/dev/null 2>&1 && "$1" -c \
        'import platform, sys; sys.exit(not (platform.python_implementation() == "CPython" and (3, 10) <= sys.version_info[:2] < (3, 13)))' \
        >/dev/null 2>&1
}

retire_incomplete_venv() {
    if [[ -e "$VENV_DIR" ]]; then
        local retired
        retired="$(mktemp -d "$SCRATCH_DIR/previous-venv.XXXXXX")"
        mv "$VENV_DIR" "$retired/venv"
        echo "[setup] Preserved the old/incomplete environment at $retired/venv"
    fi
}

create_kaggle_venv() {
    local candidate selected bootstrap_python bootstrap_dir
    mkdir -p "$SCRATCH_DIR" "$OUTPUT_DIR" "$TMPDIR"
    if compatible_python "$VENV_DIR/bin/python" && \
        "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
        echo "[setup] Reusing $VENV_DIR/bin/python"
        return
    fi
    selected=""
    if [[ -n "${PA3_PYTHON:-}" ]]; then
        if ! compatible_python "$PA3_PYTHON"; then
            echo 'PA3_PYTHON must name a CPython 3.10-3.12 executable. Unset it for automatic setup.' >&2
            return 1
        fi
        selected="$PA3_PYTHON"
    else
        for candidate in python3.11 python3.12 python3.10 python3; do
            if compatible_python "$candidate"; then
                selected="$candidate"
                break
            fi
        done
    fi
    retire_incomplete_venv
    if [[ -n "$selected" ]]; then
        echo "[setup] Creating an isolated environment with $selected"
        if "$selected" -m venv "$VENV_DIR"; then
            return
        fi
        echo '[setup] System venv support is unavailable; switching to managed Python.'
        retire_incomplete_venv
    fi
    # Kaggle's default Python can be newer than the pinned ML packages support.
    # Install only the bootstrap tool under scratch, leaving the kernel untouched.
    bootstrap_python="$(command -v python3)"
    bootstrap_dir="$SCRATCH_DIR/bootstrap"
    echo '[setup] Installing Python 3.11 in scratch space (Internet must be enabled).'
    "$bootstrap_python" -m pip install --no-cache-dir --target "$bootstrap_dir" 'uv==0.9.7'
    export UV_PYTHON_INSTALL_DIR="$SCRATCH_DIR/python"
    export UV_PYTHON_BIN_DIR="$SCRATCH_DIR/python-bin"
    export UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"
    PYTHONPATH="$bootstrap_dir" "$bootstrap_python" -m uv --no-config python install 3.11
    PYTHONPATH="$bootstrap_dir" "$bootstrap_python" -m uv --no-config venv \
        --python 3.11 --seed "$VENV_DIR"
    if ! compatible_python "$VENV_DIR/bin/python"; then
        echo 'Managed Python setup did not produce a supported environment.' >&2
        return 1
    fi
}
