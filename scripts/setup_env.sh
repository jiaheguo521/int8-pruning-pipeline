#!/usr/bin/env bash
# setup_env.sh — build the virtualenvs. One command, both export paths.
#
# Usage:
#   ./scripts/setup_env.sh                       # ./pruning-env, the whole pipeline
#   ONNX2TF_ENABLED=1 ./scripts/setup_env.sh     # also ./onnx2tf-env, the retired path
# Every toggle, and why two venvs rather than one: docs/SETUP.md section 1.

set -euo pipefail

if [[ $# -gt 0 ]]; then
    echo "[ERROR] $(basename "$0") takes no arguments; it is env-var driven." >&2
    echo "        e.g. ONNX2TF_ENABLED=1 ./scripts/setup_env.sh   (see the header)" >&2
    exit 1
fi

# PROJECT_DIR comes from scripts/config.sh. No require_venv here: this is what creates the venv.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$PROJECT_DIR"

PRUNING_ENV_ENABLED="${PRUNING_ENV_ENABLED:-1}"
ONNX2TF_ENABLED="${ONNX2TF_ENABLED:-0}"

PY="${PY:-python3}"
EXTRAS="${EXTRAS:-all}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/pruning-env}"
ONNX2TF_VENV_DIR="${ONNX2TF_VENV_DIR:-$PROJECT_DIR/onnx2tf-env}"
# tensorflow 2.19's ceiling. It violates onnx2tf's own `numpy==1.26.4` pin, which pip reports as a
# warning and which works; export_onnx2tf carries the one numpy-2 workaround that makes it work.
ONNX2TF_NUMPY="${ONNX2TF_NUMPY:-2.1.3}"

step() { printf '\n\033[1m>>> %s\033[0m\n' "$*"; }

make_venv() {  # make_venv <dir>
    if [[ -d "$1" ]]; then
        # pip is NOT upgraded here: re-running must not change the resolver under a working environment.
        echo "    [exists] $1 — reusing it; pip re-resolves in place"
        return
    fi
    if ! command -v "$PY" >/dev/null 2>&1; then
        echo "[ERROR] PY=$PY not found on PATH. Set PY to a Python >=3.10, e.g." >&2
        echo "        PY=python3.12 ./scripts/setup_env.sh" >&2
        exit 1
    fi
    echo "    creating $1 with $PY ($("$PY" -V 2>&1))"
    "$PY" -m venv "$1"
    "$1/bin/pip" install --quiet --upgrade pip
}

echo "pruning-env: $([[ "$PRUNING_ENV_ENABLED" == 1 ]] && echo "yes, extras [$EXTRAS]" || echo "skipped (PRUNING_ENV_ENABLED=0)")"
echo "onnx2tf-env: $([[ "$ONNX2TF_ENABLED" == 1 ]] && echo "yes" || echo "skipped (set ONNX2TF_ENABLED=1 to build it)")"

if [[ "$PRUNING_ENV_ENABLED" == 1 ]]; then
    step "pruning-env — pruning, recovery, and the litert int8 export path"
    make_venv "$VENV_DIR"

    # torch goes in FIRST from the CUDA-matched index; listed in pyproject the resolver takes a CPU wheel.
    echo "    torch==$TORCH_VERSION + torchvision from $TORCH_INDEX"
    "$VENV_DIR/bin/pip" install "torch==$TORCH_VERSION" torchvision --index-url "$TORCH_INDEX"

    echo "    pip install -e '.[$EXTRAS]'"
    "$VENV_DIR/bin/pip" install -e ".[$EXTRAS]"

    # Versions come from the installed distributions, not module attributes: torch_pruning.__version__
    # reports 1.6.0 inside the 1.6.1 wheel, and ai_edge_quantizer has no such attribute at all.
    "$VENV_DIR/bin/python" - <<'PY'
import sys
from importlib.metadata import version
import torch, torch_pruning, int8_pruning  # noqa: F401
print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()}"
      f"  torch-pruning {version('torch-pruning')}")
try:
    import litert_torch, ai_edge_quantizer, ai_edge_litert  # noqa: F401
except ImportError as e:
    print(f"    [note] litert export path absent ({e.name}): these EXTRAS do not "
          f"include 'convert'. Pruning works; scripts/convert.sh will not.")
else:
    print(f"    litert-torch {version('litert-torch')}  "
          f"ai-edge-quantizer {version('ai-edge-quantizer')}  "
          f"ai-edge-litert {version('ai-edge-litert')}")
    if tuple(int(p) for p in torch.__version__.split('+')[0].split('.')[:2]) < (2, 10):
        print(f"    [ERROR] torch {torch.__version__} is below the 2.10 floor the "
              f"export path needs. Re-run with TORCH_VERSION=2.10.0.", file=sys.stderr)
        sys.exit(1)
PY
fi

if [[ "$ONNX2TF_ENABLED" == 1 ]]; then
    step "onnx2tf-env — the pre-2026-08-22 export path (tensorflow + onnx)"
    make_venv "$ONNX2TF_VENV_DIR"

    # [families] and not [all]: this venv converts, it does not prune, and converting a family's .pt
    # means unpickling it, which imports that family's package by name. torch/torchvision come from
    # PyPI here rather than TORCH_INDEX, this path never touching the GPU.
    echo "    pip install -e '.[families,onnx2tf]' torch torchvision"
    "$ONNX2TF_VENV_DIR/bin/pip" install -e '.[families,onnx2tf]' torch torchvision

    echo "    pip install numpy==$ONNX2TF_NUMPY   # expect one dependency-conflict warning"
    "$ONNX2TF_VENV_DIR/bin/pip" install "numpy==$ONNX2TF_NUMPY"

    "$ONNX2TF_VENV_DIR/bin/python" - <<'PY'
import numpy, onnx, tensorflow, torch  # noqa: F401
print(f"    numpy {numpy.__version__}  tensorflow {tensorflow.__version__}  "
      f"onnx {onnx.__version__}  torch {torch.__version__}")
PY
fi

step "done — activate what you are about to use"
if [[ "$PRUNING_ENV_ENABLED" == 1 ]]; then
    echo "    source $VENV_DIR/bin/activate"
fi
if [[ "$ONNX2TF_ENABLED" == 1 ]]; then
    echo "    source $ONNX2TF_VENV_DIR/bin/activate   # then EXPORT_PATH=onnx2tf ./scripts/convert.sh"
fi
