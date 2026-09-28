#!/usr/bin/env bash
# Create / refresh the project-local virtualenv and Jupyter kernel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[notebooks]"
# scGPT is vendored; install editable without pulling its full (often conflict-prone) dep tree.
# Core runtime deps are already covered by ".[notebooks]" + requirements.txt.
pip install -e "third_party/scGPT" --no-deps
python -m ipykernel install --user --name=attention --display-name="Python (attention)"

echo
echo "Environment ready."
echo "  activate:  source $ROOT/.venv/bin/activate"
echo "  kernel:    Python (attention)"
echo "  notebooks: $ROOT/notebooks/"
