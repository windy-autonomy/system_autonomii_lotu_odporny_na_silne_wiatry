#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Virtual environment not found at: $VENV_DIR"
  echo "Create it first with: ./setup_venv.sh $VENV_DIR"
  exit 1
fi

"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo "Requirements installed in: $VENV_DIR"
