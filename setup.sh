#!/usr/bin/env bash
# AFM project: one-shot setup for a fresh Mac. Run it from anywhere:
#   bash ~/afm_project/setup.sh
# It installs "uv" (a small tool by Astral that fetches Python for you) if needed,
# creates .venv with Python 3.12, installs the libraries and runs the tests.
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "==> 1/4 uv (Python manager)"
if ! command -v uv >/dev/null 2>&1; then
  echo "    uv not found, installing from astral.sh ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version

echo "==> 2/4 Python 3.12 virtual environment (.venv)"
uv venv --python 3.12 --allow-existing .venv

echo "==> 3/4 libraries"
uv pip install --python .venv/bin/python -r requirements.txt
uv pip freeze --python .venv/bin/python > requirements.lock.txt

echo "==> 4/4 tests"
.venv/bin/python -m pytest -q

echo
echo "==> JPG files in data/raw:"
find data/raw -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) ! -name '._*' | wc -l
echo "==> DONE. To use the environment later:  cd ~/afm_project && source .venv/bin/activate"
