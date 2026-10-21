# Source this before running the matching scripts by hand:  . tools/env.sh
# Puts the calc/bc stand-ins and gnuplot on PATH and points $PYTHON at the project venv.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$_ROOT/tools/bin:/c/Program Files/gnuplot/bin:$PATH"
export PYTHON="$_ROOT/.venv/Scripts/python.exe"
