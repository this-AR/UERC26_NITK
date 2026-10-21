#!/bin/bash
# Run the full scripts_d_t_2 evaluation for the given datasets (default: all four).
#   bash run_all.sh [AWE ITW UBEAR NITJ]
set -e
. /c/Users/hp/Downloads/ITW/dt2_tools/env.sh
TOOLS=/c/Users/hp/Downloads/ITW/dt2_tools
SCRIPTS="/c/Users/hp/Downloads/ITW/scripts_d_t_2 (Matching Results)/scripts_d_t_2 (Matching Results)"
DATASETS=${@:-AWE ITW UBEAR NITJ}

for key in $DATASETS; do
    echo "==================== $key ($(date +%T))"
    cd "$TOOLS"
    $PYTHON make_scores.py "$key" | grep -v "embeddings computed\|rows \["
    out=$($PYTHON -c "from datasets import output_dir; print(output_dir('$key').as_posix())")
    out=$(cygpath -u "$out")
    rm -rf "$out/run"; mkdir -p "$out/run"
    cp "$SCRIPTS"/* "$out/run/"; chmod +x "$out/run/"*.sh
    mv "$out/$key.txt" "$out/run/"
    cd "$out/run"
    echo "max_min.sh + main.sh started ($(date +%T))"
    ./max_min.sh "$key.txt" > max_min.log 2>&1
    echo "lines in: $(wc -l < $key.txt)  lines after max_min filter: $(wc -l < $key.txtf)"
    cd "$TOOLS"
    echo "extras started ($(date +%T))"
    $PYTHON extras.py "$key"
done
if [ "${DT2_FINALIZE:-1}" = 1 ]; then
    echo "==================== finalize ($(date +%T))"
    cd "$TOOLS"
    $PYTHON finalize.py AWE ITW UBEAR NITJ
fi
echo "ALL DONE ($(date +%T))"
