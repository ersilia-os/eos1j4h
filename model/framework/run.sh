#!/bin/bash

# Usage:
# ./run.sh <input_smiles_csv> <output_npy> <output_csv> [output_json] 
# output json is optional

set -e

SMILES_PATH=$1
OUTPUT_NPY=$2
OUTPUT_CSV=$3
OUTPUT_JSON=$4

PYTHON_SCRIPT="./code/main.py"

if [ -z "$OUTPUT_JSON" ]; then
  python "$PYTHON_SCRIPT" \
    --smiles "$SMILES_PATH" \
    --output "$OUTPUT_NPY" \
    --csv "$OUTPUT_CSV"
else
  python "$PYTHON_SCRIPT" \
    --smiles "$SMILES_PATH" \
    --output "$OUTPUT_NPY" \
    --csv "$OUTPUT_CSV" \
    --json "$OUTPUT_JSON"
fi
