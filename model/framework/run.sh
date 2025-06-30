#!/bin/bash

# Usage:
# ./run.sh <input_smiles_csv> <output_csv> [device]
# device is optional, defaults to cpu

set -e

SMILES_PATH=$1
OUTPUT_CSV=$2
DEVICE=${3:-cpu}  # default to cpu if not provided

PYTHON_SCRIPT="./code/main.py"

python "$PYTHON_SCRIPT" \
  --smiles "$SMILES_PATH" \
  --csv "$OUTPUT_CSV" \
  --device "$DEVICE"
