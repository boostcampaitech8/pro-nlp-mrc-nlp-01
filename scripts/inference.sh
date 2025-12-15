#!/bin/bash
# Inference script wrapper

cd "$(dirname "$0")/.." || exit
python -m src.inference.inference "$@"

 