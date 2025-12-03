#!/bin/bash
# Training script wrapper

cd "$(dirname "$0")/.." || exit
python -m src.training.train "$@"

