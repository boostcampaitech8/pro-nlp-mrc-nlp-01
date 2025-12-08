#!/bin/bash
# Training script wrapper


cd "$(dirname "$0")/.." || exit

python -m src.training.train \
    --model_name_or_path HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned \
    --save_strategy no \
    "$@"

