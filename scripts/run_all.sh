#!/bin/bash

models=(
    "monologg/kobigbird-bert-base"
    "lighthouse/mdeberta-v3-base-kor-further"
    "DHBaek/xlm-roberta-large-korquad-mask"
)

dataset_path="data/train_dataset"

for m in "${models[@]}"; do
    safe_name="${m//\//_}"
    out_dir="temp_runs/$safe_name"   # 🔥 모델별 독립적인 임시 폴더

    echo "======================================"
    echo " 🚀 Start Training: $m"
    echo " 🕒 Time: $(date)"
    echo "======================================"


    python -m src.training.train \
        --model_name_or_path "$m" \
        --dataset_name "$dataset_path" \
        --output_dir models/train_dataset \
        --overwrite_output_dir \
        --do_train \
        --save_strategy no \
        --do_eval \
        --run_name "$safe_name"

    echo "======================================"
    echo " ✅ Finished: $m"
    echo " 🏁 End Time: $(date)"
    echo "======================================"
done
