import os
from datasets import load_from_disk, DatasetDict

def main():
    # 1. 경로 설정 (사용자 환경에 맞게)
    # (A) 오답이 포함된 데이터 (Train용)
    negative_data_path = "/data/ephemeral/git/pro-nlp-mrc-nlp-01/data_negative"
    
    # (B) 원본 데이터 (Validation용)
    original_data_path = "/data/ephemeral/git/pro-nlp-mrc-nlp-01/data/train_dataset"
    
    # (C) 저장할 경로
    output_path = "/data/ephemeral/git/pro-nlp-mrc-nlp-01/data_hybrid"

    print("Loading datasets...")
    ds_neg = load_from_disk(negative_data_path)   # Train (columns: id, question, context, answers)
    ds_orig = load_from_disk(original_data_path)  # Validation (columns: + title, document_id...)

    # -------------------------------------------------------------------------
    # [핵심 수정] Validation 데이터의 컬럼을 Train 데이터와 똑같이 맞춰주기
    # -------------------------------------------------------------------------
    train_columns = ds_neg["train"].column_names
    val_data = ds_orig["validation"]
    val_columns = val_data.column_names
    
    # Train에는 없는데 Validation에만 있는 '불필요한 컬럼' 찾기
    extra_columns = [col for col in val_columns if col not in train_columns]
    
    if extra_columns:
        print(f"Detected extra columns in validation set: {extra_columns}")
        print("Removing extra columns to match train set schema...")
        # 불필요한 컬럼 삭제!
        val_data = val_data.remove_columns(extra_columns)
    else:
        print("Columns already match. No removal needed.")
    # -------------------------------------------------------------------------

    # 2. 데이터 섞기 (Cleaned Validation 사용)
    hybrid_datasets = DatasetDict({
        "train": ds_neg["train"], 
        "validation": val_data  # 컬럼이 정리된 데이터셋
    })

    # 3. 저장
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"Saving hybrid dataset to {output_path}...")
    hybrid_datasets.save_to_disk(output_path)
    
    print("-" * 30)
    print("Done!")
    print(f"Train columns: {hybrid_datasets['train'].column_names}")
    print(f"Valid columns: {hybrid_datasets['validation'].column_names}")
    print("-" * 30)

if __name__ == "__main__":
    main()