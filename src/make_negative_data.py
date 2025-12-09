import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_from_disk, Dataset, DatasetDict, Features, Value, Sequence
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from tqdm import tqdm

# [수정 1] src/config/arguments.py 위치 반영
try:
    from config.arguments import ModelArguments, DataTrainingArguments
except ImportError:
    # 혹시 모듈 경로가 안 잡힐 경우를 대비해 예외 처리
    sys.path.append(os.path.join(os.path.dirname(__file__), "config"))
    from arguments import ModelArguments, DataTrainingArguments

# retrieval.py는 src/retrieval.py (또는 src/retrieval 폴더)에 있으므로 바로 import
from retrieval import SparseRetrieval

# [수정 2] arguments.py에 없는 경로 인자를 처리하기 위한 로컬 클래스 정의
@dataclass
class ScriptArguments:
    data_path: str = field(
        default="../data",
        metadata={"help": "Path to data directory (containing sparse_embedding.bin)"}
    )
    context_path: str = field(
        default="wikipedia_documents.json",
        metadata={"help": "Path to wikipedia documents json file"}
    )

def augment_split(
    split_name: str,
    dataset: Dataset,
    retriever: SparseRetrieval,
    k_retrieval: int = 20,
    target_negatives: int = 2
) -> Dataset:
    print(f"[{split_name}] Processing {len(dataset)} samples...")
    
    # 1. 대량 검색 (Bulk Search)
    queries = dataset["question"]
    # 검색할 쿼리가 너무 많으면 메모리 이슈가 있을 수 있으니 배치 처리가 좋지만,
    # 여기서는 일단 전체 수행 (Retrieval 구현에 따라 다름)
    print(f"[{split_name}] Retrieving Top-{k_retrieval} passages...")
    
    # doc_scores, doc_indices 반환
    _, doc_indices = retriever.get_relevant_doc_bulk(queries, k=k_retrieval)

    # 2. 데이터 구축
    new_data = {
        "id": [],
        "question": [],
        "context": [],
        "answers": []
    }
    
    # 3. 데이터 증강 루프
    for idx, example in enumerate(tqdm(dataset, desc=f"Augmenting {split_name}")):
        # (A) Positive Sample (원본 정답)
        original_context = example["context"]
        
        new_data["id"].append(example["id"])
        new_data["question"].append(example["question"])
        new_data["context"].append(original_context)
        new_data["answers"].append(example["answers"])

        # (B) Negative Samples (Hard Negatives)
        retrieved_indices = doc_indices[idx]
        negative_count = 0
        
        for doc_idx in retrieved_indices:
            if negative_count >= target_negatives:
                break
            
            candidate_context = retriever.contexts[doc_idx]
            
            # 원본 정답 문서와 텍스트가 같으면 Skip
            if candidate_context == original_context:
                continue

            # Negative Sample 추가
            new_data["id"].append(f"{example['id']}_neg_{negative_count}")
            new_data["question"].append(example["question"])
            new_data["context"].append(candidate_context)
            
            # 정답 없음 처리: 빈 리스트
            new_data["answers"].append({"text": [], "answer_start": []})
            
            negative_count += 1

    # 4. Dataset 객체 생성
    features = Features({
        "id": Value("string"),
        "question": Value("string"),
        "context": Value("string"),
        "answers": Sequence(feature={'text': Value(dtype='string'), 'answer_start': Value(dtype='int32')})
    })

    augmented_dataset = Dataset.from_dict(new_data, features=features)
    print(f"[{split_name}] Done! Size: {len(dataset)} -> {len(augmented_dataset)}")
    
    return augmented_dataset

def main():
    # 1. 인자 파싱 (ScriptArguments 추가)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, ScriptArguments))
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()

    # 2. 데이터 로드
    print(f"Loading original dataset from: {data_args.dataset_name}")
    datasets = load_from_disk(data_args.dataset_name)

    # 3. 토크나이저 및 Retrieval 초기화
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True,
    )

    print(f"Initializing SparseRetrieval with data_path: {script_args.data_path}")
    
    # script_args를 통해 경로 주입
    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=script_args.data_path,
        context_path=script_args.context_path,
    )
    retriever.get_sparse_embedding()

    # 4. 증강 수행
    augmented_splits = {}

    if "train" in datasets:
        augmented_splits["train"] = augment_split("train", datasets["train"], retriever)
    
    if "validation" in datasets:
        augmented_splits["validation"] = augment_split("validation", datasets["validation"], retriever)

    # 5. 저장
    final_dataset = DatasetDict(augmented_splits)
    
    # 저장 경로 설정 (사용자가 지정한 절대 경로)
    output_dir = "/data/ephemeral/git/pro-nlp-mrc-nlp-01/data_negative"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("-" * 30)
    print(f"Saving augmented datasets to: {output_dir}")
    final_dataset.save_to_disk(output_dir)
    print("All tasks completed successfully.")
    print("-" * 30)

if __name__ == "__main__":
    main()