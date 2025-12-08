import torch
import random
from datasets import load_from_disk
from transformers import AutoTokenizer

# ==========================================
# 사용자 설정 (train.py 실행 시의 인자값과 맞춰주세요)
# ==========================================
dataset_path = "/data/ephemeral/git/pro-nlp-mrc-nlp-01/data/train_dataset"  # dataset.arrow가 있는 폴더
model_name = "HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned"             # 사용 중인 모델 이름
max_seq_length = 384                    # max_seq_length 설정값
doc_stride = 128                        # doc_stride 설정값
# ==========================================

def inspect_random_data():
    # 1. 데이터와 토크나이저 로드
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    total_len = len(dataset['train'])
    print(f"▶ 전체 학습 데이터 개수: {total_len}개")

    # [추가 2] 0부터 전체 개수 사이에서 중복 없이 '3개' 숫자를 랜덤으로 뽑기
    # 데이터가 3개보다 적을 경우를 대비해 min 처리
    num_samples = min(3, total_len)
    random_indices = random.sample(range(total_len), num_samples)
    
    print(f"▶ 이번에 확인할 랜덤 인덱스: {random_indices}\n")

    # 랜덤으로 뽑은 인덱스들을 하나씩 돌면서 출력
    for step, idx in enumerate(random_indices):
        print(f"============================================================")
        print(f"   [랜덤 샘플 {step+1}/3] (Index: {idx})")
        print(f"============================================================")
        
        example = dataset['train'][idx]  # <--- 랜덤 인덱스 사용
        
        # --- (1) 원본 확인 ---
        print(f"ID: {example.get('id', 'N/A')}")
        print(f"Question: {example['question']}")
        # 지문은 너무 길 수 있으니 앞부분 100자만 출력
        context_preview = example['context'][:].replace('\n', ' ') 
        print(f"Context (앞부분): {context_preview}...")
        print(f"Answers: {example['answers']}")

        # --- (2) 전처리 및 토큰화 ---
        tokenized_inputs = tokenizer(
            example["question"],
            example["context"],
            truncation="only_second",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, 
            padding="max_length",
        )

        input_ids = tokenized_inputs["input_ids"][0]
        decoded = tokenizer.decode(input_ids)

        # --- (3) 모델 입력값 확인 ---
        print(f"\n▶ Decoded Input (모델이 보는 텍스트):\n{decoded}")
        
        # --- (4) 정답 포함 여부 체크 ---
        answer_text = example['answers']['text'][0]
        if answer_text in decoded:
            print(f"\n▶ 상태: ✅ 정답('{answer_text}')이 입력 안에 잘 포함됨")
        else:
            print(f"\n▶ 상태: ⚠️ 정답('{answer_text}')이 잘렸거나(Truncation), 특수문자 변환됨")
        
        print("\n\n") # 보기 좋게 줄바꿈

if __name__ == "__main__":
    inspect_random_data()