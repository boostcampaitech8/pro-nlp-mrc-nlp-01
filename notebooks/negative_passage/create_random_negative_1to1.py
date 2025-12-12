"""
7. Random Negative Passage 1:1 비율 생성

본 스크립트는 1:1 비율로 Random Negative Passage를 생성합니다.
각 Positive 예시마다 정확히 1개의 Random Negative Passage를 생성합니다.
"""

# OpenMP 충돌 방지 설정 (Windows 환경)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 필요한 라이브러리 import
import sys
import json
import random
from pathlib import Path
from datasets import Dataset, DatasetDict, load_from_disk
from typing import List
from tqdm.auto import tqdm
import re

# 프로젝트 루트 경로 설정
from notebooks.utils import setup_project_path, extract_answer_from_example

project_root = setup_project_path()

# 재현성을 위한 시드 설정
SEED = 42
random.seed(SEED)

print(f"프로젝트 루트: {project_root}")


def check_answer_in_context(answer_text: str, context: str) -> bool:
    """Context에 정답이 포함되어 있는지 확인"""
    if not answer_text or not context:
        return False
    answer_normalized = re.sub(r'\s+', ' ', answer_text.strip().lower())
    context_normalized = re.sub(r'\s+', ' ', context.strip().lower())
    return answer_normalized in context_normalized


def sample_random_negative(
    positive_context: str,
    answer_text: str,
    all_contexts: List[str],
    max_attempts: int = 100
) -> str:
    """
    랜덤 Negative Passage 1개 샘플링
    
    Args:
        positive_context: Positive context (제외할 문서)
        answer_text: 정답 텍스트 (포함되지 않아야 함)
        all_contexts: 전체 문서 리스트
        max_attempts: 최대 시도 횟수
    
    Returns:
        negative_context: Negative passage (없으면 빈 문자열)
    """
    attempts = 0
    
    while attempts < max_attempts:
        idx = random.randint(0, len(all_contexts) - 1)
        candidate = all_contexts[idx]
        
        # 조건: positive가 아니고, 정답이 포함되지 않은 문서
        if candidate != positive_context and not check_answer_in_context(answer_text, candidate):
            return candidate
        attempts += 1
    
    # 실패 시 빈 문자열 반환
    print(f"Warning: Negative passage를 찾지 못했습니다. (시도 횟수: {max_attempts})")
    return ""


def create_1to1_negative_dataset(
    original_dataset,
    all_contexts: List[str],
    dataset_name: str = "train"
) -> Dataset:
    """
    1:1 비율로 Random Negative Passage를 포함한 데이터셋 생성
    
    Args:
        original_dataset: 원본 데이터셋
        all_contexts: 전체 Wikipedia 문서 리스트
        dataset_name: 데이터셋 이름 (로깅용)
    
    Returns:
        augmented_dataset: 증강된 데이터셋 (Positive:Negative = 1:1)
    """
    augmented_data = {
        'id': [],
        'title': [],
        'context': [],
        'question': [],
        'answers': [],
        'is_negative': []  # 메타데이터
    }
    
    print(f"\n[{dataset_name}] 1:1 비율 Negative Passage 생성 중...")
    
    failed_count = 0
    
    for example in tqdm(original_dataset, desc=f"Augmenting {dataset_name}"):
        question = example['question']
        positive_context = example['context']
        answer_text = extract_answer_from_example(example) or ''
        
        # 1. Positive 예시 추가
        augmented_data['id'].append(example['id'])
        augmented_data['title'].append(example.get('title', ''))
        augmented_data['context'].append(positive_context)
        augmented_data['question'].append(question)
        augmented_data['answers'].append(answers)
        augmented_data['is_negative'].append(False)
        
        # 2. Negative 예시 추가 (1:1 비율)
        if answer_text:
            negative_context = sample_random_negative(
                positive_context, answer_text, all_contexts
            )
            
            if negative_context:  # Negative를 찾은 경우만 추가
                augmented_data['id'].append(f"{example['id']}_neg")
                augmented_data['title'].append(example.get('title', ''))
                augmented_data['context'].append(negative_context)
                augmented_data['question'].append(question)
                # Negative의 경우 빈 정답
                augmented_data['answers'].append({'text': [], 'answer_start': []})
                augmented_data['is_negative'].append(True)
            else:
                failed_count += 1
    
    augmented_dataset = Dataset.from_dict(augmented_data)
    
    positive_count = sum(1 for x in augmented_data['is_negative'] if not x)
    negative_count = sum(1 for x in augmented_data['is_negative'] if x)
    
    print(f"\n[{dataset_name}] 증강 완료:")
    print(f"  - 원본: {len(original_dataset)} samples")
    print(f"  - 증강 후: {len(augmented_dataset)} samples")
    print(f"  - Positive: {positive_count} samples")
    print(f"  - Negative: {negative_count} samples")
    print(f"  - 비율: {positive_count}:{negative_count} = 1:{negative_count/positive_count if positive_count > 0 else 0:.2f}")
    if failed_count > 0:
        print(f"  - Warning: Negative를 찾지 못한 예시: {failed_count}개")
    
    return augmented_dataset


if __name__ == "__main__":
    # 데이터 경로 설정
    data_root = project_root / "data"
    train_dataset_path = data_root / "train_dataset"
    wikipedia_documents_path = data_root / "wikipedia_documents.json"
    
    # 데이터 로드
    print("\n데이터셋 로드 중...")
    train_datasets = load_from_disk(str(train_dataset_path))
    
    with open(wikipedia_documents_path, 'r', encoding='utf-8') as f:
        wiki_documents = json.load(f)
    
    print(f"Train dataset: {len(train_datasets['train'])} samples")
    print(f"Validation dataset: {len(train_datasets['validation'])} samples")
    print(f"Wikipedia documents: {len(wiki_documents)} documents")
    
    # Wikipedia 문서를 리스트로 변환 (중복 제거)
    context_texts = [v["text"] for v in wiki_documents.values()]
    all_contexts = list(dict.fromkeys(context_texts))  # 중복 제거
    print(f"\n중복 제거 후 Wikipedia 문서 개수: {len(all_contexts)}")
    
    # Train 데이터셋 생성 (1:1 비율)
    augmented_train = create_1to1_negative_dataset(
        train_datasets['train'],
        all_contexts,
        dataset_name="train"
    )
    
    # Validation 데이터셋 생성 (1:1 비율)
    augmented_val = create_1to1_negative_dataset(
        train_datasets['validation'],
        all_contexts,
        dataset_name="validation"
    )
    
    # 데이터셋 저장
    output_dir = project_root / "data" / "train_dataset_1to1_random_negative"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DatasetDict 생성
    augmented_datasets = DatasetDict({
        'train': augmented_train,
        'validation': augmented_val
    })
    
    # 저장
    print(f"\n데이터셋 저장 중: {output_dir}")
    augmented_datasets.save_to_disk(str(output_dir))
    print(f"저장 완료!")
    
    # 검증
    print("\n" + "="*80)
    print("1:1 비율 Random Negative Passage 데이터셋 생성 완료!")
    print("="*80)
    print(f"\n저장 위치: {output_dir}")
    print(f"\n사용 방법:")
    print(f"  from datasets import load_from_disk")
    print(f"  datasets = load_from_disk('{output_dir}')")
    print(f"  train_dataset = datasets['train']")
    print(f"  val_dataset = datasets['validation']")
    print("\n" + "="*80)

