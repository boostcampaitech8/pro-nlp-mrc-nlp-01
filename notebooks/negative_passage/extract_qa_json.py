"""
데이터셋에서 질문과 답변을 JSON 형식으로 추출하는 스크립트

- train_dataset: 질문과 답변 추출
- validation_dataset: 질문과 답변 추출
- test_dataset: 질문과 정답 추출 (답변 없음이어도 answer 필드 포함)
"""

import json
import sys
from pathlib import Path
from datasets import load_from_disk
from tqdm.auto import tqdm

# 프로젝트 루트 경로 설정
from notebooks.utils import setup_project_path, extract_answer_from_example

project_root = setup_project_path()

# 데이터 경로 설정
data_root = project_root / "data"
train_dataset_path = data_root / "train_dataset"
test_dataset_path = data_root / "test_dataset"

# 출력 디렉토리
output_dir = project_root / "notebooks" / "negative_passage"
output_dir.mkdir(parents=True, exist_ok=True)


def extract_qa_json(dataset, dataset_name: str, include_answer: bool = True):
    """
    데이터셋에서 질문과 답변을 JSON 형식으로 추출

    Args:
        dataset: HuggingFace Dataset 객체
        dataset_name: 데이터셋 이름 (로깅용)
        include_answer: 답변 포함 여부 (test는 False이어도 answer 필드 포함)

    Returns:
        qa_list: 질문-답변 리스트
    """
    from typing import List, Dict
    from datasets import Dataset
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} 데이터 처리 중...")
    print(f"{'='*60}")

    qa_list: List[Dict[str, str]] = []
    skipped_count = 0

    for example in tqdm(dataset, desc=f"Processing {dataset_name}"):
        question = example.get('question', '').strip()
        example_id = example.get('id', '')

        # 공통 유틸리티 함수 사용
        answer = extract_answer_from_example(example) or ""
        
        # include_answer True 혹은 False에 상관없이 answer 필드를 항상 뽑음
        if question:
            qa_list.append({
                'id': example_id,
                'question': question,
                'answer': answer
            })
        else:
            skipped_count += 1

    print(f"\n총 {len(qa_list)}개 항목 추출 완료")
    if skipped_count > 0:
        print(f"건너뛴 샘플: {skipped_count}개")

    return qa_list


def main():
    """메인 함수"""
    print("="*60)
    print("질문-답변 JSON 추출 스크립트")
    print("="*60)

    # 1. Train 데이터셋 로드
    print(f"\n데이터셋 로드 중...")
    try:
        train_datasets = load_from_disk(str(train_dataset_path))
        print(f"Train dataset 로드 완료!")
        print(f"  - Train: {len(train_datasets['train'])} samples")
        print(f"  - Validation: {len(train_datasets['validation'])} samples")
    except Exception as e:
        print(f"❌ Train dataset 로드 실패: {e}")
        return

    # 2. Test 데이터셋 로드
    try:
        test_datasets = load_from_disk(str(test_dataset_path))
        print(f"Test dataset 로드 완료!")
        print(f"  - Test: {len(test_datasets['validation'])} samples")
    except Exception as e:
        print(f"❌ Test dataset 로드 실패: {e}")
        test_datasets = None

    # 3. Train 데이터 추출
    train_qa = extract_qa_json(
        train_datasets['train'],
        dataset_name="Train",
        include_answer=True
    )

    train_output = output_dir / "train_qa.json"
    print(f"\n파일 저장 중: {train_output}")
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_qa, f, ensure_ascii=False, indent=2)
    print(f"✅ Train 데이터 저장 완료: {len(train_qa)}개 항목")

    # 4. Validation 데이터 추출
    val_qa = extract_qa_json(
        train_datasets['validation'],
        dataset_name="Validation",
        include_answer=True
    )

    val_output = output_dir / "validation_qa.json"
    print(f"\n파일 저장 중: {val_output}")
    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(val_qa, f, ensure_ascii=False, indent=2)
    print(f"✅ Validation 데이터 저장 완료: {len(val_qa)}개 항목")

    # 5. Test 데이터 추출 (질문+빈 answer 필드)
    if test_datasets:
        test_qa = extract_qa_json(
            test_datasets['validation'],
            dataset_name="Test",
            include_answer=False
        )

        test_output = output_dir / "test_questions.json"
        print(f"\n파일 저장 중: {test_output}")
        with open(test_output, 'w', encoding='utf-8') as f:
            json.dump(test_qa, f, ensure_ascii=False, indent=2)
        print(f"✅ Test 데이터 저장 완료: {len(test_qa)}개 항목")

    # 요약
    print("\n" + "="*60)
    print("추출 완료!")
    print("="*60)
    print(f"\n생성된 파일:")
    print(f"  1. {train_output}")
    print(f"     - {len(train_qa)}개 질문-답변 쌍")
    print(f"  2. {val_output}")
    print(f"     - {len(val_qa)}개 질문-답변 쌍")
    if test_datasets:
        print(f"  3. {test_output}")
        print(f"     - {len(test_qa)}개 질문")
    print("="*60)


if __name__ == "__main__":
    main()
