"""
질문과 정답 페어쌍을 txt 파일로 추출하는 스크립트

사용법:
    python notebooks/extract_qa_pairs.py

출력:
    - notebooks/qa_pairs_train.txt: 학습 데이터의 질문-정답 페어
    - notebooks/qa_pairs_validation.txt: 검증 데이터의 질문-정답 페어
    - notebooks/qa_pairs_all.txt: 전체 데이터의 질문-정답 페어
"""

import sys
from pathlib import Path
from datasets import load_from_disk

# 프로젝트 루트 경로 설정
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# 데이터 경로 설정
data_root = project_root / "data"
train_dataset_path = data_root / "train_dataset"


def extract_qa_pairs(dataset, output_path: Path, dataset_name: str = ""):
    """
    데이터셋에서 질문과 정답 페어를 추출하여 txt 파일로 저장
    
    Args:
        dataset: HuggingFace Dataset 객체
        output_path: 출력 파일 경로
        dataset_name: 데이터셋 이름 (로깅용)
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} 데이터 처리 중...")
    print(f"{'='*60}")
    
    qa_pairs = []
    skipped_count = 0
    
    for example in dataset:
        question = example.get('question', '').strip()
        answers = example.get('answers', {})
        
        # answers가 딕셔너리인 경우
        if isinstance(answers, dict):
            answer_texts = answers.get('text', [])
        # answers가 리스트인 경우
        elif isinstance(answers, list):
            answer_texts = answers
        else:
            answer_texts = []
        
        # 정답이 있는 경우만 추가
        if answer_texts and len(answer_texts) > 0:
            # 첫 번째 정답 사용 (여러 정답이 있는 경우)
            answer = answer_texts[0].strip()
            if question and answer:
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'id': example.get('id', '')
                })
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    
    # txt 파일로 저장
    print(f"\n총 {len(qa_pairs)}개의 질문-정답 페어 추출 완료")
    if skipped_count > 0:
        print(f"정답이 없는 {skipped_count}개 샘플 건너뜀")
    
    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# 질문-정답 페어 데이터셋\n")
        f.write(f"# 총 {len(qa_pairs)}개 페어\n")
        f.write(f"# 형식: 질문 | 정답\n")
        f.write(f"{'='*60}\n\n")
        
        for idx, pair in enumerate(qa_pairs, 1):
            f.write(f"[{idx}] ID: {pair['id']}\n")
            f.write(f"질문: {pair['question']}\n")
            f.write(f"정답: {pair['answer']}\n")
            f.write(f"{'-'*60}\n\n")
    
    print(f"파일 저장 완료: {output_path}")
    
    return qa_pairs


def main():
    """메인 함수"""
    print("="*60)
    print("질문-정답 페어 추출 스크립트")
    print("="*60)
    
    # 데이터셋 로드
    print(f"\n데이터셋 로드 중: {train_dataset_path}")
    try:
        train_datasets = load_from_disk(str(train_dataset_path))
        print(f"데이터셋 로드 완료!")
        print(f"  - Train: {len(train_datasets['train'])} samples")
        print(f"  - Validation: {len(train_datasets['validation'])} samples")
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return
    
    # 출력 디렉토리 생성
    output_dir = project_root / "notebooks"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Train 데이터 추출
    train_pairs = extract_qa_pairs(
        train_datasets['train'],
        output_dir / "qa_pairs_train.txt",
        "Train"
    )
    
    # 2. Validation 데이터 추출
    val_pairs = extract_qa_pairs(
        train_datasets['validation'],
        output_dir / "qa_pairs_validation.txt",
        "Validation"
    )
    
    # 3. 전체 데이터 통합
    all_pairs = train_pairs + val_pairs
    all_output_path = output_dir / "qa_pairs_all.txt"
    
    print(f"\n{'='*60}")
    print("전체 데이터 통합 중...")
    print(f"{'='*60}")
    
    with open(all_output_path, 'w', encoding='utf-8') as f:
        f.write(f"# 전체 질문-정답 페어 데이터셋\n")
        f.write(f"# Train: {len(train_pairs)}개, Validation: {len(val_pairs)}개\n")
        f.write(f"# 총 {len(all_pairs)}개 페어\n")
        f.write(f"# 형식: 질문 | 정답\n")
        f.write(f"{'='*60}\n\n")
        
        # Train 데이터
        f.write(f"# === Train 데이터 ===\n")
        for idx, pair in enumerate(train_pairs, 1):
            f.write(f"[TRAIN-{idx}] ID: {pair['id']}\n")
            f.write(f"질문: {pair['question']}\n")
            f.write(f"정답: {pair['answer']}\n")
            f.write(f"{'-'*60}\n\n")
        
        # Validation 데이터
        f.write(f"\n# === Validation 데이터 ===\n")
        for idx, pair in enumerate(val_pairs, 1):
            f.write(f"[VAL-{idx}] ID: {pair['id']}\n")
            f.write(f"질문: {pair['question']}\n")
            f.write(f"정답: {pair['answer']}\n")
            f.write(f"{'-'*60}\n\n")
    
    print(f"전체 데이터 파일 저장 완료: {all_output_path}")
    
    # 요약 정보 출력
    print(f"\n{'='*60}")
    print("추출 완료 요약")
    print(f"{'='*60}")
    print(f"Train 페어: {len(train_pairs)}개")
    print(f"Validation 페어: {len(val_pairs)}개")
    print(f"전체 페어: {len(all_pairs)}개")
    print(f"\n출력 파일:")
    print(f"  - {output_dir / 'qa_pairs_train.txt'}")
    print(f"  - {output_dir / 'qa_pairs_validation.txt'}")
    print(f"  - {output_dir / 'qa_pairs_all.txt'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()






