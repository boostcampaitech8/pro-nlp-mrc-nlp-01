"""
Random Negative 1:1 데이터셋으로 학습한 모델의 틀린 케이스 추출

본 스크립트는:
1. 1:1 random negative 데이터셋으로 학습된 모델을 평가
2. 틀린 케이스(예측값과 정답이 다른 경우)를 추출
3. 오류와 정답 쌍 리스트를 파일로 저장
"""

# OpenMP 충돌 방지 설정 (Windows 환경)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 필요한 라이브러리 import
import sys
import json
from pathlib import Path
from datasets import load_from_disk
from typing import List, Dict
from tqdm.auto import tqdm
import evaluate

# 프로젝트 루트 경로 설정 (스크립트 파일 위치 기준)
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.append(str(project_root))

# src 모듈 import
from src.utils import postprocess_qa_predictions
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch


def normalize_answer(s: str) -> str:
    """정답 정규화 (비교를 위해)"""
    import re
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s


def is_exact_match(prediction: str, ground_truth: str) -> bool:
    """Exact Match 여부 확인"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def extract_wrong_cases(
    predictions: Dict[str, str],
    eval_examples: List[Dict],
    output_file: Path
) -> List[Dict]:
    """
    틀린 케이스를 추출하여 파일로 저장
    
    Args:
        predictions: 예측 결과 딕셔너리 {id: prediction_text}
        eval_examples: 평가 데이터셋 예시 리스트
        output_file: 출력 파일 경로
    
    Returns:
        wrong_cases: 틀린 케이스 리스트
    """
    wrong_cases = []
    
    print("\n틀린 케이스 추출 중...")
    
    for example in tqdm(eval_examples, desc="Extracting wrong cases"):
        example_id = example['id']
        question = example['question']
        context = example['context']
        answers = example.get('answers', {})
        answer_texts = answers.get('text', [])
        
        # 정답이 없는 경우는 제외 (Negative sample)
        if not answer_texts:
            continue
        
        ground_truth = answer_texts[0]
        prediction = predictions.get(example_id, "")
        
        # Exact Match가 아닌 경우 틀린 케이스로 분류
        if not is_exact_match(prediction, ground_truth):
            wrong_case = {
                'id': example_id,
                'question': question,
                'context': context,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'is_exact_match': False
            }
            wrong_cases.append(wrong_case)
    
    # 파일로 저장
    print(f"\n틀린 케이스 {len(wrong_cases)}개 발견")
    print(f"파일 저장 중: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(wrong_cases, f, ensure_ascii=False, indent=2)
    
    print(f"저장 완료!")
    
    return wrong_cases


def load_model_and_predict(
    model_path: str,
    eval_dataset,
    eval_examples: List[Dict],
    tokenizer,
    config
) -> Dict[str, str]:
    """
    모델을 로드하고 예측 수행
    
    Args:
        model_path: 모델 체크포인트 경로
        eval_dataset: 평가 데이터셋 (전처리된)
        eval_examples: 평가 데이터셋 예시 (원본)
        tokenizer: 토크나이저
        config: 실험 설정
    
    Returns:
        predictions: 예측 결과 딕셔너리
    """
    from transformers import TrainingArguments, DataCollatorWithPadding
    from src.training.trainer_qa import QuestionAnsweringTrainer
    
    # 모델 로드
    print(f"\n모델 로드 중: {model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    print("모델 로드 완료")
    
    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir="./temp_eval",
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Data Collator
    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )
    
    # Post-processing 함수
    def post_process_function(examples, features, predictions, args):
        # examples가 리스트인 경우 Dataset으로 변환
        if isinstance(examples, list):
            from datasets import Dataset
            examples = Dataset.from_list(examples)
        
        processed_preds = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=config.max_answer_length,
            output_dir=args.output_dir,
        )
        # 딕셔너리 형태로 반환 (id: prediction_text)
        return processed_preds
    
    # Compute metrics 함수 (더미 - predict()가 post_process_function을 호출하도록 하기 위함)
    def compute_metrics(p):
        return {}
    
    # Trainer 생성
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_process_function,
        compute_metrics=compute_metrics,
    )
    
    # 예측 수행 (test_examples 인자 필요)
    print("\n예측 수행 중...")
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        training_args.output_dir = temp_dir
        result = trainer.predict(
            test_dataset=eval_dataset,
            test_examples=eval_examples
        )
    
    # predict()는 post_process_function의 반환값(딕셔너리)을 반환
    # 만약 EvalLoopOutput이 반환되면 직접 처리
    if hasattr(result, 'predictions'):
        # EvalLoopOutput인 경우 직접 post-processing
        processed_predictions = post_process_function(
            eval_examples, eval_dataset, result.predictions, training_args
        )
        return processed_predictions
    else:
        # 이미 딕셔너리인 경우
        return result


def main():
    """메인 함수"""
    from dataclasses import dataclass
    
    @dataclass
    class ExperimentConfig:
        """실험 설정"""
        model_name: str = "klue/bert-base"
        max_seq_length: int = 384
        doc_stride: int = 128
        max_answer_length: int = 30
        per_device_eval_batch_size: int = 32
    
    # 설정
    config = ExperimentConfig()
    
    # 경로 설정
    data_root = project_root / "data"
    dataset_path = data_root / "train_dataset_1to1_random_negative"
    
    # 모델 경로 설정 (학습된 모델이 있는 경우)
    # 명령줄 인자로 받거나 자동으로 찾기
    import argparse
    parser = argparse.ArgumentParser(description='틀린 케이스 추출')
    parser.add_argument('--model_path', type=str, default=None,
                       help='학습된 모델 경로 (예: notebooks/negative_passage/experiments/random_neg_1/checkpoint-xxx)')
    parser.add_argument('--auto_find', action='store_true',
                       help='experiments 디렉토리에서 자동으로 최신 모델 찾기')
    args = parser.parse_args()
    
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = project_root / model_path
    elif args.auto_find:
        # experiments 디렉토리에서 random 관련 모델 찾기
        experiments_dir = project_root / "notebooks" / "negative_passage" / "experiments"
        if experiments_dir.exists():
            # random_neg_1 또는 random_neg_3 디렉토리 찾기
            random_dirs = list(experiments_dir.glob("random_neg_*"))
            if random_dirs:
                # 가장 최근 체크포인트 찾기
                latest_dir = max(random_dirs, key=lambda p: p.stat().st_mtime)
                checkpoints = list(latest_dir.glob("checkpoint-*"))
                if checkpoints:
                    model_path = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
                    print(f"자동으로 찾은 모델: {model_path}")
                else:
                    model_path = latest_dir
                    print(f"체크포인트를 찾지 못해 디렉토리 사용: {model_path}")
            else:
                print("random_neg 관련 실험 디렉토리를 찾을 수 없습니다.")
                return
        else:
            print("experiments 디렉토리를 찾을 수 없습니다.")
            return
    else:
        # 대화형 입력
        model_path_input = input("학습된 모델 경로를 입력하세요 (예: notebooks/negative_passage/experiments/random_neg_1/checkpoint-xxx): ").strip()
        if not model_path_input:
            print("모델 경로가 입력되지 않았습니다.")
            return
        model_path = Path(model_path_input)
        if not model_path.is_absolute():
            model_path = project_root / model_path
    
    if not model_path.exists():
        print(f"모델 경로가 존재하지 않습니다: {model_path}")
        return
    
    print(f"사용할 모델: {model_path}")
    
    # 출력 파일 경로
    output_file = project_root / "notebooks" / "negative_passage" / "wrong_cases_random_1to1.json"
    
    # 데이터셋 로드
    print(f"\n데이터셋 로드 중: {dataset_path}")
    datasets = load_from_disk(str(dataset_path))
    eval_dataset_raw = datasets['validation']
    
    # Positive 예시만 필터링 (Negative 제외)
    print("\nPositive 예시만 필터링 중...")
    eval_examples = [
        ex for ex in eval_dataset_raw 
        if not ex.get('is_negative', False) and len(ex.get('answers', {}).get('text', [])) > 0
    ]
    print(f"평가 예시 개수: {len(eval_examples)}")
    
    # 토크나이저 로드
    print(f"\n토크나이저 로드 중: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    
    # 데이터 전처리 (05번 노트북과 동일한 방식)
    print("\n데이터 전처리 중...")
    from transformers import TrainingArguments
    
    def prepare_validation_features(examples):
        """검증 데이터 전처리"""
        padding_right = tokenizer.padding_side == "right"
        
        tokenized_examples = tokenizer(
            examples['question'] if padding_right else examples['context'],
            examples['context'] if padding_right else examples['question'],
            truncation="only_second" if padding_right else "only_first",
            max_length=config.max_seq_length,
            stride=config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        overflow_to_sample = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        
        total_examples = len(tokenized_examples["input_ids"])
        for example_index in range(total_examples):
            seq_id_list = tokenized_examples.sequence_ids(example_index)
            context_id = 1 if padding_right else 0
            orig_sample_index = overflow_to_sample[example_index]
            tokenized_examples["example_id"].append(eval_examples[orig_sample_index]["id"])
            
            current_offsets = tokenized_examples["offset_mapping"][example_index]
            updated_offsets = []
            for pos_idx, offset_value in enumerate(current_offsets):
                if seq_id_list[pos_idx] == context_id:
                    updated_offsets.append(offset_value)
                else:
                    updated_offsets.append(None)
            tokenized_examples["offset_mapping"][example_index] = updated_offsets
        
        return tokenized_examples
    
    from datasets import Dataset
    eval_dataset_processed = Dataset.from_list(eval_examples).map(
        prepare_validation_features,
        batched=True,
        remove_columns=Dataset.from_list(eval_examples).column_names,
        desc="Processing eval"
    )
    
    # 모델 로드 및 예측
    predictions = load_model_and_predict(
        str(model_path),
        eval_dataset_processed,
        eval_examples,
        tokenizer,
        config
    )
    
    # 틀린 케이스 추출
    wrong_cases = extract_wrong_cases(
        predictions,
        eval_examples,
        output_file
    )
    
    # 요약 출력
    print("\n" + "="*80)
    print("틀린 케이스 추출 완료!")
    print("="*80)
    print(f"\n총 평가 예시: {len(eval_examples)}개")
    print(f"틀린 케이스: {len(wrong_cases)}개")
    print(f"정확도: {(len(eval_examples) - len(wrong_cases)) / len(eval_examples) * 100:.2f}%")
    print(f"\n저장 위치: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()

