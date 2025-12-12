###############################
# 예시 실행 명령어
# python -m src.training.eval \
#   --output_dir outputs/best_reader_eval \
#   --dataset_name data/train_dataset \
#   --model_name_or_path models/best_reader \
#   --do_eval
#############################
import logging
import os
import sys
import random
import numpy as np
import torch
import evaluate
from typing import NoReturn

from ..config import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk
from .trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from ..utils import check_no_error, postprocess_qa_predictions

logger = logging.getLogger(__name__)

def main():
    # 1. 인자 파싱 (Arguments Parsing)
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Model is from {model_args.model_name_or_path}")
    print(f"Data is from {data_args.dataset_name}")

    # 2. 로깅 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # 3. 시드 설정
    set_seed(training_args.seed)

    # 4. 데이터셋 로드
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # 5. 모델 & 토크나이저 로드
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=True
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
    )

    # 6. 평가 실행 (MRC Evaluation)
    run_evaluation(data_args, training_args, model_args, datasets, tokenizer, model)

def run_evaluation(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:
    
    # Validation 데이터셋 컬럼 확인
    dataset_columns = datasets["validation"].column_names
    q_col = "question" if "question" in dataset_columns else dataset_columns[0]
    ctx_col = "context" if "context" in dataset_columns else dataset_columns[1]
    ans_col = "answers" if "answers" in dataset_columns else dataset_columns[2]

    padding_right = tokenizer.padding_side == "right"
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # ---------------------------------------------------------
    # [전처리 함수] Validation 데이터 토크나이징
    # ---------------------------------------------------------
    def prepare_validation_features(examples):
        q_seq = examples[q_col if padding_right else ctx_col]
        c_seq = examples[ctx_col if padding_right else q_col]
        trunc_mode = "only_second" if padding_right else "only_first"
        
        tokenized_examples = tokenizer(
            q_seq,
            c_seq,
            truncation=trunc_mode,
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, # 모델에 따라 조정 (RoBERTa: False, BERT: True)
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        overflow_to_sample = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if padding_right else 0
            sample_index = overflow_to_sample[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            offset_mapping = tokenized_examples["offset_mapping"][i]
            tokenized_examples["offset_mapping"][i] = [
                o if sequence_ids[k] == context_index else None
                for k, o in enumerate(offset_mapping)
            ]

        return tokenized_examples

    # 전처리 실행
    validation_dataset = datasets["validation"]
    processed_val_data = validation_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=dataset_columns,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # ---------------------------------------------------------
    # [후처리 함수] Post-processing (여기가 핵심!)
    # ---------------------------------------------------------
    def post_processing_function(examples, features, predictions, training_args):
        # utils.py의 수정된 함수가 여기서 호출됩니다.
        processed_preds = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
            version_2_with_negative=True,
            null_score_diff_threshold=0.0,
        )
        
        # EvalPrediction 객체 생성을 위한 포맷팅
        formatted_preds = [
            {"id": k, "prediction_text": v} for k, v in processed_preds.items()
        ]
        
        references = [
            {"id": ex["id"], "answers": ex[ans_col] if len(ex[ans_col]["text"]) > 0 else {"text": [""], "answer_start": [-1]}}
            for ex in datasets["validation"]
        ]
        
        return EvalPrediction(predictions=formatted_preds, label_ids=references)

    # ---------------------------------------------------------
    # [메트릭 함수] 평가 지표 계산
    # ---------------------------------------------------------
    metric = evaluate.load("squad")
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Data Collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,  # 학습 데이터 없음
        eval_dataset=processed_val_data,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # ---------------------------------------------------------
    # [실행] 평가 시작
    # ---------------------------------------------------------
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    
    # 결과 출력 및 저장
    print(metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()