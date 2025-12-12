"""
SPLADE 기반 Sparse Retrieval + MRC Inference with Wandb

실행 예시:
python -m src.inference.inference_splade \
  --output_dir outputs/eval_splade_k100/ \
  --dataset_name data/train_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_eval \
  --eval_retrieval \
  --top_k_retrieval 100 \
  --use_wandb True \
  --wandb_project "retrieval" \
  --wandb_run_name "splade_k100_eval"

python -m src.inference.inference_splade \
  --output_dir outputs/pred_splade_k100/ \
  --dataset_name data/test_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_predict \
  --eval_retrieval \
  --top_k_retrieval 100 \
  --use_wandb True \
  --wandb_project "retrieval" \
  --wandb_run_name "splade_k100_submit"
"""

import logging
import os
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

import evaluate
import numpy as np
from ..config import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
)
from ..retrieval.retrieval_splade import SparseRetrieval
from ..training import QuestionAnsweringTrainer
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


def init_wandb(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    splade_model_name: str = "telepix/PIXIE-Splade-Preview",
):
    """
    Wandb 초기화 및 설정
    
    Args:
        data_args: 데이터 관련 인자
        model_args: 모델 관련 인자
        training_args: 학습 관련 인자
        splade_model_name: SPLADE 모델명
        
    Returns:
        wandb module
    """
    import wandb
    
    # Run name 자동 생성
    run_name = data_args.wandb_run_name
    if run_name is None:
        run_name = f"splade_k{data_args.top_k_retrieval}"
    
    wandb.init(
        project=data_args.wandb_project,
        name=run_name,
        config={
            # Retrieval parameters
            "retrieval_method": "SPLADE",
            "splade_model": splade_model_name,
            "top_k_retrieval": data_args.top_k_retrieval,
            
            # Model parameters
            "model_name_or_path": model_args.model_name_or_path,
            
            # Data parameters
            "dataset_name": data_args.dataset_name,
            "max_seq_length": data_args.max_seq_length,
            "doc_stride": data_args.doc_stride,
            "max_answer_length": data_args.max_answer_length,
            
            # Training parameters
            "output_dir": training_args.output_dir,
            "do_predict": training_args.do_predict,
            "do_eval": training_args.do_eval,
            "seed": training_args.seed,
            
            # Experiment note
            "experiment_note": getattr(data_args, "experiment_note", ""),
        }
    )
    
    print(f"\n{'='*50}")
    print(f"Wandb initialized!")
    print(f"Project: {data_args.wandb_project}")
    print(f"Run name: {run_name}")
    print(f"Retrieval: SPLADE ({splade_model_name})")
    print(f"{'='*50}\n")
    
    return wandb


def main():
    """메인 실행 함수"""
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # 로깅 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Training/evaluation parameters %s", training_args)

    # 재현성을 위한 시드 설정
    set_seed(training_args.seed)

    # Wandb 초기화
    wandb = None
    splade_model_name = "telepix/PIXIE-Splade-Preview"
    if data_args.use_wandb:
        wandb = init_wandb(data_args, model_args, training_args, splade_model_name)

    # 데이터셋 로드
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # 모델 설정 로드
    model_config_path = (
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path
    )
    model_config = AutoConfig.from_pretrained(model_config_path)

    # 토크나이저 로드
    tokenizer_path = (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    # 모델 로드
    is_tensorflow_model = ".ckpt" in model_args.model_name_or_path
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=is_tensorflow_model,
        config=model_config,
    )

    

    # SPLADE Retrieval 수행 여부
    should_run_retrieval = data_args.eval_retrieval
    retrieval_metrics = None

    # 원래 validation 예제 별도로 백업 (질문 단위 GT)
    original_eval_examples = datasets["validation"]

    # SPLADE Retrieval
    if should_run_retrieval:
        datasets, retrieval_metrics = run_splade_retrieval(
            datasets,
            training_args,
            data_args,
            splade_model_name=splade_model_name,
        )

        # Wandb에 retrieval metrics 로깅
        if wandb is not None and retrieval_metrics is not None:
            wandb.log({
                "retrieval/accuracy": retrieval_metrics.get("retrieval_accuracy"),
                "retrieval/mrr": retrieval_metrics.get("mrr"),
                "retrieval/correct_count": retrieval_metrics.get("correct_count"),
                "retrieval/total_count": retrieval_metrics.get("total_count"),
                "retrieval/top_k": retrieval_metrics.get("top_k"),
            })
            print(f"[Wandb] Logged retrieval metrics: {retrieval_metrics}")

    # MRC 수행 여부
    should_run_mrc = training_args.do_eval or training_args.do_predict
    if should_run_mrc:
        mrc_metrics = run_mrc(
            data_args, 
            training_args, 
            model_args, 
            datasets,  # retrieval 결과 (질문 x passage)
            tokenizer, 
            model,
            eval_examples=original_eval_examples, # 원래 validation(질문 단위) 
        )

        # Wandb에 MRC metrics 로깅
        if wandb is not None and mrc_metrics is not None:
            wandb.log({
                "mrc/exact_match": mrc_metrics.get("exact_match"),
                "mrc/f1": mrc_metrics.get("f1"),
            })
            print(f"[Wandb] Logged MRC metrics: {mrc_metrics}")

    # Wandb run 종료
    if wandb is not None:
        import wandb as _wandb
        _wandb.finish()
        print("[Wandb] Run finished successfully!")


def run_splade_retrieval(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "data",
    context_path: str = "wikipedia_documents.json",
    splade_model_name: str = "telepix/PIXIE-Splade-Preview",
) -> Tuple[DatasetDict, Dict]:
    """
    SPLADE 기반 sparse retrieval 수행 및 metrics 계산
    
    Args:
        datasets: 입력 데이터셋
        training_args: 학습 관련 인자
        data_args: 데이터 관련 인자
        data_path: 데이터 디렉토리 경로
        context_path: Wikipedia 문서 JSON 파일명
        splade_model_name: SPLADE 모델명
        
    Returns:
        (검색 결과 DatasetDict, retrieval metrics dict)
    """
    logger.info("*** Running SPLADE Retrieval ***")
    
    # SPLADE Retrieval 초기화
    splade_retriever = SparseRetrieval(
        data_path=data_path,
        context_path=context_path,
        splade_model_name=splade_model_name,
    )
    
    # Sparse embedding 생성 또는 로드
    logger.info("Building or loading SPLADE embeddings...")
    splade_retriever.get_sparse_embedding()
    
    # Retrieval 수행
    topk = data_args.top_k_retrieval
    logger.info(f"Retrieving top-{topk} passages...")
    retrieved_df = splade_retriever.retrieve(
        datasets["validation"], topk=topk
    )
    
    logger.info(f"Retrieved {len(retrieved_df)} (question, passage) pairs")

    # 📊 Retrieval metrics 계산
    retrieval_metrics = None
    if "original_context" in retrieved_df.columns:
        # id(질문)별로 groupby
        grouped = retrieved_df.groupby("id")
        total_questions = grouped.ngroups
        correct_questions = 0 

        def check_retrieval_success_for_group(group):
            original = group["original_context"].iloc[0].strip()
            
            for retrieved in group["context"]:
                retrieved = retrieved.strip()

                if original in retrieved:
                    return True
            
                # 긴 context의 경우: 앞뒤 100자씩 샘플링해서 둘 다 포함되면 성공
                if len(original) >= 100:
                    start_sample = original[:100]
                    end_sample = original[-100:]
                    if (start_sample in retrieved) and (end_sample in retrieved):
                        return True
        
            return False

        for _, group in grouped:
            if check_retrieval_success_for_group(group):
                correct_questions += 1

        accuracy = correct_questions / total_questions if total_questions > 0 else 0.0


        retrieval_metrics = {
            "retrieval_accuracy": accuracy,
            "correct_count": correct_questions,
            "total_count": total_questions,
            "top_k": topk,
            "mrr": None,
        }
        
        logger.info(f"Retrieval Accuracy(question-level): {accuracy:.4f} ({correct_questions}/{total_questions})")

        # MRC 학습용으로는 original_context 필요 없으니 제거
        retrieved_df = retrieved_df.drop(columns=["original_context"])

    # Dataset Features 정의
    dataset_features = None
    is_predict_mode = training_args.do_predict
    is_eval_mode = training_args.do_eval

    if is_predict_mode:
        # Predict 모드: answers 없음, 대신 retrieval_rank/score 포함 
        dataset_features = Features(
            {
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                "context": Value(dtype="string", id=None),
                "retrieval_rank": Value("int32"),
                "retrieval_score": Value("float32"),
            }
        )
    elif is_eval_mode:
        # 🔵 eval 모드: answers + retrieval_rank/score 포함
        dataset_features = Features(
            {
                "id": Value("string"),
                "question": Value("string"),
                "context": Value("string"),
                "retrieval_rank": Value("int32"),
                "retrieval_score": Value("float32"),
                "answers": Sequence(
                    feature={
                        "text": Value("string"),
                        # pandas에서 int64로 들어오니까 여기도 int64로 맞추기
                        "answer_start": Value("int64"),
                    },
                    length=-1,
                ),
            }
        )

    # DataFrame을 Dataset으로 변환
    result_datasets = DatasetDict(
        {"validation": Dataset.from_pandas(retrieved_df, features=dataset_features)}
    )
    
    return result_datasets, retrieval_metrics


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    eval_examples,
) -> Dict:
    """
    Machine Reading Comprehension 수행 및 metrics 반환
    
    Args:
        data_args: 데이터 관련 인자
        training_args: 학습 관련 인자
        model_args: 모델 관련 인자
        datasets: 데이터셋
        tokenizer: 토크나이저
        model: QA 모델
        
    Returns:
        MRC metrics dict (exact_match, f1)
    """
    # Validation 데이터셋의 컬럼명 확인
    validation_dataset = datasets["validation"]   # ← retrieval 결과 (질문×passage)    
    val_column_names = validation_dataset.column_names

    has_question_col = "question" in val_column_names
    has_context_col = "context" in val_column_names
    has_answers_col = "answers" in val_column_names

    # 컬럼명 매핑
    question_col = "question" if has_question_col else val_column_names[0]
    context_col = "context" if has_context_col else val_column_names[1]
    answer_col = "answers" if has_answers_col else val_column_names[2]

    # Padding 방향 확인
    is_padding_right = tokenizer.padding_side == "right"

    # 에러 체크 및 설정값 가져오기
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    def prepare_validation_features(examples):
        """
        Validation 데이터 전처리 함수
        
        - Tokenization
        - Stride를 이용한 긴 문서 처리
        - Offset mapping으로 답변 위치 추적
        """
        # Padding 방향에 따라 입력 순서 결정
        first_seq = examples[question_col if is_padding_right else context_col]
        second_seq = examples[context_col if is_padding_right else question_col]
        truncation_mode = "only_second" if is_padding_right else "only_first"
        padding_mode = "max_length" if data_args.pad_to_max_length else False

        # Tokenization with stride
        tokenized_examples = tokenizer(
            first_seq,
            second_seq,
            truncation=truncation_mode,
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=padding_mode,
            return_token_type_ids=False,
        )

        # Overflow mapping
        overflow_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        num_tokenized = len(tokenized_examples["input_ids"])

        for idx in range(num_tokenized):
            seq_ids = tokenized_examples.sequence_ids(idx)
            ctx_idx = 1 if is_padding_right else 0
            orig_sample_idx = overflow_mapping[idx]
            
            tokenized_examples["example_id"].append(examples["id"][orig_sample_idx])

            # Offset mapping: context 부분만 유지
            current_offset_map = tokenized_examples["offset_mapping"][idx]
            new_offset_map = []
            for pos, offset_val in enumerate(current_offset_map):
                if seq_ids[pos] == ctx_idx:
                    new_offset_map.append(offset_val)
                else:
                    new_offset_map.append(None)
            tokenized_examples["offset_mapping"][idx] = new_offset_map
            
        return tokenized_examples

    processed_dataset = validation_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=val_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # 🔵 RoBERTa에서는 token_type_ids 안 쓰도록 제거 (혹시 남아있어도 날리기)
    if "token_type_ids" in processed_dataset.column_names and model.config.model_type == "roberta":
        processed_dataset = processed_dataset.remove_columns("token_type_ids")


    # Data collator 설정
    pad_multiple = 8 if training_args.fp16 else None
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=pad_multiple
    )

    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        """예측 결과 후처리"""
        # Logits를 실제 답변 텍스트로 변환
        processed_predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        
        # SQuAD 형식으로 변환
        formatted_predictions = []
        for prediction_id, prediction_text in processed_predictions.items():
            formatted_predictions.append(
                {"id": prediction_id, "prediction_text": prediction_text}
            )

        is_predict = training_args.do_predict
        is_eval = training_args.do_eval

        if is_predict:
            return formatted_predictions
        elif is_eval:
            reference_list = []
            for val_example in eval_examples:
                reference_list.append(
                    {
                        "id": val_example["id"], 
                        "answers": val_example[answer_col], # 원래 데이터셋 기준 
                    }
                )

            return EvalPrediction(
                predictions=formatted_predictions, 
                label_ids=reference_list
            )

    # SQuAD 메트릭 로드
    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        """메트릭 계산"""
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Trainer 초기화
    print("init trainer...")
    qa_trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=processed_dataset,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    should_predict = training_args.do_predict
    should_eval = training_args.do_eval
    
    mrc_metrics = None

    # Predict 수행
    if should_predict:
        prediction_results = qa_trainer.predict(
            test_dataset=processed_dataset, test_examples=datasets["validation"]
        )

        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    # Evaluation 수행
    if should_eval:
        eval_metrics = qa_trainer.evaluate()
        eval_metrics["eval_samples"] = len(processed_dataset)

        # 메트릭 로깅 및 저장
        qa_trainer.log_metrics("test", eval_metrics)
        qa_trainer.save_metrics("test", eval_metrics)

        # Wandb용 metrics 추출
        mrc_metrics = {
            "exact_match": eval_metrics.get("exact_match"),
            "f1": eval_metrics.get("f1"),
        }
        
        logger.info(f"MRC Metrics - EM: {mrc_metrics['exact_match']:.4f}, F1: {mrc_metrics['f1']:.4f}")

    return mrc_metrics


if __name__ == "__main__":
    main()