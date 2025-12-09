"""
BGE-M3 기반 Hybrid Retrieval + MRC Inference

실행 예시:
# Dense + Sparse (추천)
python -m src.inference.inference_bge_m3 \
  --output_dir outputs/eval_bge_m3_k100/ \
  --dataset_name data/train_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_eval \
  --eval_retrieval \
  --top_k_retrieval 100 \
  --use_wandb True \
  --wandb_project "retrieval" \
  --wandb_run_name "bge_m3_dense_sparse_k100"

# Dense만 사용 (빠름)
python -m src.inference.inference_bge_m3 \
  --output_dir outputs/eval_bge_m3_dense_k100/ \
  --dataset_name data/train_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_eval \
  --eval_retrieval \
  --top_k_retrieval 100 \
  --bge_use_dense True \
  --bge_use_sparse False \
  --bge_use_colbert False

python -m src.inference.inference_bge_m3 \
  --output_dir outputs/eval_bge_m3_k100/ \
  --dataset_name data/train_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_eval \
  --eval_retrieval \
  --top_k_retrieval 100 \
  --bge_use_dense True \
  --bge_use_sparse True \
  --bge_use_colbert False \
  --bge_dense_weight 0.5 \
  --bge_sparse_weight 0.5

"""

import logging
import os
import sys
from typing import Dict, Tuple

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
from ..retrieval.retrieval_bge_m3 import BGEM3Retrieval
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
    bge_config: dict,
):
    """Wandb 초기화"""
    import wandb
    
    # 사용 중인 방법 식별
    methods = []
    if bge_config['use_dense']:
        methods.append('dense')
    if bge_config['use_sparse']:
        methods.append('sparse')
    if bge_config['use_colbert']:
        methods.append('colbert')
    method_str = '+'.join(methods)
    
    run_name = data_args.wandb_run_name
    if run_name is None:
        run_name = f"bge_m3_{method_str}_k{data_args.top_k_retrieval}"
    
    wandb.init(
        project=data_args.wandb_project,
        name=run_name,
        config={
            # Retrieval parameters
            "retrieval_method": "BGE-M3",
            "bge_methods": method_str,
            "use_dense": bge_config['use_dense'],
            "use_sparse": bge_config['use_sparse'],
            "use_colbert": bge_config['use_colbert'],
            "dense_weight": bge_config['weights']['dense'],
            "sparse_weight": bge_config['weights']['sparse'],
            "colbert_weight": bge_config['weights']['colbert'],
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
        }
    )
    
    print(f"\n{'='*50}")
    print(f"Wandb initialized!")
    print(f"Project: {data_args.wandb_project}")
    print(f"Run name: {run_name}")
    print(f"Retrieval: BGE-M3 ({method_str})")
    print(f"Weights: D={bge_config['weights']['dense']:.2f}, "
          f"S={bge_config['weights']['sparse']:.2f}, "
          f"C={bge_config['weights']['colbert']:.2f}")
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

    # 시드 설정
    set_seed(training_args.seed)

    # BGE-M3 설정
    bge_config = {
        'use_dense': getattr(data_args, 'bge_use_dense', True),
        'use_sparse': getattr(data_args, 'bge_use_sparse', True),
        'use_colbert': getattr(data_args, 'bge_use_colbert', False),
        'batch_size': getattr(data_args, 'bge_batch_size', 12),
        'max_length': getattr(data_args, 'bge_max_length', 512),
        'weights': {
            'dense': getattr(data_args, 'bge_dense_weight', 0.5),
            'sparse': getattr(data_args, 'bge_sparse_weight', 0.5),
            'colbert': getattr(data_args, 'bge_colbert_weight', 0.0),
        }
    }

    # Wandb 초기화
    wandb = None
    if data_args.use_wandb:
        wandb = init_wandb(data_args, model_args, training_args, bge_config)

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

    # BGE-M3 Retrieval 수행
    should_run_retrieval = data_args.eval_retrieval
    retrieval_metrics = None
    original_eval_examples = datasets["validation"]

    if should_run_retrieval:
        datasets, retrieval_metrics = run_bge_m3_retrieval(
            datasets,
            training_args,
            data_args,
            bge_config=bge_config,
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

    # MRC 수행
    should_run_mrc = training_args.do_eval or training_args.do_predict
    if should_run_mrc:
        mrc_metrics = run_mrc(
            data_args, 
            training_args, 
            model_args, 
            datasets,
            tokenizer, 
            model,
            eval_examples=original_eval_examples,
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


def run_bge_m3_retrieval(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "data",
    context_path: str = "wikipedia_documents.json",
    bge_config: dict = None,
) -> Tuple[DatasetDict, Dict]:
    """
    BGE-M3 기반 retrieval 수행 및 metrics 계산
    """
    logger.info("*** Running BGE-M3 Retrieval ***")
    
    if bge_config is None:
        bge_config = {
            'use_dense': True,
            'use_sparse': True,
            'use_colbert': False,
            'batch_size': 12,
            'max_length': 512,
            'weights': {'dense': 0.5, 'sparse': 0.5, 'colbert': 0.0}
        }
    
    # BGE-M3 Retrieval 초기화
    bge_retriever = BGEM3Retrieval(
        data_path=data_path,
        context_path=context_path,
        use_dense=bge_config['use_dense'],
        use_sparse=bge_config['use_sparse'],
        use_colbert=bge_config['use_colbert'],
        batch_size=bge_config['batch_size'],
        max_length=bge_config['max_length'],
    )
    
    # Embeddings 생성 또는 로드
    logger.info("Building or loading BGE-M3 embeddings...")
    bge_retriever.get_embeddings()
    
    # Retrieval 수행
    topk = data_args.top_k_retrieval
    logger.info(f"Retrieving top-{topk} passages...")
    retrieved_df = bge_retriever.retrieve(
        datasets["validation"], 
        topk=topk,
        weights=bge_config['weights']
    )
    
    logger.info(f"Retrieved {len(retrieved_df)} (question, passage) pairs")

    # Retrieval metrics 계산
    retrieval_metrics = None
    if "original_context" in retrieved_df.columns:
        grouped = retrieved_df.groupby("id")
        total_questions = grouped.ngroups
        correct_questions = 0 

        def check_retrieval_success_for_group(group):
            original = group["original_context"].iloc[0].strip()
            
            for retrieved in group["context"]:
                retrieved = retrieved.strip()

                if original in retrieved or retrieved in original:
                    return True
            
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
        
        logger.info(f"Retrieval Accuracy: {accuracy:.4f} ({correct_questions}/{total_questions})")

        retrieved_df = retrieved_df.drop(columns=["original_context"])

    # Dataset Features 정의
    dataset_features = None
    is_predict_mode = training_args.do_predict
    is_eval_mode = training_args.do_eval

    if is_predict_mode:
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
                        "answer_start": Value("int64"),
                    },
                    length=-1,
                ),
            }
        )

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
    """MRC 수행 (inference_bm25.py와 동일)"""
    validation_dataset = datasets["validation"]
    val_column_names = validation_dataset.column_names

    has_question_col = "question" in val_column_names
    has_context_col = "context" in val_column_names
    has_answers_col = "answers" in val_column_names

    question_col = "question" if has_question_col else val_column_names[0]
    context_col = "context" if has_context_col else val_column_names[1]
    answer_col = "answers" if has_answers_col else val_column_names[2]

    is_padding_right = tokenizer.padding_side == "right"

    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    def prepare_validation_features(examples):
        first_seq = examples[question_col if is_padding_right else context_col]
        second_seq = examples[context_col if is_padding_right else question_col]
        truncation_mode = "only_second" if is_padding_right else "only_first"
        padding_mode = "max_length" if data_args.pad_to_max_length else False

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

        overflow_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        num_tokenized = len(tokenized_examples["input_ids"])

        for idx in range(num_tokenized):
            seq_ids = tokenized_examples.sequence_ids(idx)
            ctx_idx = 1 if is_padding_right else 0
            orig_sample_idx = overflow_mapping[idx]
            
            tokenized_examples["example_id"].append(examples["id"][orig_sample_idx])

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

    if "token_type_ids" in processed_dataset.column_names and model.config.model_type == "roberta":
        processed_dataset = processed_dataset.remove_columns("token_type_ids")

    pad_multiple = 8 if training_args.fp16 else None
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=pad_multiple
    )

    def post_processing_function(examples, features, predictions, training_args):
        processed_predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        
        formatted_predictions = []
        for prediction_id, prediction_text in processed_predictions.items():
            formatted_predictions.append(
                {"id": prediction_id, "prediction_text": prediction_text}
            )

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            reference_list = []
            for val_example in eval_examples:
                reference_list.append({
                    "id": val_example["id"], 
                    "answers": val_example[answer_col],
                })

            return EvalPrediction(
                predictions=formatted_predictions, 
                label_ids=reference_list
            )

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

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

    mrc_metrics = None

    if training_args.do_predict:
        prediction_results = qa_trainer.predict(
            test_dataset=processed_dataset, 
            test_examples=datasets["validation"]
        )
        print("No metric can be presented because there is no correct answer given. Job done!")

    if training_args.do_eval:
        eval_metrics = qa_trainer.evaluate()
        eval_metrics["eval_samples"] = len(processed_dataset)

        qa_trainer.log_metrics("test", eval_metrics)
        qa_trainer.save_metrics("test", eval_metrics)

        mrc_metrics = {
            "exact_match": eval_metrics.get("exact_match"),
            "f1": eval_metrics.get("f1"),
        }
        
        logger.info(f"MRC Metrics - EM: {mrc_metrics['exact_match']:.4f}, F1: {mrc_metrics['f1']:.4f}")

    return mrc_metrics


if __name__ == "__main__":
    main()