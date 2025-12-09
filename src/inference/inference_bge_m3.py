"""
최적화된 BGE-M3 + Re-ranker Inference

실행 예시:

# Dense + Sparse + Re-ranker (추천, 메모리 효율)
python -m src.inference.inference_bge_m3_optimized \
  --output_dir outputs/eval_bge_m3_rerank_k10/ \
  --dataset_name data/train_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_eval \
  --eval_retrieval \
  --top_k_retrieval 10 \
  --bge_use_reranker True \
  --bge_rerank_top_k 50 \
  --bge_batch_size 8 \
  --use_wandb True

# ColBERT 포함 (메모리 충분할 때)
python -m src.inference.inference_bge_m3_optimized \
  --output_dir outputs/eval_bge_m3_full_k10/ \
  --dataset_name data/train_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_eval \
  --eval_retrieval \
  --top_k_retrieval 10 \
  --bge_use_dense True \
  --bge_use_sparse True \
  --bge_use_colbert True \
  --bge_use_reranker True \
  --bge_rerank_top_k 50 \
  --bge_dense_weight 0.4 \
  --bge_sparse_weight 0.4 \
  --bge_colbert_weight 0.2

# Hard Negative Sampling을 위한 데이터 생성
python -m src.inference.inference_bge_m3_optimized \
  --output_dir outputs/hard_negatives/ \
  --dataset_name data/train_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_eval \
  --eval_retrieval \
  --generate_hard_negatives True \
  --hard_negative_count 5 \
  --top_k_retrieval 10
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
from .retrieval_bge_m3_optimized import BGEM3RetrievalOptimized
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
    
    methods = []
    if bge_config['use_dense']:
        methods.append('dense')
    if bge_config['use_sparse']:
        methods.append('sparse')
    if bge_config['use_colbert']:
        methods.append('colbert')
    if bge_config['use_reranker']:
        methods.append('rerank')
    method_str = '+'.join(methods)
    
    run_name = data_args.wandb_run_name
    if run_name is None:
        run_name = f"bge_m3_opt_{method_str}_k{data_args.top_k_retrieval}"
    
    wandb.init(
        project=data_args.wandb_project,
        name=run_name,
        config={
            "retrieval_method": "BGE-M3-Optimized",
            "bge_methods": method_str,
            "use_dense": bge_config['use_dense'],
            "use_sparse": bge_config['use_sparse'],
            "use_colbert": bge_config['use_colbert'],
            "use_reranker": bge_config['use_reranker'],
            "rerank_top_k": bge_config.get('rerank_top_k', 100),
            "dense_weight": bge_config['weights']['dense'],
            "sparse_weight": bge_config['weights']['sparse'],
            "colbert_weight": bge_config['weights']['colbert'],
            "top_k_retrieval": data_args.top_k_retrieval,
            "batch_size": bge_config['batch_size'],
            "max_length": bge_config['max_length'],
            "model_name_or_path": model_args.model_name_or_path,
            "dataset_name": data_args.dataset_name,
        }
    )
    
    print(f"\n{'='*50}")
    print(f"Wandb initialized!")
    print(f"Project: {data_args.wandb_project}")
    print(f"Run name: {run_name}")
    print(f"Methods: {method_str}")
    if bge_config['use_reranker']:
        print(f"Re-ranking: {bge_config.get('rerank_top_k', 100)} → {data_args.top_k_retrieval}")
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
    set_seed(training_args.seed)

    # BGE-M3 설정 (최적화된 파라미터)
    bge_config = {
        'use_dense': getattr(data_args, 'bge_use_dense', True),
        'use_sparse': getattr(data_args, 'bge_use_sparse', True),
        'use_colbert': getattr(data_args, 'bge_use_colbert', False),
        'use_reranker': getattr(data_args, 'bge_use_reranker', True),
        'batch_size': getattr(data_args, 'bge_batch_size', 8),  # 12 -> 8
        'max_length': getattr(data_args, 'bge_max_length', 512),
        'rerank_top_k': getattr(data_args, 'bge_rerank_top_k', 100),
        'max_memory_gb': getattr(data_args, 'bge_max_memory_gb', 28.0),
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
    model_config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=model_config,
    )

    # BGE-M3 Retrieval 수행
    should_run_retrieval = data_args.eval_retrieval
    retrieval_metrics = None
    original_eval_examples = datasets["validation"]

    # Hard Negative 생성 모드
    generate_hard_negatives = getattr(data_args, 'generate_hard_negatives', False)
    
    if should_run_retrieval:
        if generate_hard_negatives:
            datasets = generate_hard_negative_dataset(
                datasets,
                training_args,
                data_args,
                bge_config=bge_config,
            )
        else:
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
    if should_run_mrc and not generate_hard_negatives:
        mrc_metrics = run_mrc(
            data_args, 
            training_args, 
            model_args, 
            datasets,
            tokenizer, 
            model,
            eval_examples=original_eval_examples,
        )

        if wandb is not None and mrc_metrics is not None:
            wandb.log({
                "mrc/exact_match": mrc_metrics.get("exact_match"),
                "mrc/f1": mrc_metrics.get("f1"),
            })
            print(f"[Wandb] Logged MRC metrics: {mrc_metrics}")

    # Wandb 종료
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
    """최적화된 BGE-M3 retrieval 수행"""
    logger.info("*** Running Optimized BGE-M3 Retrieval ***")
    
    if bge_config is None:
        bge_config = {
            'use_dense': True,
            'use_sparse': True,
            'use_colbert': False,
            'use_reranker': True,
            'batch_size': 8,
            'max_length': 512,
            'rerank_top_k': 100,
            'max_memory_gb': 28.0,
            'weights': {'dense': 0.5, 'sparse': 0.5, 'colbert': 0.0}
        }
    
    # BGE-M3 Retrieval 초기화 (최적화 버전)
    bge_retriever = BGEM3RetrievalOptimized(
        data_path=data_path,
        context_path=context_path,
        use_dense=bge_config['use_dense'],
        use_sparse=bge_config['use_sparse'],
        use_colbert=bge_config['use_colbert'],
        use_reranker=bge_config['use_reranker'],
        batch_size=bge_config['batch_size'],
        max_length=bge_config['max_length'],
        max_memory_gb=bge_config.get('max_memory_gb', 28.0),
    )
    
    # Embeddings 생성 또는 로드
    logger.info("Building or loading optimized BGE-M3 embeddings...")
    bge_retriever.get_embeddings()
    
    # Retrieval 수행 (Re-ranking 포함)
    topk = data_args.top_k_retrieval
    rerank_top_k = bge_config.get('rerank_top_k', 100)
    
    logger.info(f"Retrieving top-{topk} passages...")
    if bge_config['use_reranker']:
        logger.info(f"Re-ranking enabled: {rerank_top_k} candidates → {topk} final results")
    
    retrieved_df = bge_retriever.retrieve(
        datasets["validation"], 
        topk=topk,
        weights=bge_config['weights'],
        use_rerank=bge_config['use_reranker'],
        rerank_top_k=rerank_top_k,
    )
    
    logger.info(f"Retrieved {len(retrieved_df)} (question, passage) pairs")

    # Retrieval metrics 계산
    retrieval_metrics = None
    if "original_context" in retrieved_df.columns:
        grouped = retrieved_df.groupby("id")
        total_questions = grouped.ngroups
        correct_questions = 0

        def check_retrieval_success(group):
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
            if check_retrieval_success(group):
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
    if training_args.do_predict:
        dataset_features = Features({
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
            "context": Value(dtype="string", id=None),
            "retrieval_rank": Value("int32"),
            "retrieval_score": Value("float32"),
        })
    elif training_args.do_eval:
        dataset_features = Features({
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
        })

    result_datasets = DatasetDict({
        "validation": Dataset.from_pandas(retrieved_df, features=dataset_features)
    })
    
    return result_datasets, retrieval_metrics


def generate_hard_negative_dataset(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "data",
    context_path: str = "wikipedia_documents.json",
    bge_config: dict = None,
) -> DatasetDict:
    """Hard Negative Sampling을 위한 데이터셋 생성"""
    logger.info("*** Generating Hard Negative Dataset ***")
    
    bge_retriever = BGEM3RetrievalOptimized(
        data_path=data_path,
        context_path=context_path,
        use_dense=bge_config['use_dense'],
        use_sparse=bge_config['use_sparse'],
        use_colbert=bge_config['use_colbert'],
        use_reranker=False,  # Hard negative에서는 reranker 사용 안 함
        batch_size=bge_config['batch_size'],
        max_length=bge_config['max_length'],
    )
    
    bge_retriever.get_embeddings()
    
    hard_negative_count = getattr(data_args, 'hard_negative_count', 5)
    
    rows = []
    for example in tqdm(datasets["validation"], desc="Generating hard negatives"):
        query = example["question"]
        
        # Positive document ID 찾기 (실제 정답이 있는 문서)
        positive_context = example.get("context", "")
        positive_id = None
        
        # Context가 있으면 해당하는 ID 찾기
        for i, ctx in enumerate(bge_retriever.contexts):
            if positive_context in ctx or ctx in positive_context:
                positive_id = i
                break
        
        if positive_id is None:
            continue
        
        # Hard negatives 생성
        hard_negs = bge_retriever.generate_hard_negatives(
            query=query,
            positive_doc_id=positive_id,
            k=hard_negative_count,
            weights=bge_config['weights']
        )
        
        # 데이터 저장 (positive + hard negatives)
        row = {
            "id": example["id"],
            "question": query,
            "positive_context": bge_retriever.contexts[positive_id],
            "negative_contexts": [bge_retriever.contexts[i] for i in hard_negs],
            "answers": example.get("answers", None),
        }
        rows.append(row)
    
    import json
    output_path = os.path.join(training_args.output_dir, "hard_negatives.json")
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Hard negatives saved to {output_path}")
    logger.info(f"Total examples: {len(rows)}, Hard negatives per example: {hard_negative_count}")
    
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    eval_examples,
) -> Dict:
    """MRC 수행 (기존 코드 유지)"""
    validation_dataset = datasets["validation"]
    val_column_names = validation_dataset.column_names

    question_col = "question" if "question" in val_column_names else val_column_names[0]
    context_col = "context" if "context" in val_column_names else val_column_names[1]
    answer_col = "answers" if "answers" in val_column_names else val_column_names[2]

    is_padding_right = tokenizer.padding_side == "right"
    _, max_seq_length = check_no_error(data_args, training_args, datasets, tokenizer)

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

        for idx in range(len(tokenized_examples["input_ids"])):
            seq_ids = tokenized_examples.sequence_ids(idx)
            ctx_idx = 1 if is_padding_right else 0
            orig_sample_idx = overflow_mapping[idx]
            
            tokenized_examples["example_id"].append(examples["id"][orig_sample_idx])

            current_offset_map = tokenized_examples["offset_mapping"][idx]
            new_offset_map = [
                offset_val if seq_ids[pos] == ctx_idx else None
                for pos, offset_val in enumerate(current_offset_map)
            ]
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

    data_collator = DataCollatorWithPadding(
        tokenizer, 
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

    def post_processing_function(examples, features, predictions, training_args):
        processed_predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        
        formatted_predictions = [
            {"id": pred_id, "prediction_text": pred_text}
            for pred_id, pred_text in processed_predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_col]}
                for ex in eval_examples
            ]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

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
        qa_trainer.predict(test_dataset=processed_dataset, test_examples=datasets["validation"])
        print("Prediction completed!")

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