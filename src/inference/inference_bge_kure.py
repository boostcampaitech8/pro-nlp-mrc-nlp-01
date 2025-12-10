"""
KURE-v1 Dense + BGE-M3 Sparse + BGE-Reranker Hybrid Retrieval + MRC Inference
기존 inference_bge_m3_fixed.py 와 100% 동일한 플로우를 유지한 Hybrid 버전

python -m src.inference.inference_bge_kure \
  --output_dir outputs/hn_bge_kure_k100_rk_5_eval_best_rd/ \
  --overwrite_output_dir True \
  --dataset_name data/train_dataset/ \
  --model_name_or_path models/best-reader/ \
  --dense_embedding_path data/kure_dense_hn2.npy \
  --do_eval \
  --eval_retrieval \
  --top_k_retrieval 100 \
  --bge_use_reranker True \
  --bge_rerank_top_k 5 \
  --use_wandb True\
  --wandb_project "retrieval"\
  --wandb_run_name "bge-kure(hn1)-bge-tk100-rk5" 
# 

python -m src.inference.inference_bge_kure \
  --output_dir outputs/hn_bge_kure_k100_rk_5_pred_3/ \
  --overwrite_output_dir True \
  --dataset_name data/test_dataset/ \
  --model_name_or_path models/best-reader/ \
  --dense_embedding_path data/kure_dense_hn2.npy \
  --sparse_embedding_path data/bge_sparse.pkl \
  --do_predict \
  --eval_retrieval \
  --top_k_retrieval 100 \
  --bge_use_reranker True \
  --bge_rerank_top_k 5 \
  --use_wandb False \
  --fp16 True


python -m scripts.hn_mining_kure \
    --dataset_name data/train_dataset \
    --output_path data/train_with_hard_negatives.json \
    --kure_model_path nlpai-lab/KURE-v1 \
    --num_hard_negatives 5 \
    --num_candidates 100 \
    --use_bm25 True \
    --use_current_model True

python -m src.training.train_hn_kure \
    --train_data data/train_with_hard_negatives.json \
    --output_dir models/kure_finetuned_hard_neg \
    --model_name nlpai-lab/KURE-v1 \
    --batch_size 4 \
    --max_length 256 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --temperature 0.05
"""

import logging
import os
import sys
from typing import Dict, Tuple

import evaluate
import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
)
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

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config.arguments import DataTrainingArguments, ModelArguments
from src.retrieval.retrieval_bge_kure import BGEM3KUREHybridRetrieval
from src.training.trainer_qa import QuestionAnsweringTrainer
from src.utils.utils_qa import postprocess_qa_predictions, check_no_error

logger = logging.getLogger(__name__)


def init_wandb(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    config: dict,
):
    """Wandb 초기화"""
    import wandb
    
    methods = []
    if config['use_dense']:
        methods.append('kure_dense')
    if config['use_sparse']:
        methods.append('bge_sparse')
    if config['use_reranker']:
        methods.append('rerank')
    method_str = '+'.join(methods)
    
    run_name = data_args.wandb_run_name
    if run_name is None or run_name == "run":
        run_name = f"kure_bge_{method_str}_k{data_args.top_k_retrieval}"
    
    wandb.init(
        project=data_args.wandb_project,
        name=run_name,
        config={
            "retrieval_method": "KURE+BGE Hybrid",
            "methods": method_str,
            "use_dense": config['use_dense'],
            "use_sparse": config['use_sparse'],
            "use_reranker": config['use_reranker'],
            "dense_weight": config['weights']['dense'],
            "sparse_weight": config['weights']['sparse'],
            "top_k_retrieval": data_args.top_k_retrieval,
            "batch_size": config['batch_size'],
            "max_length": config['max_length'],
            "model_name_or_path": model_args.model_name_or_path,
            "dataset_name": data_args.dataset_name,
        }
    )
    
    print(f"\n{'='*50}")
    print(f"Wandb initialized!")
    print(f"Project: {data_args.wandb_project}")
    print(f"Run name: {run_name}")
    print(f"Methods: {method_str}")
    if config['use_reranker']:
        print(f"Re-ranking: {config['rerank_top_k']} → {data_args.top_k_retrieval}")
    print(f"{'='*50}\n")
    
    return wandb


def main():
    """메인 실행 함수"""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    print("="*60)
    print("KURE Dense + BGE-M3 Sparse Hybrid Retrieval + Reranker + MRC Inference")
    print("="*60)
    print(f"Reader Model: {model_args.model_name_or_path}")
    print(f"Dataset: {data_args.dataset_name}")
    print(f"Top-K Retrieval: {data_args.top_k_retrieval}")
    print(f"Use Wandb: {data_args.use_wandb}")
    print("="*60)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    # Hybrid Retrieval 설정
    kure_config = {
        'use_dense': True,
        'use_sparse': True,
        'use_reranker': getattr(data_args, "bge_use_reranker", True),
        'batch_size': getattr(data_args, "bge_batch_size", 8),
        'max_length': getattr(data_args, "bge_max_length", 512),
        'rerank_top_k': getattr(data_args, "bge_rerank_top_k", 100),
        'weights': {
            'dense': getattr(data_args, 'bge_dense_weight', 0.5),
            'sparse': getattr(data_args, 'bge_sparse_weight', 0.5),
        }
    }

    # Wandb 초기화
    wandb = None
    if data_args.use_wandb:
        wandb = init_wandb(data_args, model_args, training_args, kure_config)

    # 데이터셋 로드
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # Reader 모델 로드
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

    original_eval_examples = datasets["validation"]

    # Retrieval 실행
    retrieval_metrics = None
    if data_args.eval_retrieval:
        datasets, retrieval_metrics = run_kure_bge_retrieval(
            datasets=datasets,
            training_args=training_args,
            data_args=data_args,
            config=kure_config,
        )

        # Wandb 로깅
        if wandb is not None and retrieval_metrics:
            wandb.log({
                "retrieval/accuracy": retrieval_metrics.get("retrieval_accuracy"),
                "retrieval/mrr": retrieval_metrics.get("mrr"),
            })

    # MRC 실행
    if training_args.do_eval or training_args.do_predict:
        mrc_metrics = run_mrc(
            data_args=data_args,
            training_args=training_args,
            model_args=model_args,
            datasets=datasets,
            tokenizer=tokenizer,
            model=model,
            eval_examples=original_eval_examples,
        )

        if wandb is not None and mrc_metrics:
            wandb.log({
                "mrc/exact_match": mrc_metrics["exact_match"],
                "mrc/f1": mrc_metrics["f1"],
            })

    if wandb is not None:
        import wandb as _wandb
        _wandb.finish()
        print("[Wandb] Run finished!")


def run_kure_bge_retrieval(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path="data",
    context_path="wikipedia_documents.json",
    config=None,
) -> Tuple[DatasetDict, Dict]:

    logger.info("*** Running Hybrid KURE+BGE Retrieval ***")

    retriever = BGEM3KUREHybridRetrieval(
        data_path=data_path,
        context_path=context_path,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        use_dense=config["use_dense"],
        use_sparse=config["use_sparse"],
        use_reranker=config["use_reranker"],
        dense_embedding_path=data_args.dense_embedding_path,
        sparse_embedding_path=data_args.sparse_embedding_path,

    )

    logger.info("Building or loading embeddings...")
    retriever.get_embeddings()

    topk = data_args.top_k_retrieval
    rerank_top_k = config["rerank_top_k"]

    logger.info(f"Retrieving top-{topk} passages...")
    retrieved_df, metrics = retriever.retrieve(
        datasets["validation"],
        topk=topk,
        weights=config["weights"],
        use_rerank=config["use_reranker"],
        rerank_top_k=rerank_top_k,
    )

    logger.info(f"Retrieved {len(retrieved_df)} rows.")

    if "original_context" in retrieved_df.columns:
        retrieved_df = retrieved_df.drop(columns=["original_context"])

    if training_args.do_predict:
        drop_cols = ["answers", "original_context"]
        for col in drop_cols:
            if col in retrieved_df.columns:
                retrieved_df = retrieved_df.drop(columns=[col])

    # Dataset Features
    if training_args.do_predict:
        features = Features({
            "id": Value("string"),
            "question": Value("string"),
            "context": Value("string"),
            "retrieval_rank": Value("int32"),
            "retrieval_score": Value("float32"),
        })
    else:
        features = Features({
            "id": Value("string"),
            "question": Value("string"),
            "context": Value("string"),
            "retrieval_rank": Value("int32"),
            "retrieval_score": Value("float32"),
            "answers": Sequence(
                feature={"text": Value("string"), "answer_start": Value("int64")},
                length=-1,
            ),
        })

    result_dataset = DatasetDict({
        "validation": Dataset.from_pandas(retrieved_df, features=features)
    })

    return result_dataset, metrics


"""
⚠ run_mrc 는 기존 inference_bge_m3_fixed.py 와 완전히 동일하므로 아래에 그대로 복붙
"""
def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    eval_examples,
) -> Dict:
    """MRC 수행"""
    
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
