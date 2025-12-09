"""
Inference script for Hybrid KURE Retrieval + Cross-encoder Reranker + MRC Reader

파이프라인:
    Query → BM25 + KURE → Hybrid Score Fusion → Reranker → Top-K → Reader

Usage:
    python -m src.inference.inference_hybrid_kure \
        --output_dir outputs/hybrid_kure \
        --dataset_name data/test_dataset \
        --model_name_or_path models/train_dataset \
        --kure_model_path models/kure_finetuned/encoder \
        --do_predict \
        --eval_retrieval \
        --alpha 0.5 \
        --top_k_retrieval 1 \
        --use_reranker \
        --reranker_model upskyy/ko-reranker \
        --use_wandb \
        --wandb_project "retrieval" \
        --wandb_run_name "hybrid_kure_k1_rerank"
"""

import logging
import os
import sys
from typing import Callable, Dict, List, NoReturn, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import evaluate
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
)
from tqdm.auto import tqdm
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

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config.arguments import DataTrainingArguments as BaseDataTrainingArguments, ModelArguments
from src.retrieval.retrieval_hybrid_kure import HybridKURERetrieval
from src.training.trainer_qa import QuestionAnsweringTrainer
from src.utils.utils_qa import postprocess_qa_predictions

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments(BaseDataTrainingArguments):
    """Arguments for data training with Hybrid KURE parameters."""
    alpha: float = field(
        default=0.5,
        metadata={"help": "Weight for BM25 (0.0-1.0) in hybrid score fusion"}
    )
    use_reranker: bool = field(
        default=False,
        metadata={"help": "Enable Cross-encoder reranking"}
    )
    reranker_model: str = field(
        default="upskyy/ko-reranker",
        metadata={"help": "Reranker model name"}
    )
    kure_model_path: str = field(
        default="nlpai-lab/KURE-v1",
        metadata={"help": "KURE model path"}
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "Wandb entity (team) name."}
    )
    experiment_note: str = field(
        default="",
        metadata={"help": "Optional note for this experiment."}
    )


def init_wandb(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
):
    """Initialize wandb with config."""
    import wandb
    
    # Auto-generate run name if not provided
    run_name = data_args.wandb_run_name
    if run_name is None or run_name == "run":
        run_name = f"hybrid_kure_alpha{data_args.alpha}_k{data_args.top_k_retrieval}"
        if data_args.use_reranker:
            run_name += "_rerank"
    
    wandb.init(
        project=data_args.wandb_project,
        entity=data_args.wandb_entity,
        name=run_name,
        config={
            # Retrieval parameters
            "retrieval_method": "Hybrid_KURE",
            "top_k_retrieval": data_args.top_k_retrieval,
            "alpha": data_args.alpha,
            "use_reranker": data_args.use_reranker,
            "reranker_model": data_args.reranker_model if data_args.use_reranker else None,
            "kure_model_path": data_args.kure_model_path,
            
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
            "experiment_note": data_args.experiment_note,
        }
    )
    
    print(f"\n{'='*50}")
    print(f"Wandb initialized!")
    print(f"Project: {data_args.wandb_project}")
    print(f"Run name: {run_name}")
    print(f"{'='*50}\n")
    
    return wandb


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    print("=" * 60)
    print("Hybrid KURE Retrieval + Reranker + MRC Inference")
    print("=" * 60)
    print(f"Reader Model: {model_args.model_name_or_path}")
    print(f"KURE Model: {data_args.kure_model_path}")
    print(f"Dataset: {data_args.dataset_name}")
    print(f"Hybrid Alpha: {data_args.alpha} (BM25 weight)")
    print(f"Top-K Retrieval: {data_args.top_k_retrieval}")
    print(f"Use Reranker: {data_args.use_reranker}")
    if data_args.use_reranker:
        print(f"Reranker Model: {data_args.reranker_model}")
    print(f"Use Wandb: {data_args.use_wandb}")
    print("=" * 60)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    # Initialize wandb if enabled
    wandb = None
    if data_args.use_wandb:
        wandb = init_wandb(data_args, model_args, training_args)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    model_config_path = model_args.config_name if model_args.config_name else model_args.model_name_or_path
    model_config = AutoConfig.from_pretrained(model_config_path)
    
    tokenizer_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    
    is_tensorflow_model = ".ckpt" in model_args.model_name_or_path
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=is_tensorflow_model,
        config=model_config,
    )

    # Run Hybrid KURE retrieval and get metrics
    retrieval_metrics = None
    should_run_retrieval = data_args.eval_retrieval
    if should_run_retrieval:
        datasets, retrieval_metrics = run_hybrid_kure_retrieval(
            tokenizer=tokenizer,
            datasets=datasets,
            training_args=training_args,
            data_args=data_args,
        )
        
        # Log retrieval metrics to wandb
        if wandb is not None and retrieval_metrics:
            wandb.log({
                "retrieval/kure_accuracy": retrieval_metrics.get("kure_accuracy"),
                # Hybrid (Pre-rerank) Metrics -> Standard Names
                "retrieval/accuracy": retrieval_metrics.get("pre_rerank_accuracy"),
                "retrieval/mrr": retrieval_metrics.get("pre_rerank_mrr"),
                "retrieval/post_rerank_accuracy": retrieval_metrics.get("post_rerank_accuracy"),
                "retrieval/post_rerank_mrr": retrieval_metrics.get("post_rerank_mrr"),
                "retrieval/accuracy_improvement": retrieval_metrics.get("accuracy_improvement"),
                "retrieval/mrr_improvement": retrieval_metrics.get("mrr_improvement"),
                "retrieval/correct_count": retrieval_metrics.get("correct_count"),
            })
            print(f"[Wandb] Logged retrieval metrics: {retrieval_metrics}")

    # Run MRC
    should_run_mrc = training_args.do_eval or training_args.do_predict
    if should_run_mrc:
        mrc_metrics = run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)
        
        # Log MRC metrics to wandb (if available)
        if wandb is not None and mrc_metrics:
            wandb.log({
                "mrc/exact_match": mrc_metrics.get("exact_match"),
                "mrc/f1": mrc_metrics.get("f1"),
            })
            print(f"[Wandb] Logged MRC metrics: {mrc_metrics}")
    
    # Finish wandb run
    if wandb is not None:
        wandb.finish()
        print("[Wandb] Run finished successfully!")


def run_hybrid_kure_retrieval(
    tokenizer,
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "data",
    context_path: str = "wikipedia_documents.json",
) -> Tuple[DatasetDict, Dict]:
    """
    Hybrid KURE Retrieval을 수행하고 retrieved context를 Dataset에 추가합니다.
    """
    
    # Initialize Hybrid Retriever
    retriever = HybridKURERetrieval(
        tokenizer=tokenizer,
        data_path=data_path,
        context_path=context_path,
        kure_model_path=data_args.kure_model_path,
        use_reranker=data_args.use_reranker,
        reranker_model=data_args.reranker_model,
    )
    
    # Retrieve for test/validation set
    if "validation" in datasets:
        eval_dataset = datasets["validation"]
    elif "test" in datasets:
        eval_dataset = datasets["test"]
    else:
        raise ValueError("No validation or test set found in datasets")
    
    print(f"\nRunning Hybrid KURE Retrieval on {len(eval_dataset)} samples...")
    
    retrieved_df, metrics = retriever.retrieve(
        query_or_dataset=eval_dataset,
        topk=data_args.top_k_retrieval,
        alpha=data_args.alpha,
    )
    
    # Remove original_context column if exists
    if "original_context" in retrieved_df.columns:
        retrieved_df = retrieved_df.drop(columns=["original_context"])

    dataset_features = None
    is_predict_mode = training_args.do_predict
    is_eval_mode = training_args.do_eval
    
    if is_predict_mode:
        dataset_features = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    elif is_eval_mode:
        dataset_features = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    
    result_datasets = DatasetDict({"validation": Dataset.from_pandas(retrieved_df, features=dataset_features)})
    return result_datasets, metrics


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> Dict:
    """
    MRC Reader를 실행하여 답변을 추출합니다.
    """
    
    val_column_names = datasets["validation"].column_names

    has_question_col = "question" in val_column_names
    has_context_col = "context" in val_column_names
    has_answers_col = "answers" in val_column_names
    
    question_col = "question" if has_question_col else val_column_names[0]
    context_col = "context" if has_context_col else val_column_names[1]
    answer_col = "answers" if has_answers_col else val_column_names[2]

    is_padding_right = tokenizer.padding_side == "right"
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def prepare_validation_features(examples):
        first_seq = examples[question_col if is_padding_right else context_col]
        second_seq = examples[context_col if is_padding_right else question_col]
        truncation_mode = "only_second" if is_padding_right else "only_first"
        padding_mode = "max_length"
        
        tokenized_examples = tokenizer(
            first_seq,
            second_seq,
            truncation=truncation_mode,
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,  # RoBERTa doesn't use token_type_ids
            padding=padding_mode,
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

    validation_dataset = datasets["validation"]

    processed_dataset = validation_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=val_column_names,
        load_from_cache_file=False,
    )

    pad_multiple = 8 if training_args.fp16 else None
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=pad_multiple
    )

    metric = evaluate.load("squad")

    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        processed_predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        formatted_predictions = []
        for prediction_id, prediction_text in processed_predictions.items():
            formatted_predictions.append({"id": prediction_id, "prediction_text": prediction_text})

        is_predict = training_args.do_predict
        is_eval = training_args.do_eval
        
        if is_predict:
            return formatted_predictions
        elif is_eval:
            reference_list = []
            for val_example in datasets["validation"]:
                reference_list.append({"id": val_example["id"], "answers": val_example[answer_col]})

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=reference_list
            )

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    qa_trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=processed_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    should_predict = training_args.do_predict
    should_eval = training_args.do_eval
    
    mrc_metrics = None
    
    if should_predict:
        prediction_results = qa_trainer.predict(
            test_dataset=processed_dataset, test_examples=datasets["validation"]
        )

        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if should_eval:
        eval_metrics = qa_trainer.evaluate()
        eval_metrics["eval_samples"] = len(processed_dataset)
        
        # Extract EM and F1 for wandb logging
        mrc_metrics = {
            "exact_match": eval_metrics.get("exact_match"),
            "f1": eval_metrics.get("f1"),
        }

        qa_trainer.log_metrics("test", eval_metrics)
        qa_trainer.save_metrics("test", eval_metrics)
    
    return mrc_metrics


if __name__ == "__main__":
    main()
