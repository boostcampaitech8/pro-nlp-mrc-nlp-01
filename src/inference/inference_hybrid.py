import logging
import os
import sys
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from typing import Tuple, List, Callable, Dict
import numpy as np
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config.arguments import DataTrainingArguments, ModelArguments
from src.retrieval.retrieval_hybrid import HybridRetrieval
from src.utils.utils_qa import postprocess_qa_predictions, check_no_error

logger = logging.getLogger(__name__)

def run_hybrid_retrieval(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
    alpha: float = 0.5,
) -> Tuple[DatasetDict, Dict]:
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True,
    )

    retriever = HybridRetrieval(
        args=training_args,
        tokenizer=tokenizer,
        model_args=model_args,
        data_path=data_path,
        context_path=context_path,
    )
    
    retrieval_metrics = {}
    
    if training_args.do_eval:
        dataset = datasets["validation"]
        
        df, metrics = retriever.retrieve(dataset, topk=data_args.top_k_retrieval, alpha=alpha)
        retrieval_metrics = metrics
        
        if len(df) != len(dataset):
             logger.warning(f"Retrieved length {len(df)} != Dataset length {len(dataset)}")

        id_to_context = {row["id"]: row["context"] for _, row in df.iterrows()}
        
        def update_context(example):
            if example["id"] in id_to_context:
                example["context"] = id_to_context[example["id"]]
            return example
            
        datasets["validation"] = dataset.map(update_context)

    return datasets, retrieval_metrics

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 (0.0-1.0)")
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        outputs = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        model_args, data_args, training_args = outputs[:3]
        custom_args_namespace = outputs[3]
        alpha = custom_args_namespace.alpha

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info(f"Hybrid Retrieval Alpha: {alpha}")

    # WandB initialization
    if data_args.use_wandb:
        import wandb
        wandb.init(
            project=data_args.wandb_project,
            name=data_args.wandb_run_name,
            config={
                "alpha": alpha,
                "top_k": data_args.top_k_retrieval,
                "model": model_args.model_name_or_path,
                "retriever": model_args.retriever_name_or_path,
            }
        )
        print(f"[Wandb] Run {data_args.wandb_run_name} started on project {data_args.wandb_project}")

    # Load Dataset
    datasets = load_from_disk(data_args.dataset_name)
    
    retrieval_metrics = {}
    if training_args.do_eval:
        datasets, retrieval_metrics = run_hybrid_retrieval(
            datasets, 
            training_args, 
            data_args, 
            model_args,
            alpha=alpha
        )
        
        # Log to WandB
        if data_args.use_wandb and retrieval_metrics:
            wandb.log({
                "retrieval/accuracy": retrieval_metrics.get("accuracy", 0),
                "retrieval/mrr": retrieval_metrics.get("mrr", 0),
                "retrieval/correct_count": retrieval_metrics.get("correct_count", 0),
                "retrieval/total_count": retrieval_metrics.get("total_count", 0),
                "retrieval/top_k": retrieval_metrics.get("top_k", 100),
                "retrieval/alpha": alpha,
            })
            print(f"[Wandb] Logged retrieval metrics: {retrieval_metrics}")

    # Finish WandB run
    if data_args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
