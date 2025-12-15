import logging
import os
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

import evaluate
import numpy as np
import torch
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

from dataclasses import dataclass, field

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config.arguments import DataTrainingArguments, ModelArguments
from src.retrieval.retrieval_dpr import DenseRetrieval
from src.training.trainer_qa import QuestionAnsweringTrainer
from src.utils.utils_qa import check_no_error, postprocess_qa_predictions

logger = logging.getLogger(__name__)

@dataclass
class WandbArguments:
    """Arguments for wandb logging."""
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to use wandb for logging."}
    )
    wandb_project: str = field(
        default="retrieval",
        metadata={"help": "Wandb project name."}
    )
    wandb_run_name: str = field(
        default=None,
        metadata={"help": "Wandb run name. If not set, will be auto-generated."}
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "Wandb entity (team) name."}
    )
    
def init_wandb(
    wandb_args: WandbArguments,
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
):
    """Initialize wandb with config."""
    import wandb
    
    # Auto-generate run name if not provided
    run_name = wandb_args.wandb_run_name
    if run_name is None:
        run_name = f"dpr_k{data_args.top_k_retrieval}"
    
    wandb.init(
        project=wandb_args.wandb_project,
        entity=wandb_args.wandb_entity,
        name=run_name,
        config={
            "retrieval_method": "DPR",
            "top_k_retrieval": data_args.top_k_retrieval,
            "model_name_or_path": model_args.model_name_or_path,
            "dataset_name": data_args.dataset_name,
            "max_seq_length": data_args.max_seq_length,
            "doc_stride": data_args.doc_stride,
            "max_answer_length": data_args.max_answer_length,
            "seed": training_args.seed,
        }
    )
    print(f"[Wandb] Run {run_name} started on project {wandb_args.wandb_project}")
    return wandb

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, WandbArguments)
    )
    model_args, data_args, training_args, wandb_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    print(f"Reader model is from {model_args.model_name_or_path}")
    print(f"Data is from {data_args.dataset_name}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(training_args.seed)
    
    # Initialize wandb
    wandb = None
    if wandb_args.use_wandb:
        wandb = init_wandb(wandb_args, model_args, data_args, training_args)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)
    
    # 0. Load Retriever Model Path
    # Using model_args isn't quite right for Reader vs Retriever separation,
    # but we'll stick to 'outputs/dpr_test' convention or check for args.
    # We will pass it to run_dense_retrieval.

    # 1. Load Reader Model (BERT/RoBERTa)
    model_config_path = model_args.config_name if model_args.config_name else model_args.model_name_or_path
    model_config = AutoConfig.from_pretrained(model_config_path)
    
    tokenizer_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
    )

    # 2. Run Retrieval (DPR)
    if data_args.eval_retrieval:
        datasets, ret_metrics = run_dense_retrieval(
            datasets,
            training_args,
            data_args,
            model_args,
        )
        if wandb and ret_metrics:
            # Filter out None values to avoid WandB charting issues
            wandb_metrics = {}
            metric_mapping = {
                "retrieval/accuracy": "retrieval_accuracy",
                "retrieval/mrr": "mrr",
                "retrieval/correct_count": "correct_count",
                "retrieval/total_count": "total_count",
                "retrieval/top_k": "top_k",
            }
            for wandb_key, metric_key in metric_mapping.items():
                value = ret_metrics.get(metric_key)
                if value is not None:
                    wandb_metrics[wandb_key] = value
            
            if wandb_metrics:
                wandb.log(wandb_metrics)
                print(f"[Wandb] Logged retrieval metrics: {wandb_metrics}")

    # 3. Run Reader (MRC)
    if training_args.do_eval or training_args.do_predict:
        mrc_metrics = run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)
        if wandb and mrc_metrics:
            wandb.log({
                "mrc/exact_match": mrc_metrics.get("exact_match"),
                "mrc/f1": mrc_metrics.get("f1"),
            })
            print(f"[Wandb] Logged MRC metrics: {mrc_metrics}")
            
    if wandb:
        wandb.finish()

def run_dense_retrieval(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    
    # We assume 'retriever_model_path' is passed via some argument or hardcoded for now
    # Since ModelArguments.model_name_or_path is for Reader, we need another arg for Retriever
    # For now, let's assume the user passes the retrieval model path via a custom way or we default to 'outputs/dpr_test'
    # OR we can misuse tokenizer_name if needed, but better to use a specific path if possible.
    
    # IMPORTANT: The user must provide the path to the trained DPR encoders.
    # Currently, we don't have a separate arg in ModelArguments for 'retriever_path'.
    # We will hardcode it to 'outputs/dpr_test' OR check if it exists in env args.
    # A cleaner way: Use 'outputs/dpr_test' as default if not specified.
    
    
    dpr_model_path = model_args.retriever_name_or_path
    if not os.path.isdir(dpr_model_path):
        print(f"Warning: DPR model not found at {dpr_model_path}. Using standard BERT might fail for DPR retrieval.")
    
    print(f"Using DPR Retriever from {dpr_model_path}")

    # Initialize Retriever
    retriever = DenseRetrieval(
        args=data_args,
        dataset=datasets["validation"], # Dummy dataset for init
        model_name_or_path=dpr_model_path,
        data_path=data_path,
        context_path=context_path,
    )
    
    # Build/Load Index
    retriever.get_dense_embedding()
    retriever.build_faiss()

    # Perform Retrieval
    # Target dataset: validation or test
    target_split = "validation" # Default to validation
    if training_args.do_predict:
        # If we have a test dataset, handle it here. 
        # But usually 'dataset_name' points to train_dataset which has train/val.
        # If running on 'test_dataset', we need to load it. 
        # For now, let's assume we are evaluating on Validation split as per existing code structure.
        pass

    # Retrieve (Top-K)
    retrieved_df = retriever.retrieve(
        datasets[target_split], topk=data_args.top_k_retrieval
    )
    # Get standardized metrics (Accuracy, MRR, etc.)
    # Note: retrieval_dpr.py now calculates these internally during retrieve()
    metrics = retriever.get_retrieval_metrics()

    # Remove original_context column if exists (used only for accuracy calculation or debugging)
    if "original_context" in retrieved_df.columns:
        retrieved_df = retrieved_df.drop(columns=["original_context"])

    # Convert to Dataset with Features
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    else:
        f = Features(
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

    result_datasets = DatasetDict({target_split: Dataset.from_pandas(retrieved_df, features=f)})
    return result_datasets, metrics

def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # Determine split
    if training_args.do_eval:
        dataset_name = "validation"
    elif training_args.do_predict:
        dataset_name = "test" # Logic needs adjustment if dataset dict keys differ
        if "test" not in datasets:
            dataset_name = "validation"
    else:
        dataset_name = "validation"

    column_names = datasets[dataset_name].column_names

    question_col = "question" if "question" in column_names else column_names[0]
    context_col = "context" if "context" in column_names else column_names[1]
    answer_col = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = data_args.max_seq_length

    def prepare_validation_features(examples):
        tokens = tokenizer(
            examples[question_col if pad_on_right else context_col],
            examples[context_col if pad_on_right else question_col],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokens.pop("overflow_to_sample_mapping")
        tokens["example_id"] = []

        for i in range(len(tokens["input_ids"])):
            sequence_ids = tokens.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokens["example_id"].append(examples["id"][sample_index])

            tokens["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokens["offset_mapping"][i])
            ]

        return tokens

    eval_dataset = datasets[dataset_name]
    
    # Process dataset
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Post-processing
    def post_processing_function(examples, features, predictions, stage="eval"):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Format for metric
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        
        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
             references = [
                {"id": ex["id"], "answers": ex[answer_col]} for ex in datasets[dataset_name]
            ]
             return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets[dataset_name],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")
    
    mrc_metrics = {}

    if training_args.do_predict:
        predictions = trainer.predict(eval_dataset, datasets[dataset_name])
    elif training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        mrc_metrics = {
            "exact_match": metrics.get("exact_match", 0),
            "f1": metrics.get("f1", 0),
        }
        
    return mrc_metrics

if __name__ == "__main__":
    main()
