import logging
import os
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

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

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config.arguments import DataTrainingArguments, ModelArguments
from src.retrieval.retrieval_hybrid import HybridRetrieval
from src.training.trainer_qa import QuestionAnsweringTrainer
from src.utils.utils_qa import check_no_error, postprocess_qa_predictions

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    # Add custom arguments by parsing remaining strings if needed, 
    # but HfArgumentParser usually handles dataclasses. 
    # We'll stick to standard args and add alpha manually or via custom arg parsing if we define a dataclass.
    # For simplicity, we'll parse sys.argv for alpha manually or just expect it in command line if we added it to DataTrainingArguments (we didn't).
    # So let's use the same approach as inference_hybrid.py
    
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

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")
    print(f"Hybrid Alpha: {alpha}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

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

    if data_args.eval_retrieval:
        # Run Hybrid Retrieval
        datasets = run_hybrid_retrieval_and_update(
            tokenizer,
            datasets,
            training_args,
            data_args,
            model_args,
            alpha=alpha
        )

    # Run MRC (Reader)
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

def run_hybrid_retrieval_and_update(
    tokenizer,
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    alpha: float = 0.5,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    
    retriever = HybridRetrieval(
        args=training_args,
        tokenizer=tokenizer,
        model_args=model_args,
        data_path=data_path,
        context_path=context_path,
    )

    # We typically run retrieval on the 'validation' set (or 'test' if generic)
    # The existing code typically assumes 'validation' for eval/predict.
    target_split = "validation" 
    if target_split not in datasets:
        # If we are in predict only mode and have a test dataset
        if "test" in datasets:
            target_split = "test"
    
    print(f"Retrieving for split: {target_split}")
    dataset = datasets[target_split]
    
    df, metrics = retriever.retrieve(dataset, topk=data_args.top_k_retrieval, alpha=alpha)

    # The dataframe 'df' contains 'context', 'id', 'question'
    # We need to convert this back to a dataset and put it into datasets[target_split]
    
    # We align via IDs to ensure correct mapping if needed, but retriever returns aligned df usually
    # However, creating a new dataset from df is safer.
    
    if training_args.do_predict:
        features = Features({
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        })
    else:
        # Keep answers for evaluation
        features = Features({
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
        })
        # If 'answers' column is missing in df (it usually is unless hybrid retriever adds it back),
        # we need to recover it from the original dataset.
        
    # Helper to merge answers back if they exist in original
    id_to_answers = {}
    if "answers" in dataset.features:
        for row in dataset:
            id_to_answers[row["id"]] = row["answers"]
            
    def add_answers(row):
        if row["id"] in id_to_answers:
            row["answers"] = id_to_answers[row["id"]]
        return row

    # Create dataset from DF
    # Note: HybridRetrieval.retrieve returns DF with 'context', 'id', 'question', 'original_context'
    # We select what we need.
    
    keep_cols = ["id", "question", "context"]
    new_data = Dataset.from_pandas(df[keep_cols])
    
    if "answers" in dataset.features:
        # We need to add answers back because retrieve might have dropped them or not included them in DF
        # Actually HybridRetrieval implementation keeps original data structure usually? 
        # Looking at retrieval_hybrid.py: returns pd.DataFrame(total). 'total' has 'question', 'id', 'context'.
        # It does NOT verify 'answers' key in 'tmp'.
        # So we MUST add answers back for evaluation to work.
        new_data = new_data.map(add_answers)

    datasets[target_split] = new_data
    return datasets

def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:
    
    # Reuse the same run_mrc logic as inference.py
    # We can just copy-paste it or import it if it was standalone, 
    # but inference.py's run_mrc is defined inside.
    # We'll copy the logic here for stability.
    
    # Determine split
    if "validation" in datasets:
        eval_dataset = datasets["validation"]
    elif "test" in datasets:
        eval_dataset = datasets["test"]
    else:
        raise ValueError("No validation or test split found.")

    column_names = eval_dataset.column_names
    
    question_col = "question" if "question" in column_names else column_names[0]
    context_col = "context" if "context" in column_names else column_names[1]
    answer_col = "answers" if "answers" in column_names else column_names[2]

    is_padding_right = tokenizer.padding_side == "right"
    
    # Validation preprocessing
    max_seq_length = data_args.max_seq_length
    
    def prepare_validation_features(examples):
        first_seq = examples[question_col if is_padding_right else context_col]
        second_seq = examples[context_col if is_padding_right else question_col]
        truncation_mode = "only_second" if is_padding_right else "only_first"
        
        tokenized_examples = tokenizer(
            first_seq,
            second_seq,
            truncation=truncation_mode,
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        overflow_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # Token type ids fix for RoBERTa if needed
        if "token_type_ids" in tokenized_examples:
            tokenized_examples.pop("token_type_ids")
            
        tokenized_examples["example_id"] = []

        for idx in range(len(tokenized_examples["input_ids"])):
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

    processed_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    def post_processing_function(examples, features, predictions, training_args):
        # We need this function to be pickle-able or available
        # importing postprocess_qa_predictions inside or using the one from utils
        return postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=processed_dataset,
        eval_examples=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    if training_args.do_predict:
        predictions = trainer.predict(processed_dataset, eval_dataset)
        # Predictions are saved by post_process_function usually
    
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
