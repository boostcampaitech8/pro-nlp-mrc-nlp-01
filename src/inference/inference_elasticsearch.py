"""
elasticsearch 켜져 있는지 확인 
> curl http://localhost:9200

python -m src.inference.inference_elasticsearch \
  --output_dir outputs/eval_es_k100/ \
  --dataset_name data/train_dataset/ \
  --model_name_or_path baseline/models/train_dataset/ \
  --do_eval \
  --eval_retrieval \
  --top_k_retrieval 100 \
  --use_wandb True \
  --wandb_project "retrieval" \
  --wandb_run_name "es_k100_eval"


python -m src.inference.inference_elasticsearch \
  --output_dir outputs/pred_es_k100/ \
  --dataset_name data/test_dataset/ \
  --model_name_or_path baseline/models/train_dataset/ \
  --do_predict \
  --eval_retrieval \
  --top_k_retrieval 100 \
  --use_wandb True \
  --wandb_project "retrieval" \
  --wandb_run_name "es_k100_submit"


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
# ES
from ..retrieval.retrieval_elasticsearch import ElasticSearchRetrieval
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

# wandb
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)



def init_wandb(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
):
    """Initialize wandb with config."""
    import wandb
    
    # Auto-generate run name if not provided
    run_name = data_args.wandb_run_name
    if run_name is None:
        run_name = f"es_bm25_k{data_args.top_k_retrieval}"
    
    wandb.init(
        project=data_args.wandb_project,
        name=run_name,
        config={
            # Retrieval parameters
            "retrieval_method": "ElasticSearch_BM25",
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
    print(f"{'='*50}\n")
    
    return wandb


def main():

    # wandb 
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    ########

    training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

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
    ###########

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

    should_run_retrieval = data_args.eval_retrieval
    # wandb
    retrieval_metrics = None
    if should_run_retrieval:
        datasets, retrieval_metrics = run_sparse_retrieval(
            tokenizer.tokenize,
            datasets,
            training_args,
            data_args,
        )

        # wandb에 retrieval metrics 로깅
        if wandb is not None and retrieval_metrics is not None:
            wandb.log({
                "retrieval/accuracy": retrieval_metrics.get("retrieval_accuracy"),
                "retrieval/mrr": retrieval_metrics.get("mrr"),
                "retrieval/correct_count": retrieval_metrics.get("correct_count"),
                "retrieval/total_count": retrieval_metrics.get("total_count"),
                "retrieval/top_k": retrieval_metrics.get("top_k"),
            })
            print(f"[Wandb] Logged retrieval metrics: {retrieval_metrics}")


    should_run_mrc = training_args.do_eval or training_args.do_predict
    if should_run_mrc:
        mrc_metrics = run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

        if wandb is not None and mrc_metrics is not None:
            wandb.log({
                "mrc/exact_match": mrc_metrics.get("exact_match"),
                "mrc/f1": mrc_metrics.get("f1"),
            })
            print(f"[Wandb] Logged MRC metrics: {mrc_metrics}")

        # Finish wandb run
    if wandb is not None:
        import wandb as _wandb
        _wandb.finish()
        print("[Wandb] Run finished successfully!")



def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],  # ES) ES에서는 안 씀 
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "data",
    context_path: str = "wikipedia_documents.json",
    es_host: str = "http://localhost:9200",
    es_index: str = "wiki_mrc",
) -> Tuple[DatasetDict, Dict]:
    """
    ES
    기존 run_sparse_retrieval -> ES 버전으로 바꾼 함수 
    - validation split에 대해 ElasticSearchRetrieval로 top-k 검색 수행
    - 반환: retrieval 결과가 반영된 DatasetDict({"validation": ...})
    """

    # ✅ ES 리트리버 초기화
    es_retriever = ElasticSearchRetrieval(
        tokenize_fn=tokenize_fn,          # 내부에서 안 써도 시그니처 맞춰 전달
        data_path=data_path,
        context_path=context_path,
        es_host=es_host,
        index_name=es_index,
    )

    # ✅ 인덱스 없으면 생성, 있으면 재사용
    es_retriever.build_elasticsearch_index(recreate=False)

    # top-k 값은 기존 data_args.top_k_retrieval 재사용
    topk = data_args.top_k_retrieval

    # validation 셋에 대해 retrieval 수행 → pandas DataFrame 반환
    retrieved_df = es_retriever.retrieve(
        datasets["validation"], topk=topk
    )

    # wandb
    # 🔹 retrieval metrics 계산 (Hit@k 기준 간단 버전)
    retrieval_metrics = None
    if "original_context" in retrieved_df.columns:
        # 정답 문서가 top-k context 문자열 안에 포함되었는지만 확인
        """
        correct_mask = retrieved_df["context"].str.contains(
            retrieved_df["original_context"], regex=False
        )
        """
        correct_mask = retrieved_df.apply(
            lambda row: row["original_context"] in row["context"],
            axis=1
        )
        correct_count = int(correct_mask.sum())
        total_count = len(retrieved_df)
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        retrieval_metrics = {
            "retrieval_accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "top_k": topk,
            # MRR는 doc rank 정보가 없어서 여기선 생략하거나 None으로 둠
            "mrr": None,
        }

        # MRC 학습용으로는 original_context 필요 없으니 제거
        retrieved_df = retrieved_df.drop(columns=["original_context"])
    #######


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
) -> Dict:

    val_column_names = datasets["validation"].column_names

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
        load_from_cache_file=not data_args.overwrite_cache,
    )

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

    metric = evaluate.load("squad")

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
    
    # wandb
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

        qa_trainer.log_metrics("test", eval_metrics)
        qa_trainer.save_metrics("test", eval_metrics)

        mrc_metrics = {
            "exact_match": eval_metrics.get("exact_match"),
            "f1": eval_metrics.get("f1"),
        }

    return mrc_metrics

if __name__ == "__main__":
    main()
