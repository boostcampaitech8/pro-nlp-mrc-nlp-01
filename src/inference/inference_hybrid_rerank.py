import logging
import os
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

import numpy as np
import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
)
import evaluate
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

from src.config.arguments import DataTrainingArguments, ModelArguments
from src.retrieval.retrieval_bm25_wandb import BM25RetrievalWithMetrics
from src.retrieval.retrieval_dpr import DenseRetrieval
from src.training.trainer_qa import QuestionAnsweringTrainer
from src.utils.utils_qa import postprocess_qa_predictions

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    
    # Custom args for this script
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 (0.0-1.0)")
    parser.add_argument("--top_k_rerank", type=int, default=40, help="Number of documents to rerank (expand)")

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        outputs = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        model_args, data_args, training_args = outputs[:3]
        custom_args_namespace = outputs[3]
        alpha = custom_args_namespace.alpha
        top_k_rerank = custom_args_namespace.top_k_rerank

    print(f"Model: {model_args.model_name_or_path}")
    print(f"Data: {data_args.dataset_name}")
    print(f"Hybrid Alpha: {alpha}")
    print(f"Top-K Rerank: {top_k_rerank}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    
    # Load Model & Tokenizer
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
    )

    # 1. Hybrid Retrieval (Get individual passages)
    if data_args.eval_retrieval:
        datasets = run_hybrid_retrieval_expand(
            tokenizer,
            datasets,
            training_args,
            data_args,
            model_args,
            alpha=alpha,
            top_k=top_k_rerank
        )

    # 2. Run MRC (Reader) on Expanded Dataset
    if training_args.do_eval or training_args.do_predict:
        run_mrc_rerank(data_args, training_args, model_args, datasets, tokenizer, model)

def run_hybrid_retrieval_expand(
    tokenizer,
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    alpha: float = 0.5,
    top_k: int = 40,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    
    # Initialize Retrievers
    print("Initializing Retrievers for Reranking...")
    bm25 = BM25RetrievalWithMetrics(tokenizer.tokenize, data_path=data_path, context_path=context_path)
    bm25.get_sparse_embedding()
    
    dpr_path = model_args.retriever_name_or_path or "outputs/train_dataset_hn"
    dpr = DenseRetrieval(training_args, None, dpr_path, data_path=data_path, context_path=context_path)
    dpr.get_dense_embedding()
    dpr.build_faiss()
    
    target_split = "validation" if "validation" in datasets else "test"
    dataset = datasets[target_split]
    queries = dataset["question"]
    ids = dataset["id"]
    
    # Bulk Retrieval
    search_k = top_k * 3 # Fetch more for hybrid mixing
    print(f"Retrieving Top-{search_k} candidates...")
    
    b_scores_list, b_indices_list = bm25.get_relevant_doc_bulk(queries, k=search_k)
    d_scores_list, d_indices_list = dpr.get_relevant_doc_bulk(queries, k=search_k)
    
    expanded_data = [] # List of dicts
    
    contexts = bm25.contexts # Shared context list
    
    for i, query in enumerate(tqdm(queries, desc="Hybrid Expansion")):
        # Mix Scores
        b_s, b_i = b_scores_list[i], b_indices_list[i]
        d_s, d_i = d_scores_list[i], d_indices_list[i]
        
        # Normalize
        if b_s:
            b_min, b_max = min(b_s), max(b_s)
            b_norm = [(x - b_min)/(b_max - b_min + 1e-9) for x in b_s]
        else: b_norm = []
        
        if d_s:
            d_min, d_max = min(d_s), max(d_s)
            d_norm = [(x - d_min)/(d_max - d_min + 1e-9) for x in d_s]
        else: d_norm = []
        
        b_map = {idx: s for idx, s in zip(b_i, b_norm)}
        d_map = {idx: s for idx, s in zip(d_i, d_norm)}
        
        all_indices = set(b_i) | set(d_i)
        hybrid_scores = []
        for idx in all_indices:
            s_b = b_map.get(idx, 0.0)
            s_d = d_map.get(idx, 0.0)
            score = alpha * s_b + (1-alpha) * s_d
            hybrid_scores.append((idx, score))
        
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = hybrid_scores[:top_k]
        
        # Create Expanded Examples
        # Logic: For 1 Question, we make K examples.
        # We need to track which question they belong to, to aggregate later.
        
        qid = ids[i]
        
        for rank, (doc_idx, score) in enumerate(top_candidates):
            ctx = contexts[doc_idx]
            expanded_data.append({
                "question": query,
                "context": ctx,
                "id": f"{qid}_{rank}", # Unique ID for Trainer
                "original_id": qid, # To group back
                "retrieval_score": score,
                "rank": rank
            })
            
    print(f"Expanded {len(queries)} queries to {len(expanded_data)} examples (Top-{top_k})")
    
    # Create Dataset
    features = Features({
        "question": Value("string"),
        "context": Value("string"),
        "id": Value("string"),
        "original_id": Value("string"),
        "retrieval_score": Value("float32"),
        "rank": Value("int32")
    })
    
    new_dataset = Dataset.from_pandas(pd.DataFrame(expanded_data), features=features)
    datasets[target_split] = new_dataset
    return datasets

def run_mrc_rerank(
    data_args, training_args, model_args, datasets, tokenizer, model
):
    eval_dataset = datasets["validation"] if "validation" in datasets else datasets["test"]
    
    # Preprocessing
    max_seq_length = data_args.max_seq_length
    pad_to_max_length = data_args.pad_to_max_length
    padding = "max_length" if pad_to_max_length else False
    
    def prepare_features(examples):
        # inputs: question, context
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=padding,
        )

        # Fix for RoBERTa: remove token_type_ids if present (cause of CUDA device-side assert)
        if "token_type_ids" in tokenized_examples:
            tokenized_examples.pop("token_type_ids")
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        
        for i in range(len(tokenized_examples["input_ids"])):
            sample_idx = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_idx])
            
            # Offset mapping fix
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 
            offset_mapping = tokenized_examples["offset_mapping"][i]
            tokenized_examples["offset_mapping"][i] = [
                o if sequence_ids[k] == context_index else None
                for k, o in enumerate(offset_mapping)
            ]
            
        return tokenized_examples

    
    # Check if expanded dataset exists to speed up debugging
    expanded_dataset_path = os.path.join(training_args.output_dir, "expanded_dataset")
    if os.path.exists(expanded_dataset_path) and not training_args.overwrite_output_dir:
        print(f"Loading expanded dataset from {expanded_dataset_path}...")
        processed_dataset = load_from_disk(expanded_dataset_path)
    else:
        print("Tokenizing expanded dataset...")
        processed_dataset = eval_dataset.map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
        )
        print(f"Saving expanded dataset to {expanded_dataset_path}...")
        processed_dataset.save_to_disk(expanded_dataset_path)
    
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Dummy metrics to force Trainer to output predictions (not just loss)
    def compute_metrics(p): return {}

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        eval_dataset=processed_dataset,
        eval_examples=eval_dataset, 
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, # Added to enforce prediction output
    )
    
    print("Running QA Inference on expanded candidates...")
    predictions = trainer.predict(test_dataset=processed_dataset, test_examples=eval_dataset)
    # predictions.predictions is tuple (start_logits, end_logits)
    
    start_logits, end_logits = predictions.predictions
    
    # Reranking Selection Logic
    # We need to map predictions back to {original_id -> best_answer}
    
    # 1. Map example_id (qid_rank) back to predictions
    # processed_dataset has 'example_id'
    
    # Store all candidates: { original_id: [ (text, score, rank), ... ] }
    from collections import defaultdict
    candidates_map = defaultdict(list)
    
    # We need to robustly extract answer text from logits
    # Using simple nbest logic or just argmax for now?
    # Better to use `postprocess_qa_predictions` logic but it's designed for sliding window aggregation on ONE document.
    # Here we have distinct documents.
    # We can treat each (question, doc) as a separate QA task.
    
    # To use existing utilities, we can iterate
    
    # Lightweight Answer Extraction
    n_best_size = 20
    max_answer_len = 30
    
    print("Selecting Best Answers...")
    
    # Map feature_index -> example_index of eval_dataset
    # But processed_dataset is flattend.
    # We iterate over processed features.
    
    for feature_idx in tqdm(range(len(processed_dataset))):
        start_logit = start_logits[feature_idx]
        end_logit = end_logits[feature_idx]
        
        # Get feature info
        example_id = processed_dataset[feature_idx]["example_id"] # "qid_rank"
        # Recover context
        # We need mapping from example_id to context.
        # eval_dataset has filtering support?
        
        # This is slow if we look up every time.
        # Let's pre-build map: qid_rank -> context
    
    # Optimized Approach:
    # 1. Group features by example_id (dataset row)
    # 2. For each dataset row (Question+Context), find best span & score.
    # 3. Group by original_id (Question).
    # 4. Pick best score among all contexts.
    
    # Pre-compute feature to example map? No, just loop.
    
    # Let's make a lookup for eval_dataset
    id_to_example = {row["id"]: row for row in eval_dataset}
    
    results = {} # original_id -> {text, score}
    
    for i, feature in enumerate(tqdm(processed_dataset, desc="Scoring Candidates")):
        ex_id = feature["example_id"] # qid_rank
        original_id = id_to_example[ex_id]["original_id"]
        
        s_logits = start_logits[i]
        e_logits = end_logits[i]
        
        # Top-K indices
        start_indexes = np.argsort(s_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(e_logits)[-1 : -n_best_size - 1 : -1].tolist()
        
        valid_answers = []
        
        # input_ids needed to decode
        input_ids = feature["input_ids"]
        
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= len(input_ids) or end_index >= len(input_ids): continue
                if end_index < start_index: continue
                if end_index - start_index + 1 > max_answer_len: continue
                
                # Check if in context (offset mapping is None for query/pad)
                offset = feature["offset_mapping"][start_index]
                if offset is None: continue
                
                score = s_logits[start_index] + e_logits[end_index]
                
                # Decode
                span_ids = input_ids[start_index : end_index + 1]
                text = tokenizer.decode(span_ids, skip_special_tokens=True)
                
                valid_answers.append({
                    "text": text,
                    "score": score
                })
        
        if not valid_answers:
            best_ans = {"text": "", "score": -9999.0}
        else:
            best_ans = max(valid_answers, key=lambda x: x["score"])
            
        # Add Retrieval Score impact?
        # Typically Reader Score is dominant. We use retrieval score just for sorting candidates input.
        
        # Update Global Best for this Original Question
        if original_id not in results:
            results[original_id] = best_ans
        else:
            if best_ans["score"] > results[original_id]["score"]:
                results[original_id] = best_ans

    # Format Output
    final_predictions = []
    for qid in results:
        final_predictions.append({"id": qid, "prediction_text": results[qid]["text"]})
        
    output_csv = os.path.join(training_args.output_dir, "predictions_submit.csv")
    df = pd.DataFrame(final_predictions)
    # Evaluator expects TSV format without header despite .csv extension
    df.to_csv(output_csv, index=False, sep='\t', header=False)
    print(f"Saved submission to {output_csv}")
    
    import json
    with open(os.path.join(training_args.output_dir, "predictions.json"), "w") as f:
        # Convert to standard format {id: text}
        json_dict = {item["id"]: item["prediction_text"] for item in final_predictions}
        json.dump(json_dict, f, indent=4, ensure_ascii=False)

    # Calculate EM/F1 if answers are available
    if "answers" in eval_dataset.column_names:
        print("Calculating EM/F1 metrics...")
        metric = evaluate.load("squad")
        
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_dataset]
        # predictions format for metric: {'id': '...', 'prediction_text': '...'}
        
        metrics = metric.compute(predictions=final_predictions, references=references)
        print(f" >> EM: {metrics['exact_match']:.2f}, F1: {metrics['f1']:.2f}")
        
        with open(os.path.join(training_args.output_dir, "results.json"), "w") as f:
            json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
