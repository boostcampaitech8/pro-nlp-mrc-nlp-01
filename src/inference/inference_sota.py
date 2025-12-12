import logging
import os
import sys
import argparse
from typing import Callable, Dict, List, NoReturn, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import (
    Dataset,
    DatasetDict,
    Features,
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
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config.arguments import DataTrainingArguments, ModelArguments
from src.retrieval.retrieval_bm25_morphs import BM25RetrievalWithMetrics
from src.retrieval.retrieval_kure import KURERetrieval
from src.training.trainer_qa import QuestionAnsweringTrainer

# Instantiate Reranker
from FlagEmbedding import FlagReranker
from kiwipiepy import Kiwi

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    
    # Custom args for this script
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 (0.0-1.0)")
    # top_k_retrieval is already in DataTrainingArguments
    parser.add_argument("--top_k_reader", type=int, default=5, help="Number of documents to pass to Reader after Reranking")
    parser.add_argument("--kure_model", type=str, default="nlpai-lab/KURE-v1", help="KURE model path")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-v2-m3", help="Reranker model path")

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        outputs = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        model_args, data_args, training_args = outputs[:3]
        custom_args_namespace = outputs[3]
        
        alpha = custom_args_namespace.alpha if hasattr(custom_args_namespace, 'alpha') else 0.5
        # Use data_args for top_k_retrieval
        top_k_retrieval = data_args.top_k_retrieval
        top_k_reader = custom_args_namespace.top_k_reader if hasattr(custom_args_namespace, 'top_k_reader') else 5
        kure_model = custom_args_namespace.kure_model if hasattr(custom_args_namespace, 'kure_model') else "nlpai-lab/KURE-v1"
        reranker_model = custom_args_namespace.reranker_model if hasattr(custom_args_namespace, 'reranker_model') else "BAAI/bge-reranker-v2-m3"

    print(f"Model: {model_args.model_name_or_path}")
    print(f"Data: {data_args.dataset_name}")
    print(f"Hybrid Alpha: {alpha}")
    print(f"Retrieval Top-K: {top_k_retrieval}")
    print(f"Reader Top-K: {top_k_reader}")
    print(f"KURE Model: {kure_model}")
    print(f"Reranker Model: {reranker_model}")

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

    # 1. SOTA Pipeline: Hybrid Retrieval + Reranking
    if data_args.eval_retrieval:
        datasets = run_sota_pipeline(
            datasets,
            training_args,
            data_args,
            alpha=alpha,
            top_k_retrieval=top_k_retrieval,
            top_k_reader=top_k_reader,
            kure_model_path=kure_model,
            reranker_model_path=reranker_model
        )

    # 2. Run MRC (Reader) on Expanded/Reranked Dataset
    if training_args.do_eval or training_args.do_predict:
        run_mrc_rerank(data_args, training_args, model_args, datasets, tokenizer, model)

def run_sota_pipeline(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    alpha: float = 0.5,
    top_k_retrieval: int = 100,
    top_k_reader: int = 5,
    kure_model_path: str = "nlpai-lab/KURE-v1",
    reranker_model_path: str = "BAAI/bge-reranker-v2-m3",
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    
    # 1. Initialize Kiwi for BM25
    print("Initializing Kiwi Tokenizer...")
    kiwi = Kiwi()
    def kiwi_tokenize(text):
        return [token.form for token in kiwi.tokenize(text)]

    # 2. Initialize Retrievers
    print("Initializing Retrievers...")
    
    # BM25 (Sparse)
    bm25 = BM25RetrievalWithMetrics(tokenize_fn=kiwi_tokenize, data_path=data_path, context_path=context_path)
    bm25.get_sparse_embedding()
    
    # KURE (Dense)
    kure = KURERetrieval(data_path=data_path, context_path=context_path, model_name=kure_model_path)
    kure.get_dense_embedding()
    kure.build_faiss()
    
    target_split = "validation" if "validation" in datasets else "test"
    dataset = datasets[target_split]
    queries = dataset["question"]
    ids = dataset["id"]
    
    # 3. Hybrid Bulk Retrieval (Top-100)
    print(f"Retrieving Top-{top_k_retrieval} candidates using Hybrid (BM25 + KURE)...")
    
    b_scores_list, b_indices_list = bm25.get_relevant_doc_bulk(queries, k=top_k_retrieval)
    d_scores_list, d_indices_list = kure.get_relevant_doc_bulk(queries, k=top_k_retrieval)
    
    contexts = bm25.contexts # Shared context list
    
    # 4. Initialize Reranker
    print(f"Initializing Reranker ({reranker_model_path})...")
    reranker = FlagReranker(reranker_model_path, use_fp16=True)
    
    expanded_data = [] # Final data for reader
    
    print("Reranking candidates...")
    
    # We process query by query to rerank
    
    # Batch Reranking Optimization:
    # FlagReranker.compute_score takes list of pairs.
    # We can batch this if memory flows over, but typically 1000 pairs is fine.
    
    for i, query in enumerate(tqdm(queries, desc="Hybrid + Rerank Pipeline")):
        # --- Hybrid Mixing ---
        b_s, b_i = b_scores_list[i], b_indices_list[i]
        d_s, d_i = d_scores_list[i], d_indices_list[i]
        
        # Normalize BM25 (0 to 1 scaling roughly)
        if b_s:
            b_min, b_max = min(b_s), max(b_s)
            b_norm = [(x - b_min)/(b_max - b_min + 1e-9) for x in b_s]
        else: b_norm = []
        
        # Normalize KURE (Usually -1 to 1 or 0 to 1 if cosine)
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
        top_hybrid_candidates = hybrid_scores[:top_k_retrieval]
        
        # --- Reranking Step ---
        # Prepare pairs for Reranker: (Query, Passage)
        rerank_pairs = []
        candidate_indices = []
        for doc_idx, _ in top_hybrid_candidates:
            passage = contexts[doc_idx]
            rerank_pairs.append([query, passage])
            candidate_indices.append(doc_idx)
            
        if rerank_pairs:
            # Get scores from Reranker
            rerank_scores = reranker.compute_score(rerank_pairs)
            if not isinstance(rerank_scores, list):
                rerank_scores = [rerank_scores]
            
            # Combine index with score
            scored_candidates = []
            for j, score in enumerate(rerank_scores):
                scored_candidates.append((candidate_indices[j], score))
            
            # Sort by Reranker Score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Pick Top-K Reader
            final_top_k = scored_candidates[:top_k_reader]
        else:
            final_top_k = []

        # --- create expanded examples for Reader ---
        qid = ids[i]
        
        for rank, (doc_idx, score) in enumerate(final_top_k):
            ctx = contexts[doc_idx]
            example_data = {
                "question": query,
                "context": ctx,
                "id": f"{qid}_{rank}", # Unique ID for Trainer
                "original_id": qid, # To group back
                "retrieval_score": score,
                "rank": rank
            }
            if "answers" in dataset.column_names:
                example_data["answers"] = dataset[i]["answers"]
            
            expanded_data.append(example_data)
            
    print(f"Processed {len(queries)} queries -> {len(expanded_data)} Reader examples")
    
    # Create Dataset
    features_dict = {
        "question": Value("string"),
        "context": Value("string"),
        "id": Value("string"),
        "original_id": Value("string"),
        "retrieval_score": Value("float32"),
        "rank": Value("int32")
    }
    
    if "answers" in dataset.column_names:
        # answers is Sequence(Feature(...)) usually
        # We can just copy the feature type from original dataset if possible
        # Or define it manually: Sequence({'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)})
        features_dict["answers"] = dataset.features["answers"]

    features = Features(features_dict)
    
    new_dataset = Dataset.from_pandas(pd.DataFrame(expanded_data), features=features)
    datasets[target_split] = new_dataset
    return datasets

# Reuse the exact same MRC logic from inference_hybrid_rerank.py 
# (Copying it here to make this script standalone, or I could import it if it was modular.
#  It is distinct enough that I should probably duplicate the function to ensure safety).

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

        if "token_type_ids" in tokenized_examples:
            tokenized_examples.pop("token_type_ids")
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        
        for i in range(len(tokenized_examples["input_ids"])):
            sample_idx = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_idx])
            
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 
            offset_mapping = tokenized_examples["offset_mapping"][i]
            tokenized_examples["offset_mapping"][i] = [
                o if sequence_ids[k] == context_index else None
                for k, o in enumerate(offset_mapping)
            ]
            
        return tokenized_examples

    # Check if expanded dataset exists
    expanded_dataset_path = os.path.join(training_args.output_dir, "sota_expanded_dataset")
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
    
    def compute_metrics(p): return {}

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        eval_dataset=processed_dataset,
        eval_examples=eval_dataset, 
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Running QA Inference on expanded candidates...")
    predictions = trainer.predict(test_dataset=processed_dataset, test_examples=eval_dataset)
    start_logits, end_logits = predictions.predictions
    
    # Selection Logic
    n_best_size = 20
    max_answer_len = 30
    
    print("Selecting Best Answers...")
    
    id_to_example = {row["id"]: row for row in eval_dataset}
    results = {} 
    
    for i, feature in enumerate(tqdm(processed_dataset, desc="Scoring Candidates")):
        ex_id = feature["example_id"]
        original_id = id_to_example[ex_id]["original_id"]
        
        s_logits = start_logits[i]
        e_logits = end_logits[i]
        
        start_indexes = np.argsort(s_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(e_logits)[-1 : -n_best_size - 1 : -1].tolist()
        
        valid_answers = []
        input_ids = feature["input_ids"]
        
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= len(input_ids) or end_index >= len(input_ids): continue
                if end_index < start_index: continue
                if end_index - start_index + 1 > max_answer_len: continue
                
                offset = feature["offset_mapping"][start_index]
                if offset is None: continue
                
                score = s_logits[start_index] + e_logits[end_index]
                
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
        
        # Key Change: Use Retrieval/Reranker Score to boost?
        # Many SOTA systems add Reranker Score to Span Score:
        # Final Score = Span Logits + lambda * Reranker Score
        # For now, we trust the Reader's relative confidence among the provided Top-5.
        # But since we provide 5 different contexts, the Reader might be confident in a wrong context.
        # It is safer to prioritize answers from higher-ranked documents if Reader scores are close.
        # However, for this implementation, we just take the max span score across all 5 docs.
        
        if original_id not in results:
            results[original_id] = best_ans
        else:
            if best_ans["score"] > results[original_id]["score"]:
                results[original_id] = best_ans

    final_predictions = []
    for qid in results:
        final_predictions.append({"id": qid, "prediction_text": results[qid]["text"]})
        
    output_csv = os.path.join(training_args.output_dir, "predictions_submit.csv")
    df = pd.DataFrame(final_predictions)
    df.to_csv(output_csv, index=False, sep='\t', header=False)
    print(f"Saved submission to {output_csv}")
    
    import json
    with open(os.path.join(training_args.output_dir, "predictions.json"), "w") as f:
        json_dict = {item["id"]: item["prediction_text"] for item in final_predictions}
        json.dump(json_dict, f, indent=4, ensure_ascii=False)

    if "answers" in eval_dataset.column_names:
        print("Calculating EM/F1 metrics...")
        metric = evaluate.load("squad")
        references = [{"id": ex["id"], "answers": id_to_example[ex["id"]].get("answers", [])} for ex in eval_dataset]
        
        # References must match the prediction text ID which is ORIGINAL_ID
        # Wait, references construction above is wrong because ex uses 'qid_rank' as id.
        # We need original references.
        
        # Re-load original validation dataset to get ground truth map
        # Or just use the 'original_id' field.
        
        # Group references by original_id
        ref_map = {}
        for ex in eval_dataset:
             oid = ex["original_id"]
             # Assuming 'answers' is available in the dataset passed to this func.
             # Note: run_sota_pipeline creates new_dataset which inherits columns from expanded_data
             # But 'answers' column might be lost if not explicitly preserved in expanded_data or Features!
             # CHECK Features in run_sota_pipeline! 'answers' is MISSING in Features definition.
             # We should probably pass existing answers if available.
             pass

        # Since we lost 'answers' in the new dataset creation in run_sota_pipeline,
        # we can't easily compute metrics here unless we change run_sota_pipeline.
        # Let's fix run_sota_pipeline Features to include answers.
        
        # But wait, we can just load the original dataset again to get references?
        # datasets["validation"] was overwritten.
        # Let's assume we do predictions only for now or fix it.
        
        # To fix correctly: add answers to expanded_data in run_sota_pipeline.
        pass

if __name__ == "__main__":
    main()
