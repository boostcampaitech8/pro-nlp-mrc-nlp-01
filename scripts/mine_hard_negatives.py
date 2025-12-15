
import os
import sys
import pickle
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_from_disk, Dataset, DatasetDict
from multiprocessing import Pool, cpu_count
from functools import partial

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.retrieval.retrieval_bm25_wandb import BM25RetrievalWithMetrics
from transformers import AutoTokenizer

def process_chunk(chunk_data, retriever_contexts, top_k):
    """
    Process a chunk of queries to find hard negatives.
    chunk_data: list of (index, sample)
    retriever_contexts: list of all contexts (from retriever)
    """
    results = []
    
    # We need a local instance or access to BM25. 
    # Since BM25 object is large (pickle), passing it to every process might be heavy if not careful.
    # However, 'rank_bm25' object is picklable.
    # But usually, it's better to share the memory or just do scoring in parallel.
    
    # Actually, the bottleneck is 'get_scores'.
    # If we pass the 'retriever' object, it might be large.
    # Let's rely on the fact that we can just perform the search in parallel.
    # But we need the 'retriever' object in the worker.
    
    # Alternative: Use 'retriever.get_relevant_doc_bulk' but parallelize IT.
    # But 'retriever' holds 'self.bm25'.
    pass

# Global variable for worker processes to share the retriever
global_retriever = None

def init_worker(dataset_name, model_name_or_path, data_path, context_path):
    global global_retriever
    
    # Re-initialize tokenizer and retriever in each worker
    # This might be slow to init, but faster for processing if N is large.
    # Actually, loading BM25 pickle (200MB+) in every process is memory heavy.
    # Better approach: parallelize ONLY the calculation if possible, or use ThreadPool if GIL is not the issue.
    # BM25 get_scores is numpy heavy -> releases GIL? 
    # rank_bm25 uses list comp + numpy, so it might hold GIL.
    
    # Let's try to just load it once in global if we use fork (Linux default).
    # On Linux, global variables are inherited by child processes (COW).
    pass

def worker_retrieve_and_mine(data_chunk, top_k):
    # global_retriever is available here
    global global_retriever
    
    processed_samples = []
    
    # Batch retrieve
    queries = [sample["question"] for sample in data_chunk]
    doc_scores_list, doc_indices_list = global_retriever.get_relevant_doc_bulk(queries, k=top_k)
    
    for i, sample in enumerate(data_chunk):
        original_context = sample["context"]
        answers = sample["answers"]
        
        doc_indices = doc_indices_list[i]
        doc_scores = doc_scores_list[i]
        
        found_hn = False
        hn_context = ""
        
        for rank, doc_idx in enumerate(doc_indices):
            retrieved_context = global_retriever.contexts[doc_idx]
            
            # 1. Exact match check
            if retrieved_context == original_context:
                continue
            
            # 2. Answer containment check
            has_answer = False
            ans_list = [a['text'] for a in answers] if isinstance(answers, list) else ([answers['text'][0]] if 'text' in answers else [])
            # Fix answer extraction relative to dataset format
            # In dataset, answers is usually a dict: {'text': [...], 'answer_start': [...]}
            if isinstance(answers, dict) and 'text' in answers:
                ans_list = answers['text']
            elif isinstance(answers, list) and len(answers) > 0 and 'text' in answers[0]:
                ans_list = [a['text'] for a in answers]
                
            for ans_text in ans_list:
                if ans_text in retrieved_context:
                    has_answer = True
                    break
            
            if has_answer:
                continue
                
            # Found Hard Negative
            hn_context = retrieved_context
            found_hn = True
            break
        
        if not found_hn:
            # Fallback: Pick a random one from top-k (filtering out positive)
            for idx in range(len(doc_indices)-1, -1, -1):
                ctx = global_retriever.contexts[doc_indices[idx]]
                if ctx != original_context:
                     hn_context = ctx
                     found_hn = True
                     break
        
        if not found_hn:
             hn_context = global_retriever.contexts[random.randint(0, len(global_retriever.contexts)-1)]

        sample_copy = sample.copy()
        sample_copy["hard_negative_context"] = hn_context
        processed_samples.append(sample_copy)
        
    return processed_samples

def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives using BM25")
    parser.add_argument("--dataset_name", type=str, default="./data/train_dataset", help="Path to original dataset")
    parser.add_argument("--output_dir", type=str, default="./data/train_dataset_hn", help="Path to save new dataset")
    parser.add_argument("--model_name_or_path", type=str, default="klue/bert-base", help="Model for tokenizer")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--context_path", type=str, default="wikipedia_documents.json", help="Context file name")
    parser.add_argument("--top_k", type=int, default=100, help="Number of documents to retrieve for mining")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes")
    
    args = parser.parse_args()
    
    # 1. Load Dataset
    print(f"Loading dataset from {args.dataset_name}...")
    datasets = load_from_disk(args.dataset_name)
    train_dataset = datasets["train"]
    print(f"Train dataset size: {len(train_dataset)}")
    
    # 2. Initialize BM25 Retriever (Global)
    global global_retriever
    print(f"Initializing BM25 Retriever with {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    
    global_retriever = BM25RetrievalWithMetrics(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )
    # Check/Load Embedding
    global_retriever.get_sparse_embedding()
    
    # 3. Parallel Mining
    print(f"Mining Hard Negatives with {args.num_proc} processes...")
    
    data_list = [sample for sample in train_dataset]
    chunk_size = (len(data_list) + args.num_proc - 1) // args.num_proc
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
    
    # Use multiprocessing with global_retriever inherited
    # Forking approach works on Linux
    with Pool(processes=args.num_proc) as pool:
        results = list(tqdm(
            pool.imap(partial(worker_retrieve_and_mine, top_k=args.top_k), chunks),
            total=len(chunks),
            desc="Mining"
        ))
        
    # Flatten results
    new_data = [item for sublist in results for item in sublist]

    # 4. Save New Dataset
    print(f"Saving new dataset to {args.output_dir}...")
    df = pd.DataFrame(new_data)
    
    # Reconstruct Dataset
    # We can rely on Auto Feature inference or copy from original
    train_dataset_hn = Dataset.from_pandas(df)
    
    # Ensure features match original for verification
    # But usually simpler is fine for training unless we rely on complex features
    
    dataset_dict = DatasetDict({
        "train": train_dataset_hn,
        "validation": datasets["validation"]
    })
    
    dataset_dict.save_to_disk(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
