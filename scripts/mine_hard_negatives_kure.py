import os
import sys
import argparse
import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import load_from_disk, Dataset, DatasetDict

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.retrieval.retrieval_kure import KURERetrieval

def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives using KURE (Dense Retrieval)")
    parser.add_argument("--dataset_name", type=str, default="./data/train_dataset", help="Path to original dataset")
    parser.add_argument("--output_dir", type=str, default="./data/train_dataset_hn_kure", help="Path to save new dataset")
    parser.add_argument("--model_name_or_path", type=str, default="nlpai-lab/KURE-v1", help="Model for retrieval")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--context_path", type=str, default="wikipedia_documents.json", help="Context file name")
    parser.add_argument("--top_k", type=int, default=100, help="Number of documents to retrieve for mining")
    parser.add_argument("--num_hn", type=int, default=1, help="Number of hard negatives to mine per query")
    
    args = parser.parse_args()

    # 1. Load Dataset
    print(f"Loading dataset from {args.dataset_name}...")
    try:
        datasets = load_from_disk(args.dataset_name)
        train_dataset = datasets["train"]
        valid_dataset = datasets["validation"]
    except:
        print(f"Could not load from disk, trying load_dataset...")
        # Fallback handling if needed, or just let error propagate
        raise
        
    print(f"Train dataset size: {len(train_dataset)}")

    # 2. Initialize KURE Retriever
    print(f"Initializing KURE Retriever with {args.model_name_or_path}...")
    retriever = KURERetrieval(
        data_path=args.data_path,
        context_path=args.context_path,
        model_name=args.model_name_or_path
    )
    # Build/Load Embedding & Index
    retriever.get_dense_embedding()
    retriever.build_faiss()

    # 3. Batch Retrieval
    print(f"Retrieving top-{args.top_k} documents for all queries...")
    queries = train_dataset["question"]
    
    # get_relevant_doc_bulk uses GPU and batching internally
    doc_scores_list, doc_indices_list = retriever.get_relevant_doc_bulk(queries, k=args.top_k)

    # 4. Mine Hard Negatives
    print("Mining Hard Negatives...")
    new_data = []
    
    # We will pick hard negatives that don't contain the answer
    
    for i, sample in enumerate(tqdm(train_dataset, desc="Mining")):
        original_context = sample["context"]
        answers = sample["answers"]
        
        # Normalize answers to a list of strings
        ans_list = []
        if isinstance(answers, dict) and 'text' in answers:
            ans_list = answers['text']
        elif isinstance(answers, list):
             if len(answers) > 0 and isinstance(answers[0], dict) and 'text' in answers[0]:
                 ans_list = [a['text'] for a in answers]
             elif len(answers) > 0 and isinstance(answers[0], str):
                 ans_list = answers

        doc_indices = doc_indices_list[i]
        
        found_hns = []
        
        for doc_idx in doc_indices:
            if len(found_hns) >= args.num_hn:
                break
                
            retrieved_context = retriever.contexts[doc_idx]
            
            # 1. Exact match check (Skip Positive)
            if retrieved_context == original_context:
                continue
            
            # 2. Answer containment check (Skip False Negative / Positive)
            has_answer = False
            for ans_text in ans_list:
                if ans_text in retrieved_context:
                    has_answer = True
                    break
            
            if has_answer:
                continue
            
            # Found a hard negative
            found_hns.append(retrieved_context)
            
        # Fallback: if not enough hard negatives found, pick random from retrieved (if valid) or random from corpus
        # Priority: Retrieved (non-positive) > Random Corpus
        
        # Try to fill from retrieved non-answers first (already done above)
        # If we are here, we exhausted retrieved or they all had answers/were positive (unlikely for 100)
        
        # Fill with random if still missing
        while len(found_hns) < args.num_hn:
            rand_idx = random.randint(0, len(retriever.contexts)-1)
            rand_ctx = retriever.contexts[rand_idx]
            if rand_ctx != original_context:
                found_hns.append(rand_ctx)
        
        # Create new sample
        sample_copy = sample.copy()
        
        # For compatibility with training script expecting single 'hard_negative_context'
        if args.num_hn == 1:
            sample_copy["hard_negative_context"] = found_hns[0]
        else:
             # If multiple, maybe join them or store as list?
             # train_kure.py adapted from train_dpr.py usually handles one. 
             # Or we can put a list. Let's stick to 1 logic for now or list if requested.
             # User requested optimal, we decided 1.
             sample_copy["hard_negative_context"] = found_hns[0]
             
        new_data.append(sample_copy)

    # 5. Save New Dataset
    print(f"Saving new dataset to {args.output_dir}...")
    df = pd.DataFrame(new_data)
    
    # Preserve features
    # Dataset.from_pandas often infers types. 
    # To be safe, we can cast 'answers' if needed, but usually it works.
    train_dataset_hn = Dataset.from_pandas(df)
    
    # Reconstruct DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset_hn,
        "validation": valid_dataset
    })
    
    dataset_dict.save_to_disk(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
