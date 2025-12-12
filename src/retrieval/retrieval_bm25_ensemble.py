import json
import os
import pickle
import time
import random
import argparse
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm
from kiwipiepy import Kiwi
from transformers import AutoTokenizer

# Import existing retrieval classes
from .retrieval_bm25_wandb import BM25RetrievalWithMetrics as BM25RetrievalWandb
from .retrieval_bm25_morphs import BM25RetrievalWithMetrics as BM25RetrievalMorphs

seed = 2024
random.seed(seed)
np.random.seed(seed)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class BM25EnsembleRetrieval:
    """
    Ensemble retrieval using BM25-wandb (BERT tokenizer) and BM25-morphs (Kiwi tokenizer).
    Supports Weighted Sum and Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(
        self,
        args,
        data_path: Optional[str] = "data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        
        # Initialize sub-retrievers
        print("Initializing BM25-wandb...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
        self.retriever_wandb = BM25RetrievalWandb(
            tokenize_fn=tokenizer.tokenize,
            data_path=data_path,
            context_path=context_path
        )
        self.retriever_wandb.get_sparse_embedding()
        
        print("Initializing BM25-morphs...")
        kiwi = Kiwi()
        def kiwi_tokenize(text):
            return [token.form for token in kiwi.tokenize(text)]
            
        self.retriever_morphs = BM25RetrievalMorphs(
            tokenize_fn=kiwi_tokenize,
            data_path=data_path,
            context_path=context_path
        )
        self.retriever_morphs.get_sparse_embedding()
        
        self.retrieval_metrics = {}

    def retrieve(
        self, 
        query_or_dataset: Union[str, Dataset], 
        topk: Optional[int] = 100,
        ensemble_method: str = "weighted_sum", # 'weighted_sum' or 'rrf'
        alpha: float = 0.7 # Weight for wandb (0.0 ~ 1.0)
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        
        if isinstance(query_or_dataset, str):
            # Single query mode
            query = query_or_dataset
            
            # Get results from both retrievers
            # Note: We need to get more than topk to ensure overlap for RRF/Weighted Sum
            # But for simplicity and speed, we'll fetch topk*2 or just topk if k is large
            search_k = topk * 2 
            
            scores_wandb, indices_wandb = self.retriever_wandb.get_relevant_doc(query, k=search_k)
            scores_morphs, indices_morphs = self.retriever_morphs.get_relevant_doc(query, k=search_k)
            
            final_scores, final_indices = self._ensemble_single(
                scores_wandb, indices_wandb, 
                scores_morphs, indices_morphs, 
                topk, ensemble_method, alpha
            )
            
            print("[Search query]\n", query, "\n")
            top_passages = []
            for idx in range(topk):
                print(f"Top-{idx+1} passage with score {final_scores[idx]:4f}")
                passage = self.contexts[final_indices[idx]]
                print(passage)
                top_passages.append(passage)
            return (final_scores, top_passages)

        elif isinstance(query_or_dataset, Dataset):
            # Bulk query mode
            total = []
            correct_count = 0
            has_ground_truth = False
            
            queries = query_or_dataset["question"]
            search_k = topk * 2 # Fetch more candidates for better ensemble
            
            with timer("Bulk search wandb"):
                scores_wandb_list, indices_wandb_list = self.retriever_wandb.get_relevant_doc_bulk(queries, k=search_k)
            
            with timer("Bulk search morphs"):
                scores_morphs_list, indices_morphs_list = self.retriever_morphs.get_relevant_doc_bulk(queries, k=search_k)
                
            print(f"Ensembling with method: {ensemble_method}, alpha: {alpha}")
            
            doc_scores = []
            doc_indices = []
            
            for i in tqdm(range(len(queries)), desc="Ensembling"):
                s_wandb = scores_wandb_list[i]
                i_wandb = indices_wandb_list[i]
                s_morphs = scores_morphs_list[i]
                i_morphs = indices_morphs_list[i]
                
                f_scores, f_indices = self._ensemble_single(
                    s_wandb, i_wandb, 
                    s_morphs, i_morphs, 
                    topk, ensemble_method, alpha
                )
                doc_scores.append(f_scores)
                doc_indices.append(f_indices)

            # Construct result DataFrame
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Processing results")):
                retrieved_context = " ".join(
                    [self.contexts[pid] for pid in doc_indices[idx]]
                )
                
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": retrieved_context,
                }
                
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                    has_ground_truth = True
                    
                    original_context = example["context"]
                    retrieved_contexts_list = [self.contexts[pid] for pid in doc_indices[idx]]
                    
                    is_correct = any(
                        original_context in rc or rc in original_context 
                        for rc in retrieved_contexts_list
                    )
                    if is_correct:
                        correct_count += 1
                        
                total.append(tmp)

            cqas = pd.DataFrame(total)
            
            if has_ground_truth:
                retrieval_accuracy = correct_count / len(query_or_dataset)
                
                # Calculate MRR
                mrr_sum = 0.0
                for idx, example in enumerate(query_or_dataset):
                    original_context = example["context"]
                    retrieved_contexts_list = [self.contexts[pid] for pid in doc_indices[idx]]
                    
                    rank = 0
                    for i, rc in enumerate(retrieved_contexts_list):
                        if original_context in rc or rc in original_context:
                            rank = i + 1
                            break
                    
                    if rank > 0:
                        mrr_sum += 1.0 / rank
                
                mrr = mrr_sum / len(query_or_dataset)

                self.retrieval_metrics = {
                    "retrieval_accuracy": retrieval_accuracy,
                    "mrr": mrr,
                    "correct_count": correct_count,
                    "total_count": len(query_or_dataset),
                    "top_k": topk,
                    "ensemble_method": ensemble_method,
                    "alpha": alpha
                }
                print(f"\n{'='*50}")
                print(f"Ensemble Retrieval Accuracy @ {topk}: {retrieval_accuracy:.4f} ({correct_count}/{len(query_or_dataset)})")
                print(f"Ensemble MRR @ {topk}: {mrr:.4f}")
                print(f"{'='*50}\n")
            
            return cqas

    def _ensemble_single(
        self, 
        scores1, indices1, 
        scores2, indices2, 
        topk, method, alpha
    ):
        # Map doc_id to score/rank
        combined_scores = {}
        
        if method == "weighted_sum":
            # Normalize scores (Min-Max)
            def normalize(scores):
                if not scores: return []
                min_s = min(scores)
                max_s = max(scores)
                if max_s == min_s: return [1.0] * len(scores)
                return [(s - min_s) / (max_s - min_s) for s in scores]
            
            norm_scores1 = normalize(scores1)
            norm_scores2 = normalize(scores2)
            
            # Add weighted scores
            for idx, doc_id in enumerate(indices1):
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + alpha * norm_scores1[idx]
                
            for idx, doc_id in enumerate(indices2):
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 - alpha) * norm_scores2[idx]
                
        elif method == "rrf":
            k = 60 # RRF constant
            for rank, doc_id in enumerate(indices1):
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 1 / (k + rank + 1)
                
            for rank, doc_id in enumerate(indices2):
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        # Sort by score descending
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_docs = sorted_docs[:topk]
        
        final_indices = [doc_id for doc_id, score in top_k_docs]
        final_scores = [score for doc_id, score in top_k_docs]
        
        return final_scores, final_indices

    def get_retrieval_metrics(self) -> Dict:
        return self.retrieval_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", metavar="./data/train_dataset", type=str, default="../data/train_dataset")
    parser.add_argument("--model_name_or_path", metavar="bert-base-multilingual-cased", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--data_path", metavar="./data", type=str, default="../data")
    parser.add_argument("--context_path", metavar="wikipedia_documents", type=str, default="wikipedia_documents.json")
    parser.add_argument("--ensemble_method", type=str, default="weighted_sum", choices=["weighted_sum", "rrf"])
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for BM25-wandb (0.0-1.0)")
    
    args = parser.parse_args()

    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    retriever = BM25EnsembleRetrieval(
        args=args,
        data_path=args.data_path,
        context_path=args.context_path,
    )
    
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    with timer("bulk query by ensemble"):
        df = retriever.retrieve(full_ds, ensemble_method=args.ensemble_method, alpha=args.alpha)
        metrics = retriever.get_retrieval_metrics()
        print(f"Retrieval Metrics: {metrics}")

    with timer("single query by ensemble"):
        scores, indices = retriever.retrieve(query, ensemble_method=args.ensemble_method, alpha=args.alpha)
