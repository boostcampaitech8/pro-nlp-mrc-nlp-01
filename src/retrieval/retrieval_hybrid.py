import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union, Dict
from tqdm.auto import tqdm
from datasets import Dataset

# Import existing classes (assuming they are in the same package or accessible)
from .retrieval_bm25_wandb import BM25RetrievalWithMetrics
from .retrieval_dpr import DenseRetrieval
from .reranker import CrossEncoderReranker

class HybridRetrieval:
    def __init__(
        self, 
        args,
        tokenizer, # For BM25
        model_args, # For DPR configuration (retriever_name_or_path)
        data_path: str = "./data",
        context_path: str = "wikipedia_documents.json",
        use_reranker: bool = False,
        reranker_model: str = "upskyy/ko-reranker",
    ):
        self.data_path = data_path
        self.args = args
        self.use_reranker = use_reranker
        
        # Initialize BM25 Retriever
        print("Initializing BM25 Retriever...")
        self.bm25_retriever = BM25RetrievalWithMetrics(
            tokenize_fn=tokenizer.tokenize,
            data_path=data_path,
            context_path=context_path
        )
        self.bm25_retriever.get_sparse_embedding()
        
        # Initialize DPR Retriever
        print("Initializing DPR Retriever...")
        # DenseRetrieval.__init__(args, dataset, model_name_or_path, data_path, context_path)
        retriever_path = model_args.retriever_name_or_path if hasattr(model_args, 'retriever_name_or_path') else model_args.model_name_or_path
        print(f"Using DPR Retriever from {retriever_path}")
        self.dpr_retriever = DenseRetrieval(
            args=args,
            dataset=None,
            model_name_or_path=retriever_path,
            data_path=data_path,
            context_path=context_path
        )
        self.dpr_retriever.get_dense_embedding()
        self.dpr_retriever.build_faiss()  # Needed to initialize self.indexer
        
        # Initialize Cross-encoder Reranker (optional)
        self.reranker = None
        if use_reranker:
            print(f"Initializing Cross-encoder Reranker: {reranker_model}")
            self.reranker = CrossEncoderReranker(
                model_name=reranker_model,
                cache_dir="/data/ephemeral/models/reranker",
            )
        
        self.contexts = self.bm25_retriever.contexts
        self.ids = self.bm25_retriever.ids
        
    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int = 100,
        alpha: float = 0.5,
    ) -> Tuple[pd.DataFrame, Dict]:
        
        search_k = topk * 3
        
        if isinstance(query_or_dataset, str):
            queries = [query_or_dataset]
            is_single = True
        else:
            queries = query_or_dataset["question"]
            is_single = False
            
        print(f"Hybrid Retrieval with alpha={alpha} (BM25 weight)")
        
        print("Retrieving with BM25...")
        bm25_scores_list, bm25_indices_list = self.bm25_retriever.get_relevant_doc_bulk(queries, k=search_k)
        
        print("Retrieving with DPR...")
        dpr_scores_list, dpr_indices_list = self.dpr_retriever.get_relevant_doc_bulk(queries, k=search_k)
        
        total = []
        correct_count = 0
        mrr_sum = 0.0
        has_ground_truth = not is_single and "context" in query_or_dataset.features
        
        for i, query in tqdm(enumerate(queries), total=len(queries), desc="Merging scores"):
            b_scores = bm25_scores_list[i]
            b_indices = bm25_indices_list[i]
            
            if len(b_scores) > 0:
                b_min, b_max = min(b_scores), max(b_scores)
                if b_max == b_min: b_norm = [1.0] * len(b_scores)
                else: b_norm = [(s - b_min) / (b_max - b_min) for s in b_scores]
            else:
                b_norm = []
                
            b_map = {idx: score for idx, score in zip(b_indices, b_norm)}
            
            d_scores = dpr_scores_list[i]
            d_indices = dpr_indices_list[i]
            
            if len(d_scores) > 0:
                d_min, d_max = min(d_scores), max(d_scores)
                if d_max == d_min: d_norm = [1.0] * len(d_scores)
                else: d_norm = [(s - d_min) / (d_max - d_min) for s in d_scores]
            else:
                d_norm = []
                
            d_map = {idx: score for idx, score in zip(d_indices, d_norm)}
            
            all_indices = set(b_indices) | set(d_indices)
            
            hybrid_scores = []
            for idx in all_indices:
                s_bm25 = b_map.get(idx, 0.0)
                s_dpr = d_map.get(idx, 0.0)
                final_score = alpha * s_bm25 + (1 - alpha) * s_dpr
                hybrid_scores.append((idx, final_score))
                
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Reranking 단계 (활성화된 경우)
            if self.use_reranker and self.reranker is not None:
                # Hybrid에서 topk * 3개 후보 선정
                rerank_candidates = hybrid_scores[:topk * 3]
                candidate_indices = [x[0] for x in rerank_candidates]
                candidate_passages = [self.contexts[idx] for idx in candidate_indices]
                
                # Cross-encoder로 재정렬
                reranked = self.reranker.rerank_with_indices(
                    query=query,
                    passages=candidate_passages,
                    original_indices=candidate_indices,
                    top_k=topk,
                )
                top_indices, top_scores = reranked
            else:
                # Reranker 없이 Hybrid 점수만 사용
                top_hybrid = hybrid_scores[:topk]
                top_indices = [x[0] for x in top_hybrid]
                top_scores = [x[1] for x in top_hybrid]
            
            retrieved_context = " ".join([self.contexts[pid] for pid in top_indices])
            
            tmp = {
                "question": query,
                "id": query_or_dataset[i]["id"] if not is_single else "0",
                "context": retrieved_context,
            }
            
            if has_ground_truth:
                example = query_or_dataset[i]
                original_context = example["context"]
                tmp["original_context"] = original_context
                
                retrieved_contexts_list = [self.contexts[pid] for pid in top_indices]
                
                if any(original_context in rc or rc in original_context for rc in retrieved_contexts_list):
                    correct_count += 1
                    
                for rank, rc in enumerate(retrieved_contexts_list):
                    if original_context in rc or rc in original_context:
                        mrr_sum += 1.0 / (rank + 1)
                        break
            
            total.append(tmp)
        
        metrics = {}
        if not is_single and has_ground_truth:
            acc = correct_count / len(queries)
            mrr = mrr_sum / len(queries)
            metrics = {
                "accuracy": acc,
                "mrr": mrr,
                "correct_count": correct_count,
                "total_count": len(queries),
                "top_k": topk,
                "alpha": alpha,
            }
            print(f"Hybrid Accuracy: {acc:.4f}, MRR: {mrr:.4f}")
            
        return pd.DataFrame(total), metrics
