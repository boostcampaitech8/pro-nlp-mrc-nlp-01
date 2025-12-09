"""
Hybrid Retrieval using BM25 + KURE (Dense) + Cross-encoder Reranker
"""
import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union, Dict
from tqdm.auto import tqdm
from datasets import Dataset

from .retrieval_bm25_wandb import BM25RetrievalWithMetrics
from .retrieval_kure import KURERetrieval
from .reranker import CrossEncoderReranker


class HybridKURERetrieval:
    """
    BM25 + KURE + Cross-encoder Reranker를 조합한 Hybrid Retrieval.
    
    Pipeline:
        Query → BM25 + KURE → Hybrid Score Fusion → Reranker → Top-K → Reader
    """
    
    def __init__(
        self, 
        tokenizer,  # For BM25
        data_path: str = "./data",
        context_path: str = "wikipedia_documents.json",
        kure_model_path: str = "nlpai-lab/KURE-v1",
        use_reranker: bool = False,
        reranker_model: str = "upskyy/ko-reranker",
    ):
        self.data_path = data_path
        self.use_reranker = use_reranker
        
        # Initialize BM25 Retriever
        print("=" * 50)
        print("Initializing BM25 Retriever...")
        self.bm25_retriever = BM25RetrievalWithMetrics(
            tokenize_fn=tokenizer.tokenize,
            data_path=data_path,
            context_path=context_path
        )
        self.bm25_retriever.get_sparse_embedding()
        
        # Initialize KURE Retriever
        print("=" * 50)
        print(f"Initializing KURE Retriever: {kure_model_path}")
        self.kure_retriever = KURERetrieval(
            data_path=data_path,
            context_path=context_path,
            model_name=kure_model_path
        )
        self.kure_retriever.get_dense_embedding()
        self.kure_retriever.build_faiss()
        
        # Initialize Cross-encoder Reranker (optional)
        self.reranker = None
        if use_reranker:
            print("=" * 50)
            print(f"Initializing Cross-encoder Reranker: {reranker_model}")
            self.reranker = CrossEncoderReranker(
                model_name=reranker_model,
                cache_dir="/data/ephemeral/models/reranker",
            )
        
        self.contexts = self.bm25_retriever.contexts
        self.ids = self.bm25_retriever.ids
        print("=" * 50)
        print("Hybrid KURE Retrieval initialized!")
        
    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int = 100,
        alpha: float = 0.5,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Hybrid Retrieval을 수행합니다.
        
        Args:
            query_or_dataset: 단일 쿼리 또는 Dataset
            topk: 최종 반환할 문서 수
            alpha: BM25 가중치 (0~1). alpha=0.5면 BM25와 KURE 동일 비중.
            
        Returns:
            pd.DataFrame: 검색 결과
            Dict: 평가 메트릭 (ground truth가 있는 경우)
        """
        search_k = topk * 3  # 더 많은 후보에서 선택
        
        if isinstance(query_or_dataset, str):
            queries = [query_or_dataset]
            is_single = True
        else:
            queries = query_or_dataset["question"]
            is_single = False
            
        print(f"\n[Hybrid Retrieval] alpha={alpha} (BM25 weight), topk={topk}")
        
        # BM25 Retrieval
        print("Retrieving with BM25...")
        bm25_scores_list, bm25_indices_list = self.bm25_retriever.get_relevant_doc_bulk(queries, k=search_k)
        
        # KURE Retrieval
        print("Retrieving with KURE...")
        kure_scores_list, kure_indices_list = self.kure_retriever.get_relevant_doc_bulk(queries, k=search_k)
        
        total = []
        # Rerank 전 메트릭
        pre_rerank_correct = 0
        pre_rerank_mrr = 0.0
        # KURE 단독 메트릭
        kure_correct = 0
        kure_mrr = 0.0
        # Rerank 후 메트릭
        post_rerank_correct = 0
        post_rerank_mrr = 0.0
        
        has_ground_truth = not is_single and "context" in query_or_dataset.features
        
        for i, query in tqdm(enumerate(queries), total=len(queries), desc="Merging scores"):
            # BM25 점수 정규화 (Min-Max)
            b_scores = bm25_scores_list[i]
            b_indices = bm25_indices_list[i]
            
            if len(b_scores) > 0:
                b_min, b_max = min(b_scores), max(b_scores)
                if b_max == b_min: 
                    b_norm = [1.0] * len(b_scores)
                else: 
                    b_norm = [(s - b_min) / (b_max - b_min) for s in b_scores]
            else:
                b_norm = []
                
            b_map = {idx: score for idx, score in zip(b_indices, b_norm)}
            
            # KURE 점수 정규화 (Min-Max)
            k_scores = kure_scores_list[i]
            k_indices = kure_indices_list[i]
            
            if len(k_scores) > 0:
                k_min, k_max = min(k_scores), max(k_scores)
                if k_max == k_min: 
                    k_norm = [1.0] * len(k_scores)
                else: 
                    k_norm = [(s - k_min) / (k_max - k_min) for s in k_scores]
            else:
                k_norm = []
                
            k_map = {idx: score for idx, score in zip(k_indices, k_norm)}
            
            # Hybrid Score 계산
            all_indices = set(b_indices) | set(k_indices)
            
            hybrid_scores = []
            for idx in all_indices:
                s_bm25 = b_map.get(idx, 0.0)
                s_kure = k_map.get(idx, 0.0)
                final_score = alpha * s_bm25 + (1 - alpha) * s_kure
                hybrid_scores.append((idx, final_score))
                
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Hybrid 결과 (Rerank 전)
            pre_rerank_indices = [x[0] for x in hybrid_scores[:topk]]

            # KURE 단독 결과 (Rerank 전)
            kure_scores_with_idx = []
            for j, k_idx in enumerate(k_indices):
                score = k_scores[j] if j < len(k_scores) else 0.0
                kure_scores_with_idx.append((k_idx, score))
            
            kure_scores_with_idx.sort(key=lambda x: x[1], reverse=True)
            kure_top_indices = [x[0] for x in kure_scores_with_idx[:topk]]

            # Rerank 전 메트릭 계산 (Hybrid & KURE)
            if has_ground_truth:
                example = query_or_dataset[i]
                original_context = example["context"]
                
                # Hybrid
                pre_rerank_contexts = [self.contexts[pid] for pid in pre_rerank_indices]
                if any(original_context in rc or rc in original_context for rc in pre_rerank_contexts):
                    pre_rerank_correct += 1
                for rank, rc in enumerate(pre_rerank_contexts):
                    if original_context in rc or rc in original_context:
                        pre_rerank_mrr += 1.0 / (rank + 1)
                        break
                
                # KURE Only
                kure_contexts_list = [self.contexts[pid] for pid in kure_top_indices]
                if any(original_context in rc or rc in original_context for rc in kure_contexts_list):
                    kure_correct += 1
                for rank, rc in enumerate(kure_contexts_list):
                    if original_context in rc or rc in original_context:
                        kure_mrr += 1.0 / (rank + 1)
                        break
            
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
                top_indices = pre_rerank_indices
                top_scores = [x[1] for x in hybrid_scores[:topk]]
            
            retrieved_context = " ".join([self.contexts[pid] for pid in top_indices])
            
            tmp = {
                "question": query,
                "id": query_or_dataset[i]["id"] if not is_single else "0",
                "context": retrieved_context,
            }
            
            # Rerank 후 메트릭 계산
            if has_ground_truth:
                example = query_or_dataset[i]
                original_context = example["context"]
                tmp["original_context"] = original_context
                
                retrieved_contexts_list = [self.contexts[pid] for pid in top_indices]
                
                # Accuracy (정답 포함 여부)
                if any(original_context in rc or rc in original_context for rc in retrieved_contexts_list):
                    post_rerank_correct += 1
                    
                # MRR
                for rank, rc in enumerate(retrieved_contexts_list):
                    if original_context in rc or rc in original_context:
                        post_rerank_mrr += 1.0 / (rank + 1)
                        break
            
            total.append(tmp)
        
        # 최종 메트릭 계산
        metrics = {}
        if not is_single and has_ground_truth:
            n = len(queries)
            
            # KURE Only 메트릭
            kure_acc = kure_correct / n
            kure_mrr_val = kure_mrr / n

            # Rerank 전 메트릭
            pre_acc = pre_rerank_correct / n
            pre_mrr = pre_rerank_mrr / n
            
            # Rerank 후 메트릭
            post_acc = post_rerank_correct / n
            post_mrr = post_rerank_mrr / n
            
            metrics = {
                "kure_accuracy": kure_acc,
                "kure_mrr": kure_mrr_val,
                "pre_rerank_accuracy": pre_acc,
                "pre_rerank_mrr": pre_mrr,
                "post_rerank_accuracy": post_acc,
                "post_rerank_mrr": post_mrr,
                "accuracy_improvement": post_acc - pre_acc,
                "mrr_improvement": post_mrr - pre_mrr,
                "correct_count": pre_rerank_correct,
                "total_count": n,
                "top_k": topk,
                "alpha": alpha,
                "use_reranker": self.use_reranker,
            }
            
            print(f"\n{'='*50}")
            print(f"[KURE Only]     Accuracy: {kure_acc:.4f}, MRR: {kure_mrr_val:.4f}")
            print(f"[Before Rerank] Accuracy: {pre_acc:.4f}, MRR: {pre_mrr:.4f}")
            if self.use_reranker:
                print(f"[After Rerank]  Accuracy: {post_acc:.4f}, MRR: {post_mrr:.4f}")
                print(f"[Improvement]   Accuracy: {post_acc - pre_acc:+.4f}, MRR: {post_mrr - pre_mrr:+.4f}")
            print(f"{'='*50}")
            
        return pd.DataFrame(total), metrics

