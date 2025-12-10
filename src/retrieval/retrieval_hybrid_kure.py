"""
Hybrid Reieval using BM25 + KURE (Dense) + Cross-encoder Reranker
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

from .retrieval_bm25_ensemble import BM25EnsembleRetrieval
from .retrieval_kure import KURERetrieval
from .reranker import CrossEncoderReranker


class HybridKURERetrieval:
    """
    BM25(Ensemble) + KURE + Cross-encoder Reranker를 조합한 Hybrid Retrieval.
    
    Pipeline:
        Query → BM25(Ensemble) + KURE → Hybrid Score Fusion → Reranker → Top-K → Reader
    """
    
    def __init__(
        self, 
        tokenizer,  # For BM25-wandb (kept for compatibility, though ensemble init handles it)
        data_path: str = "./data",
        context_path: str = "wikipedia_documents.json",
        kure_model_path: str = "nlpai-lab/KURE-v1",
        use_reranker: bool = False,
        reranker_model: str = "upskyy/ko-reranker",
        # Ensemble arguments (defaults)
        ensemble_method: str = "weighted_sum",
        bm25_alpha: float = 0.7, 
    ):
        self.data_path = data_path
        self.use_reranker = use_reranker
        self.ensemble_method = ensemble_method
        self.bm25_alpha = bm25_alpha
        
        # Initialize BM25 Ensemble Retriever
        print("=" * 50)
        print("Initializing BM25 Ensemble Retriever...")
        
        # Mock args object for BM25EnsembleRetrieval compatibility
        class MockArgs:
            model_name_or_path = "bert-base-multilingual-cased" # Default or passed
        
        args = MockArgs()
        if hasattr(tokenizer, "name_or_path"): # Try to get from passed tokenizer
             args.model_name_or_path = tokenizer.name_or_path
             
        self.bm25_retriever = BM25EnsembleRetrieval(
            args=args,
            data_path=data_path,
            context_path=context_path
        )
        # Note: BM25EnsembleRetrieval inits embeddings inside __init__, no need to call get_sparse_embedding explicitly
        
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
        print("Hybrid KURE (with BM25 Ensemble) Retrieval initialized!")
        
    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int = 100,
        alpha: float = 0.5, # Hybrid Fusion Weight (BM25 vs KURE)
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Hybrid Retrieval을 수행합니다.
        
        Args:
            query_or_dataset: 단일 쿼리 또는 Dataset
            topk: 최종 반환할 문서 수
            alpha: BM25(Ensemble) 가중치 (0~1). alpha=0.5면 BM25와 KURE 동일 비중.
            
        Returns:
            pd.DataFrame: 검색 결과
            Dict: 평가 메트릭 (ground truth가 있는 경우)
        """
        if self.use_reranker:
            # Reranker 사용 시, 200개 pool을 채우기 위해 충분히 많이 가져옵니다.
            search_k = max(topk * 3, 600)
        else:
            search_k = topk * 3
        
        if isinstance(query_or_dataset, str):
            queries = [query_or_dataset]
            is_single = True
        else:
            queries = query_or_dataset["question"]
            is_single = False
            
        print(f"\n[Hybrid Retrieval] alpha={alpha} (BM25 vs KURE), topk={topk}")
        print(f"Using BM25 Ensemble: method={self.ensemble_method}, internal_alpha={self.bm25_alpha}")
        
        # BM25 Retrieval (Ensemble)
        print("Retrieving with BM25 Ensemble...")
        # Since BM25Ensemble.retrieve checks instance, we can't use bulk directly if we want consistent return.
        # But BM25EnsembleRetrieval does not expose get_relevant_doc_bulk directly in a simple way 
        # that returns scores/indices without DataFrame logic in 'retrieve'.
        # Actually it does have logic inside 'retrieve' but returns DataFrame.
        # We need raw scores/indices. 
        # Let's check BM25EnsembleRetrieval.retrieve implementation again.
        # It creates DataFrame. 
       
        # To get raw scores/indices, we'll manually call sub-retrievers and ensemble them here 
        # OR use a private method if available. 
        # Using the private _ensemble_single for each query in loop is slow? 
        # No, we should use the retrieve logic but extract scores? 
        # Wait, BM25EnsembleRetrieval code shows:
        # scores_wandb_list, indices_wandb_list = self.retriever_wandb.get_relevant_doc_bulk(queries, k=search_k)
        # scores_morphs_list, indices_morphs_list = self.retriever_morphs.get_relevant_doc_bulk(queries, k=search_k)
        # We can replicate this logic here effectively or call a helper.
        # Since I cannot easily modify ensemble file to add a new method, I will replicate bulk call here.
        
        # Call sub-retrievers directly for bulk efficiency
        search_k_bm25 = search_k * 2 # Fetch more for ensemble overlap
        
        # Wandb BM25
        print("  - BM25 Wandb...")
        w_scores, w_indices = self.bm25_retriever.retriever_wandb.get_relevant_doc_bulk(queries, k=search_k_bm25)
        
        # Morphs BM25
        print("  - BM25 Morphs...")
        m_scores, m_indices = self.bm25_retriever.retriever_morphs.get_relevant_doc_bulk(queries, k=search_k_bm25)
        
        bm25_scores_list = []
        bm25_indices_list = []
        
        # Ensemble logic (Weighted Sum only for simplicity/speed or use wrapper)
        # Implementing Weighted Sum here to match Ensemble Logic
        print("  - Ensembling...")
        for i in tqdm(range(len(queries)), desc="Ensembling BM25"):
            f_scores, f_indices = self.bm25_retriever._ensemble_single(
                w_scores[i], w_indices[i],
                m_scores[i], m_indices[i],
                search_k, # Our target search_k for Hybrid input
                self.ensemble_method,
                self.bm25_alpha
            )
            bm25_scores_list.append(f_scores)
            bm25_indices_list.append(f_indices)
        
        # KURE Retrieval
        print("Retrieving with KURE...")
        kure_scores_list, kure_indices_list = self.kure_retriever.get_relevant_doc_bulk(queries, k=search_k)
        
        # Metrics Initializations
        pre_rerank_correct = 0
        pre_rerank_mrr = 0.0
        kure_correct = 0
        kure_mrr = 0.0
        post_rerank_correct = 0
        post_rerank_mrr = 0.0
        candidate_pool_correct = 0
        candidate_pool_size_sum = 0
        
        has_ground_truth = not is_single and "context" in query_or_dataset.features
        
        # Buffers for Bulk Reranking
        rerank_candidates_buffer = []  # [[query, passage_text], ...]
        rerank_metadata_buffer = []    # [{'query_idx': int, 'passage_id': int}, ...]
        hybrid_results_buffer = [None] * len(queries)
        
        # Phase 1: Hybrid Fusion & Candidate Selection
        for i, query in tqdm(enumerate(queries), total=len(queries), desc="Hybrid Fusion & Selection"):
            # BM25 Normalization
            b_scores = bm25_scores_list[i]
            b_indices = bm25_indices_list[i]
            if len(b_scores) > 0:
                b_min, b_max = min(b_scores), max(b_scores)
                b_norm = [1.0] * len(b_scores) if b_max == b_min else [(s - b_min) / (b_max - b_min) for s in b_scores]
            else:
                b_norm = []
            b_map = {idx: score for idx, score in zip(b_indices, b_norm)}
            
            # KURE Normalization
            k_scores = kure_scores_list[i]
            k_indices = kure_indices_list[i]
            if len(k_scores) > 0:
                k_min, k_max = min(k_scores), max(k_scores)
                k_norm = [1.0] * len(k_scores) if k_max == k_min else [(s - k_min) / (k_max - k_min) for s in k_scores]
            else:
                k_norm = []
            k_map = {idx: score for idx, score in zip(k_indices, k_norm)}
            
            # Hybrid Fusion
            all_indices = set(b_indices) | set(k_indices)
            hybrid_scores = []
            for idx in all_indices:
                s_bm25 = b_map.get(idx, 0.0)
                s_kure = k_map.get(idx, 0.0)
                final_score = alpha * s_bm25 + (1 - alpha) * s_kure
                hybrid_scores.append((idx, final_score))
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Pre-rerank Result (Top-K)
            pre_rerank_indices = [x[0] for x in hybrid_scores[:topk]]
            
            # KURE Only Result (for metrics)
            kure_scores_with_idx = []
            for j, k_idx in enumerate(k_indices):
                score = k_scores[j] if j < len(k_scores) else 0.0
                kure_scores_with_idx.append((k_idx, score))
            kure_scores_with_idx.sort(key=lambda x: x[1], reverse=True)
            kure_top_indices = [x[0] for x in kure_scores_with_idx[:topk]]
            
            original_context = None
            if has_ground_truth:
                original_context = query_or_dataset[i]["context"]
                
                # Metrics: Hybrid (Pre-rerank)
                pre_rerank_contexts = [self.contexts[pid] for pid in pre_rerank_indices]
                if any(original_context in rc or rc in original_context for rc in pre_rerank_contexts):
                    pre_rerank_correct += 1
                for rank, rc in enumerate(pre_rerank_contexts):
                    if original_context in rc or rc in original_context:
                        pre_rerank_mrr += 1.0 / (rank + 1)
                        break
                        
                # Metrics: KURE Only
                kure_contexts = [self.contexts[pid] for pid in kure_top_indices]
                if any(original_context in rc or rc in original_context for rc in kure_contexts):
                    kure_correct += 1
                for rank, rc in enumerate(kure_contexts):
                    if original_context in rc or rc in original_context:
                        kure_mrr += 1.0 / (rank + 1)
                        break
            
            # Reranking Candidate Collection
            current_rerank_candidates = []
            if self.use_reranker and self.reranker is not None:
                rerank_candidate_k = max(200, topk * 3)
                current_candidates = hybrid_scores[:rerank_candidate_k] # [(idx, score), ...]
                candidate_pool_size_sum += len(current_candidates)
                
                # Check Candidate Pool Recall
                if has_ground_truth:
                    pool_contexts = [self.contexts[x[0]] for x in current_candidates]
                    if any(original_context in rc or rc in original_context for rc in pool_contexts):
                        candidate_pool_correct += 1
                
                # Add to bulk buffer
                for idx, _ in current_candidates:
                    passage_text = self.contexts[idx]
                    rerank_candidates_buffer.append([query, passage_text])
                    rerank_metadata_buffer.append({'query_idx': i, 'passage_id': idx})
            
            # Store data for Phase 3
            hybrid_results_buffer[i] = {
                'id': query_or_dataset[i]["id"] if not is_single else "0",
                'query': query,
                'pre_rerank_indices': pre_rerank_indices,
                'pre_rerank_scores': [x[1] for x in hybrid_scores[:topk]],
                'original_context': original_context
            }

        # Phase 2: Bulk Reranking
        reranked_scores_map = {} # query_idx -> [(passage_id, score), ...]
        
        if self.use_reranker and self.reranker is not None and rerank_candidates_buffer:
            print(f"\nReranking {len(rerank_candidates_buffer)} pairs in bulk (Optimized)...")
            # This is the key optimization: one massive batch inference instead of many
            flat_scores = self.reranker.score_pairs_bulk(
                rerank_candidates_buffer, 
                batch_size=256,  # Optimized batch size
                show_progress_bar=True
            )
            
            # Re-group scores
            for meta, score in zip(rerank_metadata_buffer, flat_scores):
                q_idx = meta['query_idx']
                p_id = meta['passage_id']
                if q_idx not in reranked_scores_map:
                    reranked_scores_map[q_idx] = []
                reranked_scores_map[q_idx].append((p_id, score))

        # Phase 3: Final Assembly & Rerank Metrics
        total = []
        for i in tqdm(range(len(queries)), desc="Finalizing Results"):
            res = hybrid_results_buffer[i]
            q_idx = i
            
            if self.use_reranker and q_idx in reranked_scores_map:
                # Use Reranked scores
                candidates = reranked_scores_map[q_idx]
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                top_indices = [x[0] for x in candidates[:topk]]
                # top_scores = [x[1] for x in candidates[:topk]]
            else:
                # Fallback to Pre-rerank indices
                top_indices = res['pre_rerank_indices']
                
            retrieved_context = " ".join([self.contexts[pid] for pid in top_indices])
            
            # Result Dict
            tmp = {
                "question": res['query'],
                "id": res['id'],
                "context": retrieved_context,
            }
            if res['original_context']:
                tmp['original_context'] = res['original_context']
                
                # Post-rerank Metrics
                retrieved_contexts_list = [self.contexts[pid] for pid in top_indices]
                original_context = res['original_context']
                
                if any(original_context in rc or rc in original_context for rc in retrieved_contexts_list):
                    post_rerank_correct += 1
                
                for rank, rc in enumerate(retrieved_contexts_list):
                    if original_context in rc or rc in original_context:
                        post_rerank_mrr += 1.0 / (rank + 1)
                        break
            
            total.append(tmp)

        # Final Metrics Calculation
        metrics = {}
        if not is_single and has_ground_truth:
            n = len(queries)
            
            kure_acc = kure_correct / n
            pre_acc = pre_rerank_correct / n
            pool_acc = candidate_pool_correct / n if self.use_reranker else 0.0
            post_acc = post_rerank_correct / n
            
            metrics = {
                "kure_accuracy": kure_acc,
                "kure_mrr": kure_mrr / n,
                "pre_rerank_accuracy": pre_acc,
                "pre_rerank_mrr": pre_rerank_mrr / n,
                "candidate_pool_accuracy": pool_acc,
                "candidate_pool_size": candidate_pool_size_sum / n if n > 0 else 0,
                "post_rerank_accuracy": post_acc,
                "post_rerank_mrr": post_rerank_mrr / n,
                "accuracy_improvement": post_acc - pre_acc,
                "mrr_improvement": (post_rerank_mrr / n) - (pre_rerank_mrr / n),
                "correct_count": post_rerank_correct, # Use post-rerank for consistency
                "total_count": n,
                "top_k": topk,
                "alpha": alpha,
                "use_reranker": self.use_reranker,
            }
            
            print(f"\n{'='*50}")
            print(f"[KURE Only]     Accuracy: {metrics['kure_accuracy']:.4f}, MRR: {metrics['kure_mrr']:.4f}")
            print(f"[Before Rerank] Accuracy: {metrics['pre_rerank_accuracy']:.4f}, MRR: {metrics['pre_rerank_mrr']:.4f}")
            if self.use_reranker:
                print(f"[Candidate Pool] Accuracy: {metrics['candidate_pool_accuracy']:.4f}")
                print(f"[After Rerank]  Accuracy: {metrics['post_rerank_accuracy']:.4f}, MRR: {metrics['post_rerank_mrr']:.4f}")
                print(f"[Improvement]   Accuracy: {metrics['accuracy_improvement']:+.4f}, MRR: {metrics['mrr_improvement']:+.4f}")
            print(f"{'='*50}")
            
        return pd.DataFrame(total), metrics
