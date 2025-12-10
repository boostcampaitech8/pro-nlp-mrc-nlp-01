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



# Global variables for multiprocessing to share large data (contexts)
global_contexts = None
global_bm25_retriever = None

def ensemble_wrapper(args):
    """Wrapper for BM25 ensemble step"""
    w_s, w_i, m_s, m_i, search_k, method, alpha = args
    if global_bm25_retriever is None:
        # Fallback if global not set (should not happen if properly initialized)
        return [], []
    
    # We access the _ensemble_single method. It's static-like but defined as instance method.
    # It doesn't use self except maybe for type hinting, let's check.
    # It does not use self.
    return global_bm25_retriever._ensemble_single(
        w_s, w_i,
        m_s, m_i,
        search_k, method, alpha
    )

def fusion_wrapper(args):
    """Wrapper for Hybrid Fusion & Candidate Selection step"""
    (
        b_scores, b_indices,
        k_scores, k_indices,
        alpha, topk, use_reranker,
        query, query_idx, original_context, answers,
        fusion_method # Added fusion_method
    ) = args
    
    global global_contexts
    
    # buffers to return
    local_rerank_pairs = []
    local_rerank_meta = []
    
    # BM25 Normalization
    if len(b_scores) > 0:
        b_min, b_max = min(b_scores), max(b_scores)
        b_norm = [1.0] * len(b_scores) if b_max == b_min else [(s - b_min) / (b_max - b_min) for s in b_scores]
    else:
        b_norm = []
    b_map = {idx: score for idx, score in zip(b_indices, b_norm)}
    
    # KURE Normalization
    if len(k_scores) > 0:
        k_min, k_max = min(k_scores), max(k_scores)
        k_norm = [1.0] * len(k_scores) if k_max == k_min else [(s - k_min) / (k_max - k_min) for s in k_scores]
    else:
        k_norm = []
    k_map = {idx: score for idx, score in zip(k_indices, k_norm)}
    
    # Hybrid Fusion
    all_indices = set(b_indices) | set(k_indices)
    hybrid_scores = []
    
    if fusion_method == "rrf":
        # Reciprocal Rank Fusion
        # k constant for RRF, usually 60
        rrf_k = 60
        
        # Create rank maps
        b_rank_map = {idx: i for i, idx in enumerate(b_indices)}
        k_rank_map = {idx: i for i, idx in enumerate(k_indices)}
        
        for idx in all_indices:
            # If not in list, rank is effectively infinite (score contribution 0)
            # Standard RRF implementation: sum(1 / (k + rank))
            score = 0.0
            if idx in b_rank_map:
                score += 1.0 / (rrf_k + b_rank_map[idx] + 1)
            if idx in k_rank_map:
                score += 1.0 / (rrf_k + k_rank_map[idx] + 1)
            
            hybrid_scores.append((idx, score))
            
    else: # "weighted_sum"
        for idx in all_indices:
            s_bm25 = b_map.get(idx, 0.0)
            s_kure = k_map.get(idx, 0.0)
            final_score = alpha * s_bm25 + (1 - alpha) * s_kure
            hybrid_scores.append((idx, final_score))
            
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Pre-rerank Result (Top-K)
    pre_rerank_indices = [x[0] for x in hybrid_scores[:topk]]
    pre_rerank_scores = [x[1] for x in hybrid_scores[:topk]]
    
    # Reranking Candidate Collection
    if use_reranker:
        rerank_candidate_k = max(300, topk * 3)
        current_candidates = hybrid_scores[:rerank_candidate_k] # [(idx, score), ...]
        
        # Add to local buffer
        for idx, _ in current_candidates:
            passage_text = global_contexts[idx]
            local_rerank_pairs.append([query, passage_text])
            local_rerank_meta.append({'query_idx': query_idx, 'passage_id': idx})
            
    # Metrics partial calculation can be done here or in main loop.
    # To keep it simple and clean, we return necessary data for metric calc.
    # But returning large context strings is expensive.
    # We will return indices and let main loop do metric calc to avoid pickling overhead issues with strings if possible.
    # But checking "original_context in context" requires text access.
    # Since we have global_contexts here, we can compute correctness here!
    
    metrics_data = {
        'pre_rerank_ids': pre_rerank_indices,
        'kure_ids': [], # Fill below
        'pool_ids': [x[0] for x in hybrid_scores[:max(300, topk * 3)]] if use_reranker else []
    }
    
    # KURE Only Top-K
    kure_scores_with_idx = []
    for j, k_idx in enumerate(k_indices):
        score = k_scores[j] if j < len(k_scores) else 0.0
        kure_scores_with_idx.append((k_idx, score))
    kure_scores_with_idx.sort(key=lambda x: x[1], reverse=True)
    metrics_data['kure_ids'] = [x[0] for x in kure_scores_with_idx[:topk]]
    
    # To compute metrics, we need text check.
    # We return boolean results to save space.
    # is_correct check:
    
    res_metrics = {
        'kure_correct': False,
        'kure_rank': 0, # 0 means not found
        'pre_correct': False,
        'pre_rank': 0,
        'pool_correct': False
    }
    
    if original_context:
        # KURE
        kure_contexts = [global_contexts[pid] for pid in metrics_data['kure_ids']]
        if any(original_context in rc or rc in original_context for rc in kure_contexts):
            res_metrics['kure_correct'] = True
        for rank, rc in enumerate(kure_contexts):
            if original_context in rc or rc in original_context:
                res_metrics['kure_rank'] = rank + 1
                break
                
        # Pre-rerank
        pre_contexts = [global_contexts[pid] for pid in metrics_data['pre_rerank_ids']]
        if any(original_context in rc or rc in original_context for rc in pre_contexts):
            res_metrics['pre_correct'] = True
        for rank, rc in enumerate(pre_contexts):
            if original_context in rc or rc in original_context:
                res_metrics['pre_rank'] = rank + 1
                break
                
        # Candidate Pool
        if use_reranker:
            pool_contexts = [global_contexts[pid] for pid in metrics_data['pool_ids']]
            if any(original_context in rc or rc in original_context for rc in pool_contexts):
                res_metrics['pool_correct'] = True
                
    result_struct = {
        'id': None, # Filled by main
        'query': query,
        'pre_rerank_indices': pre_rerank_indices,
        'pre_rerank_scores': pre_rerank_scores,
        'original_context': original_context,
        'local_rerank_pairs': local_rerank_pairs,
        'local_rerank_meta': local_rerank_meta,
        'metrics': res_metrics,
        'answers': answers # Return answers
    }
    return result_struct

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
        alpha: float = 0.5, # Hybrid Fusion Weight (BM25 vs KURE) - Used for weighted_sum
        fusion_method: str = "weighted_sum" # 'weighted_sum' or 'rrf'
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
            
        print(f"\n[Hybrid Retrieval] Method={fusion_method}, alpha={alpha} (if applicable), topk={topk}")
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
        # Implementing Weighted Sum here using multiprocessing
        print("  - Ensembling (Multiprocessing)...")
        
        # Set globals
        global global_bm25_retriever
        global_bm25_retriever = self.bm25_retriever
        
        from multiprocessing import Pool, cpu_count
        
        ensemble_args = []
        for i in range(len(queries)):
             ensemble_args.append((
                 w_scores[i], w_indices[i],
                 m_scores[i], m_indices[i],
                 search_k, self.ensemble_method, self.bm25_alpha
             ))
             
        with Pool(processes=cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(ensemble_wrapper, ensemble_args),
                    total=len(queries),
                    desc="Ensembling BM25"
                )
            )
            
        bm25_scores_list = [r[0] for r in results]
        bm25_indices_list = [r[1] for r in results]
        
        global_bm25_retriever = None # Clear global ref
        
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
        
        # Phase 1: Hybrid Fusion & Candidate Selection (Multiprocessing)
        print("Hybrid Fusion & Selection (Multiprocessing)...")
        
        # Set globals
        global global_contexts
        global_contexts = self.contexts
        
        fusion_args = []
        for i in range(len(queries)):
            original_context = query_or_dataset[i]["context"] if has_ground_truth else None
            answers = query_or_dataset[i]["answers"] if has_ground_truth else None # Capture answers
            fusion_args.append((
                bm25_scores_list[i], bm25_indices_list[i],
                kure_scores_list[i], kure_indices_list[i],
                alpha, topk, self.use_reranker,
                queries[i], i, original_context, answers,
                fusion_method
            ))
            
        with Pool(processes=cpu_count()) as pool:
            fusion_results = list(
                tqdm(
                    pool.imap(fusion_wrapper, fusion_args),
                    total=len(queries),
                    desc="Hybrid Fusion"
                )
            )
            
        global_contexts = None # Clear global
        
        # Aggregate Results
        for i, res in enumerate(fusion_results):
            # Aggregating Metrics
            m = res['metrics']
            if m['kure_correct']: kure_correct += 1
            if m['kure_rank'] > 0: kure_mrr += 1.0 / m['kure_rank']
            
            if m['pre_correct']: pre_rerank_correct += 1
            if m['pre_rank'] > 0: pre_rerank_mrr += 1.0 / m['pre_rank']
            
            if m['pool_correct']: candidate_pool_correct += 1
            
            # Aggregating Buffers
            if res['local_rerank_pairs']:
                rerank_candidates_buffer.extend(res['local_rerank_pairs'])
                rerank_metadata_buffer.extend(res['local_rerank_meta'])
                candidate_pool_size_sum += len(res['local_rerank_pairs'])
                
            # Store buffer
            res['id'] = query_or_dataset[i]["id"] if not is_single else "0"
            hybrid_results_buffer[i] = res

        # Phase 2: Bulk Reranking
        reranked_scores_map = {} # query_idx -> [(passage_id, score), ...]
        
        if self.use_reranker and self.reranker is not None and rerank_candidates_buffer:
            print(f"\nReranking {len(rerank_candidates_buffer)} pairs in bulk (Optimized)...")
            # This is the key optimization: one massive batch inference instead of many
            flat_scores = self.reranker.score_pairs_bulk(
                rerank_candidates_buffer, 
                batch_size=128,  # Optimized batch size
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
            
            if res.get('answers'):
                tmp['answers'] = res['answers']
                
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
                "fusion_method": fusion_method,
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
