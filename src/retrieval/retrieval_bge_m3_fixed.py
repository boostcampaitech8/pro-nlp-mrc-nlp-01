import json
import os
import time
import numpy as np
import pandas as pd
import torch
from contextlib import contextmanager
from typing import List, Union, Optional, Tuple, Dict
from datasets import Dataset
from tqdm.auto import tqdm
from datasets import disable_progress_bar

disable_progress_bar()


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BGEM3RetrievalOptimized:
    """
    메모리 최적화된 BGE-M3 Hybrid Retrieval + Re-ranker
    """
    
    def __init__(
        self,
        data_path: str = "data",
        context_path: str = "wikipedia_documents.json",
        model_name: str = "BAAI/bge-m3",
        reranker_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        batch_size: int = 8,
        max_length: int = 512,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_colbert: bool = False,
        use_reranker: bool = True,
        max_memory_gb: float = 28.0,
    ):
        self.data_path = data_path
        self.base_batch_size = batch_size
        self.max_length = max_length
        self.use_dense = use_dense
        self.use_sparse = use_sparse
        self.use_colbert = use_colbert
        self.use_reranker = use_reranker
        self.max_memory_gb = max_memory_gb

        # Wikipedia 문서 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        wiki_items = list(wiki.items())
        self.ids = [k for k, v in wiki_items]
        self.contexts = [v["text"] for k, v in wiki_items]

        print(f"Number of passages: {len(self.contexts)}")

        # BGE-M3 모델 로드
        print(f"Loading BGE-M3 model: {model_name}")
        from FlagEmbedding import BGEM3FlagModel
        
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            device='cuda'
        )
        print("✅ BGE-M3 model loaded")

        # Re-ranker 로드
        self.reranker = None
        if use_reranker:
            print(f"Loading Re-ranker: {reranker_name}")
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(
                reranker_name,
                use_fp16=use_fp16,
                device='cuda'
            )
            print("✅ Re-ranker loaded")

        # Embeddings
        self.dense_embeddings = None
        self.sparse_embeddings = None
        self.colbert_embeddings = None

    def _get_adaptive_batch_size(self, texts: List[str]) -> int:
        """텍스트 길이에 따라 동적으로 배치 크기 조정"""
        avg_length = sum(len(t) for t in texts) / len(texts)
        
        if avg_length < 128:
            return min(self.base_batch_size * 2, 16)
        elif avg_length < 256:
            return self.base_batch_size
        elif avg_length < 512:
            return max(self.base_batch_size // 2, 4)
        else:
            return max(self.base_batch_size // 4, 2)

    def _clear_memory(self):
        """메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_embeddings(self):
        """Passage embeddings 생성 또는 로드"""
        
        # 파일 경로
        dense_path = os.path.join(self.data_path, "bge_m3_dense_opt.npy")
        sparse_path = os.path.join(self.data_path, "bge_m3_sparse_opt.pkl")
        colbert_path = os.path.join(self.data_path, "bge_m3_colbert_opt.pkl")

        # 로드 시도
        if self._try_load_embeddings(dense_path, sparse_path, colbert_path):
            return

        # 새로 생성
        print("Building optimized BGE-M3 embeddings...")
        print(f"Config: dense={self.use_dense}, sparse={self.use_sparse}, colbert={self.use_colbert}")
        
        all_dense = []
        all_sparse = []
        all_colbert = []

        # 동적 배치 처리
        i = 0
        pbar = tqdm(total=len(self.contexts), desc="Encoding passages")
        
        with timer("Encoding passages"):
            while i < len(self.contexts):
                end_idx = min(i + self.base_batch_size, len(self.contexts))
                batch = self.contexts[i:end_idx]
                
                batch_size = self._get_adaptive_batch_size(batch)
                
                try:
                    # Encode
                    embeddings = self.model.encode(
                        batch,
                        batch_size=batch_size,
                        max_length=self.max_length,
                        return_dense=self.use_dense,
                        return_sparse=self.use_sparse,
                        return_colbert_vecs=self.use_colbert,
                    )

                    # Dense (float16으로 저장)
                    if self.use_dense:
                        dense_batch = embeddings['dense_vecs'].astype(np.float16)
                        all_dense.append(dense_batch)

                    # Sparse
                    if self.use_sparse:
                        all_sparse.extend(embeddings['lexical_weights'])

                    # ColBERT
                    if self.use_colbert:
                        colbert_batch = [
                            vec.astype(np.float16) for vec in embeddings['colbert_vecs']
                        ]
                        all_colbert.extend(colbert_batch)

                    i = end_idx
                    pbar.update(len(batch))
                    
                    # 주기적으로 메모리 정리
                    if i % (self.base_batch_size * 10) == 0:
                        self._clear_memory()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n⚠️ OOM at batch {i}, reducing batch size...")
                        self.base_batch_size = max(self.base_batch_size // 2, 1)
                        self._clear_memory()
                    else:
                        raise e

        pbar.close()

        # 저장
        if self.use_dense:
            self.dense_embeddings = np.vstack(all_dense)
            np.save(dense_path, self.dense_embeddings)
            print(f"✅ Dense embeddings saved: {self.dense_embeddings.shape}")

        if self.use_sparse:
            self.sparse_embeddings = all_sparse
            import pickle
            with open(sparse_path, 'wb') as f:
                pickle.dump(self.sparse_embeddings, f)
            print(f"✅ Sparse embeddings saved: {len(self.sparse_embeddings)} passages")

        if self.use_colbert:
            self.colbert_embeddings = all_colbert
            import pickle
            with open(colbert_path, 'wb') as f:
                pickle.dump(self.colbert_embeddings, f)
            print(f"✅ ColBERT embeddings saved: {len(self.colbert_embeddings)} passages")

        self._clear_memory()

    def _try_load_embeddings(self, dense_path, sparse_path, colbert_path):
        """기존 embeddings 로드 시도"""
        all_loaded = True

        if self.use_dense:
            if os.path.exists(dense_path):
                with timer("Loading dense embeddings"):
                    self.dense_embeddings = np.load(dense_path)
                    print(f"Loaded dense: {self.dense_embeddings.shape}")
            else:
                all_loaded = False

        if self.use_sparse:
            if os.path.exists(sparse_path):
                with timer("Loading sparse embeddings"):
                    import pickle
                    with open(sparse_path, 'rb') as f:
                        self.sparse_embeddings = pickle.load(f)
                    print(f"Loaded sparse: {len(self.sparse_embeddings)} passages")
            else:
                all_loaded = False

        if self.use_colbert:
            if os.path.exists(colbert_path):
                with timer("Loading colbert embeddings"):
                    import pickle
                    with open(colbert_path, 'rb') as f:
                        self.colbert_embeddings = pickle.load(f)
                    print(f"Loaded colbert: {len(self.colbert_embeddings)} passages")
            else:
                all_loaded = False

        return all_loaded

    def _encode_query(self, query: str):
        """쿼리 인코딩"""
        result = self.model.encode(
            [query],
            batch_size=1,
            max_length=self.max_length,
            return_dense=self.use_dense,
            return_sparse=self.use_sparse,
            return_colbert_vecs=self.use_colbert,
        )
        return result

    def _compute_dense_score(self, query_vec, passage_vecs):
        """Dense 스코어 계산"""
        query_vec = query_vec.astype(np.float32)
        passage_vecs = passage_vecs.astype(np.float32)
        scores = np.dot(passage_vecs, query_vec.T).squeeze()
        return scores

    def _compute_sparse_score(self, query_weights, passage_weights_list):
        """Sparse 스코어 계산"""
        scores = []
        for passage_weights in passage_weights_list:
            score = 0.0
            for token_id, q_weight in query_weights.items():
                if token_id in passage_weights:
                    score += q_weight * passage_weights[token_id]
            scores.append(score)
        return np.array(scores)

    def _compute_colbert_score(self, query_vecs, passage_vecs_list):
        """ColBERT 스코어 계산"""
        scores = []
        query_vecs = query_vecs.astype(np.float32)
        
        for passage_vecs in passage_vecs_list:
            passage_vecs = passage_vecs.astype(np.float32)
            similarity_matrix = np.dot(query_vecs, passage_vecs.T)
            max_sims = similarity_matrix.max(axis=1)
            score = max_sims.sum()
            scores.append(score)
        
        return np.array(scores)

    def get_relevant_doc(
        self, 
        query: str, 
        k: int = 1,
        weights: Optional[dict] = None,
        use_rerank: bool = None,
        rerank_top_k: int = 100,
    ) -> Tuple[List[float], List[int]]:
        """단일 쿼리 검색 with Re-ranking"""
        
        if use_rerank is None:
            use_rerank = self.use_reranker and self.reranker is not None

        if weights is None:
            num_methods = sum([self.use_dense, self.use_sparse, self.use_colbert])
            default_weight = 1.0 / num_methods if num_methods > 0 else 1.0
            weights = {
                'dense': default_weight if self.use_dense else 0.0,
                'sparse': default_weight if self.use_sparse else 0.0,
                'colbert': default_weight if self.use_colbert else 0.0,
            }

        # 쿼리 인코딩
        query_result = self._encode_query(query)

        # 각 방법별 스코어 계산
        final_scores = np.zeros(len(self.contexts))

        if self.use_dense and weights['dense'] > 0:
            query_dense = query_result['dense_vecs']
            dense_scores = self._compute_dense_score(query_dense, self.dense_embeddings)
            if dense_scores.max() > dense_scores.min():
                dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
            final_scores += weights['dense'] * dense_scores

        if self.use_sparse and weights['sparse'] > 0:
            query_sparse = query_result['lexical_weights'][0]
            sparse_scores = self._compute_sparse_score(query_sparse, self.sparse_embeddings)
            if sparse_scores.max() > sparse_scores.min():
                sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
            final_scores += weights['sparse'] * sparse_scores

        if self.use_colbert and weights['colbert'] > 0:
            query_colbert = query_result['colbert_vecs'][0]
            colbert_scores = self._compute_colbert_score(query_colbert, self.colbert_embeddings)
            if colbert_scores.max() > colbert_scores.min():
                colbert_scores = (colbert_scores - colbert_scores.min()) / (colbert_scores.max() - colbert_scores.min())
            final_scores += weights['colbert'] * colbert_scores

        # Stage 1: Initial retrieval
        initial_k = rerank_top_k if use_rerank else k
        top_indices = np.argsort(final_scores)[::-1][:initial_k]
        top_scores = final_scores[top_indices]

        # Stage 2: Re-ranking
        if use_rerank and len(top_indices) > k:
            candidates = [self.contexts[i] for i in top_indices]
            pairs = [[query, doc] for doc in candidates]
            
            rerank_batch_size = min(16, len(pairs))
            
            try:
                rerank_scores = self.reranker.compute_score(
                    pairs,
                    batch_size=rerank_batch_size,
                    max_length=512,
                )
                
                # Re-rank된 순서로 재정렬
                rerank_indices = np.argsort(rerank_scores)[::-1][:k]
                final_indices = top_indices[rerank_indices]
                final_scores_output = [rerank_scores[i] for i in rerank_indices]
                
                self._clear_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("⚠️ Re-ranking OOM, using initial scores")
                    self._clear_memory()
                    final_indices = top_indices[:k]
                    final_scores_output = top_scores[:k].tolist()
                else:
                    raise e
        else:
            final_indices = top_indices[:k]
            final_scores_output = top_scores[:k].tolist()

        return final_scores_output, final_indices.tolist()

    def get_relevant_doc_bulk(
        self, 
        queries: List[str], 
        k: int = 1,
        weights: Optional[dict] = None,
        use_rerank: bool = None,
        rerank_top_k: int = 100,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """배치 쿼리 검색"""
        doc_scores, doc_indices = [], []

        for i, query in enumerate(tqdm(queries, desc="Retrieving")):
            scores, indices = self.get_relevant_doc(
                query, k=k, weights=weights,
                use_rerank=use_rerank, rerank_top_k=rerank_top_k
            )
            doc_scores.append(scores)
            doc_indices.append(indices)
            
            # 주기적 메모리 정리
            if (i + 1) % 50 == 0:
                self._clear_memory()

        return doc_scores, doc_indices

    def retrieve(
        self, 
        query_or_dataset: Union[str, Dataset], 
        topk: int = 100,
        weights: Optional[dict] = None,
        use_rerank: bool = None,
        rerank_top_k: int = 100,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        검색 실행 with Re-ranking and metrics
        
        Returns:
            pd.DataFrame: 검색 결과
            Dict: Retrieval metrics (ground truth가 있는 경우)
        """
        # 단일 쿼리
        if isinstance(query_or_dataset, str):
            scores, indices = self.get_relevant_doc(
                query_or_dataset, k=topk, weights=weights,
                use_rerank=use_rerank, rerank_top_k=rerank_top_k
            )
            
            rows = [{
                "id": "0",
                "question": query_or_dataset,
                "context": self.contexts[idx],
                "retrieval_rank": rank,
                "retrieval_score": float(score),
            } for rank, (idx, score) in enumerate(zip(indices, scores))]
            
            return pd.DataFrame(rows), {}

        # Dataset 처리
        dataset = query_or_dataset
        queries = dataset["question"]

        with timer(f"Retrieving for {len(queries)} queries"):
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                queries, k=topk, weights=weights,
                use_rerank=use_rerank, rerank_top_k=rerank_top_k
            )

        # DataFrame 생성 및 metrics 계산
        rows = []
        has_ground_truth = "context" in dataset.features
        
        # Metrics 초기화
        correct_count = 0
        total_count = len(queries)
        mrr_sum = 0.0

        for i, example in enumerate(dataset):
            qid = example["id"]
            qtext = example["question"]
            
            # Ground truth context (있는 경우)
            original_context = example.get("context", None) if has_ground_truth else None

            # 이 쿼리에 대한 검색 결과
            found_answer = False
            
            for rank, (idx, score) in enumerate(zip(doc_indices[i], doc_scores[i])):
                retrieved_context = self.contexts[idx]
                
                row = {
                    "id": qid,
                    "question": qtext,
                    "context": retrieved_context,
                    "retrieval_rank": rank,
                    "retrieval_score": float(score),
                }

                # Ground truth 비교 (첫 번째 검색 결과에서만)
                if has_ground_truth and rank == 0:
                    row["original_context"] = original_context
                    
                # Accuracy 체크 (모든 rank 확인)
                if has_ground_truth and not found_answer and original_context:
                    if original_context in retrieved_context or retrieved_context in original_context:
                        if not found_answer:  # 첫 발견
                            correct_count += 1
                            mrr_sum += 1.0 / (rank + 1)
                            found_answer = True

                if "answers" in example:
                    row["answers"] = example["answers"]

                rows.append(row)

        # Metrics 계산
        metrics = {}
        if has_ground_truth:
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            mrr = mrr_sum / total_count if total_count > 0 else 0.0
            
            metrics = {
                "retrieval_accuracy": accuracy,
                "mrr": mrr,
                "correct_count": correct_count,
                "total_count": total_count,
                "top_k": topk,
            }
            
            print(f"\n{'='*50}")
            print(f"[Retrieval Metrics]")
            print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
            print(f"MRR: {mrr:.4f}")
            print(f"Top-K: {topk}")
            print(f"{'='*50}\n")

        return pd.DataFrame(rows), metrics

    def generate_hard_negatives(
        self,
        query: str,
        positive_doc_id: int,
        k: int = 10,
        weights: Optional[dict] = None,
    ) -> List[int]:
        """Hard negative sampling"""
        _, indices = self.get_relevant_doc(
            query, k=k+20, weights=weights, use_rerank=False
        )
        
        hard_negatives = [idx for idx in indices if idx != positive_doc_id][:k]
        return hard_negatives