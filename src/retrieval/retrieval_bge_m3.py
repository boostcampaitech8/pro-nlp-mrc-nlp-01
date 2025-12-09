import json
import os
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import List, Union, Optional
from datasets import Dataset
from tqdm.auto import tqdm


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BGEM3Retrieval:
    """
    BGE-M3 기반 Hybrid Retrieval
    - Dense retrieval (semantic similarity)
    - Sparse retrieval (learned term weights, SPLADE-like)
    - ColBERT (token-level interaction)
    
    한국어 완벽 지원!
    """
    
    def __init__(
        self,
        data_path: str = "data",
        context_path: str = "wikipedia_documents.json",
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        batch_size: int = 12,
        max_length: int = 512,  # 8192까지 가능하지만 메모리 고려
        use_dense: bool = True,
        use_sparse: bool = True,
        use_colbert: bool = False,  # 메모리 많이 씀
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_dense = use_dense
        self.use_sparse = use_sparse
        self.use_colbert = use_colbert

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
            device='cuda'  # 또는 'cpu'
        )
        print("✅ BGE-M3 model loaded successfully")

        # Embeddings
        self.dense_embeddings = None
        self.sparse_embeddings = None
        self.colbert_embeddings = None

    def get_embeddings(self):
        """Passage embeddings 생성 또는 로드"""
        
        # 파일 경로
        dense_path = os.path.join(self.data_path, "bge_m3_dense.npy")
        sparse_path = os.path.join(self.data_path, "bge_m3_sparse.npz")
        colbert_path = os.path.join(self.data_path, "bge_m3_colbert.npy")

        # 로드 시도
        loaded = self._try_load_embeddings(dense_path, sparse_path, colbert_path)
        if loaded:
            return

        # 새로 생성
        print("Building BGE-M3 embeddings...")
        print(f"Config: dense={self.use_dense}, sparse={self.use_sparse}, colbert={self.use_colbert}")
        
        all_dense = []
        all_sparse = []
        all_colbert = []

        # Batch 처리
        num_batches = (len(self.contexts) + self.batch_size - 1) // self.batch_size
        
        with timer("Encoding passages"):
            for i in tqdm(range(0, len(self.contexts), self.batch_size), total=num_batches):
                batch = self.contexts[i:i + self.batch_size]
                
                # Encode
                embeddings = self.model.encode(
                    batch,
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                    return_dense=self.use_dense,
                    return_sparse=self.use_sparse,
                    return_colbert_vecs=self.use_colbert,
                )

                # Dense
                if self.use_dense:
                    all_dense.append(embeddings['dense_vecs'])

                # Sparse
                if self.use_sparse:
                    # Sparse는 list of dicts 형태
                    all_sparse.extend(embeddings['lexical_weights'])

                # ColBERT
                if self.use_colbert:
                    all_colbert.extend(embeddings['colbert_vecs'])

        # 저장
        if self.use_dense:
            self.dense_embeddings = np.vstack(all_dense)
            np.save(dense_path, self.dense_embeddings)
            print(f"✅ Dense embeddings saved: {self.dense_embeddings.shape}")

        if self.use_sparse:
            self.sparse_embeddings = all_sparse
            # Sparse는 pickle로 저장
            import pickle
            with open(sparse_path, 'wb') as f:
                pickle.dump(self.sparse_embeddings, f)
            print(f"✅ Sparse embeddings saved: {len(self.sparse_embeddings)} passages")

        if self.use_colbert:
            self.colbert_embeddings = all_colbert
            # ColBERT는 가변 길이라 pickle 사용
            import pickle
            with open(colbert_path, 'wb') as f:
                pickle.dump(self.colbert_embeddings, f)
            print(f"✅ ColBERT embeddings saved: {len(self.colbert_embeddings)} passages")

    def _try_load_embeddings(self, dense_path, sparse_path, colbert_path):
        """기존 embeddings 로드 시도"""
        loaded_any = False

        if self.use_dense and os.path.exists(dense_path):
            with timer("Loading dense embeddings"):
                self.dense_embeddings = np.load(dense_path)
                print(f"Loaded dense: {self.dense_embeddings.shape}")
                loaded_any = True

        if self.use_sparse and os.path.exists(sparse_path):
            with timer("Loading sparse embeddings"):
                import pickle
                with open(sparse_path, 'rb') as f:
                    self.sparse_embeddings = pickle.load(f)
                print(f"Loaded sparse: {len(self.sparse_embeddings)} passages")
                loaded_any = True

        if self.use_colbert and os.path.exists(colbert_path):
            with timer("Loading colbert embeddings"):
                import pickle
                with open(colbert_path, 'rb') as f:
                    self.colbert_embeddings = pickle.load(f)
                print(f"Loaded colbert: {len(self.colbert_embeddings)} passages")
                loaded_any = True

        # 모든 필요한 embedding이 로드되었는지 확인
        all_loaded = True
        if self.use_dense and self.dense_embeddings is None:
            all_loaded = False
        if self.use_sparse and self.sparse_embeddings is None:
            all_loaded = False
        if self.use_colbert and self.colbert_embeddings is None:
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
        """Dense 스코어 계산 (cosine similarity)"""
        # query_vec: (1, dim)
        # passage_vecs: (num_passages, dim)
        scores = np.dot(passage_vecs, query_vec.T).squeeze()
        return scores

    def _compute_sparse_score(self, query_weights, passage_weights_list):
        """Sparse 스코어 계산 (learned term matching, SPLADE-like)"""
        scores = []
        
        for passage_weights in passage_weights_list:
            score = 0.0
            # Query와 passage의 공통 토큰에 대해 가중치 곱
            for token_id, q_weight in query_weights.items():
                if token_id in passage_weights:
                    score += q_weight * passage_weights[token_id]
            scores.append(score)
        
        return np.array(scores)

    def _compute_colbert_score(self, query_vecs, passage_vecs_list):
        """ColBERT 스코어 계산 (token-level MaxSim)"""
        scores = []
        
        for passage_vecs in passage_vecs_list:
            # query_vecs: (q_len, dim)
            # passage_vecs: (p_len, dim)
            # MaxSim: for each query token, find max similarity with passage tokens
            similarity_matrix = np.dot(query_vecs, passage_vecs.T)  # (q_len, p_len)
            max_sims = similarity_matrix.max(axis=1)  # (q_len,)
            score = max_sims.sum()  # ColBERT score
            scores.append(score)
        
        return np.array(scores)

    def get_relevant_doc(
        self, 
        query: str, 
        k: int = 1,
        weights: Optional[dict] = None
    ):
        """
        단일 쿼리 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            weights: 각 방법의 가중치 {'dense': 0.5, 'sparse': 0.3, 'colbert': 0.2}
                    None이면 사용하는 방법만으로 균등 분배
        """
        if weights is None:
            # 기본 가중치: 사용하는 방법들에 균등 분배
            num_methods = sum([self.use_dense, self.use_sparse, self.use_colbert])
            default_weight = 1.0 / num_methods
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
            query_dense = query_result['dense_vecs']  # (1, dim)
            dense_scores = self._compute_dense_score(query_dense, self.dense_embeddings)
            # 정규화 (0-1)
            if dense_scores.max() > dense_scores.min():
                dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
            final_scores += weights['dense'] * dense_scores

        if self.use_sparse and weights['sparse'] > 0:
            query_sparse = query_result['lexical_weights'][0]  # dict
            sparse_scores = self._compute_sparse_score(query_sparse, self.sparse_embeddings)
            # 정규화
            if sparse_scores.max() > sparse_scores.min():
                sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
            final_scores += weights['sparse'] * sparse_scores

        if self.use_colbert and weights['colbert'] > 0:
            query_colbert = query_result['colbert_vecs'][0]  # (q_len, dim)
            colbert_scores = self._compute_colbert_score(query_colbert, self.colbert_embeddings)
            # 정규화
            if colbert_scores.max() > colbert_scores.min():
                colbert_scores = (colbert_scores - colbert_scores.min()) / (colbert_scores.max() - colbert_scores.min())
            final_scores += weights['colbert'] * colbert_scores

        # Top-k 추출
        top_indices = np.argsort(final_scores)[::-1][:k]
        top_scores = final_scores[top_indices]

        return top_scores.tolist(), top_indices.tolist()

    def get_relevant_doc_bulk(
        self, 
        queries: List[str], 
        k: int = 1,
        weights: Optional[dict] = None
    ):
        """배치 쿼리 검색"""
        doc_scores, doc_indices = [], []

        for query in tqdm(queries, desc="Retrieving"):
            scores, indices = self.get_relevant_doc(query, k=k, weights=weights)
            doc_scores.append(scores)
            doc_indices.append(indices)

        return doc_scores, doc_indices

    def retrieve(
        self, 
        query_or_dataset: Union[str, Dataset], 
        topk: int = 100,
        weights: Optional[dict] = None
    ):
        """
        검색 실행
        
        Args:
            query_or_dataset: 단일 쿼리 또는 Dataset
            topk: 반환할 문서 수
            weights: {'dense': 0.5, 'sparse': 0.5, 'colbert': 0.0}
        """
        # 단일 쿼리
        if isinstance(query_or_dataset, str):
            scores, indices = self.get_relevant_doc(
                query_or_dataset, k=topk, weights=weights
            )
            passages = [self.contexts[i] for i in indices]
            return scores, passages

        # Dataset 처리
        dataset = query_or_dataset

        with timer(f"Retrieving for {len(dataset)} queries"):
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                dataset["question"], k=topk, weights=weights
            )

        # DataFrame 생성
        rows = []
        for i, example in enumerate(dataset):
            qid = example["id"]
            qtext = example["question"]

            for rank, (idx, score) in enumerate(zip(doc_indices[i], doc_scores[i])):
                row = {
                    "id": qid,
                    "question": qtext,
                    "context": self.contexts[idx],
                    "retrieval_rank": rank,
                    "retrieval_score": float(score),
                }

                if "context" in example:
                    row["original_context"] = example["context"]
                if "answers" in example:
                    row["answers"] = example["answers"]

                rows.append(row)

        return pd.DataFrame(rows)


# 간단한 테스트
if __name__ == "__main__":
    print("Testing BGE-M3 Retrieval...")
    
    retriever = BGEM3Retrieval(
        data_path="data",
        context_path="wikipedia_documents.json",
        use_dense=True,
        use_sparse=True,
        use_colbert=False,  # 메모리 절약
        batch_size=12,
        max_length=512,
    )
    
    retriever.get_embeddings()
    
    # 테스트 쿼리
    test_query = "대한민국의 수도는?"
    scores, passages = retriever.retrieve(
        test_query, 
        topk=3,
        weights={'dense': 0.5, 'sparse': 0.5, 'colbert': 0.0}
    )
    
    print(f"\nTest Query: {test_query}")
    print(f"\nTop-3 Results:")
    for i, (score, passage) in enumerate(zip(scores, passages), 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"    Context: {passage[:150]}...")