import json
import os
import time
import random
import torch
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from sentence_transformers import SparseEncoder

seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # 추가: torch seed 고정


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

"""
전체 흐름 
1. SparseRetrieval
    - 질문당 top-k passage (row k개) 생성
2. run_splade_retrieval
    - 질문 단위 hit@k 계산
    - retrieval 결과를 HF Dataset으로 변환(validation split)
3. run_mrc
    - eval_dataset = retrieval 결과 (질문×passage)
    - eval_examples = 원래 validation (질문 단위)
    - prepare_validation_features → 각 passage + sliding window → features
    - postprocess_qa_predictions → 질문 id 기준으로 여러 feature들(span들) 중 최고 스코어 답 하나 선택
    - metric은 원래 validation 기준으로 EM/F1 계산
"""

class SparseRetrieval:
    def __init__(
        self,
        data_path: str = "data",
        context_path: str = "wikipedia_documents.json",
        splade_model_name: str = "telepix/PIXIE-Splade-Preview",
    ):

        self.data_path = data_path

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # key(문서 ID)와 text를 같이 가져오기
        wiki_items = list(wiki.items())  # [(id, {text:..., ...}), ...]

        self.ids = [k for k, v in wiki_items]           # ["0", "1", ...]
        self.contexts = [v["text"] for k, v in wiki_items]  # 각 id에 대응하는 text

        print(f"Number of passages : {len(self.contexts)}")

        self.encoder = SparseEncoder(splade_model_name)
        self.p_embedding = None


    # =======================================================================

    def _to_scipy_sparse(self, emb):
        """SPLADE encoder output → scipy csr_matrix로 변환"""
        # Case 1: torch sparse tensor
        if isinstance(emb, torch.Tensor) and emb.is_sparse:
            emb = emb.to_dense().cpu().numpy()
            return csr_matrix(emb)

        # Case 2: torch dense tensor
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
            return csr_matrix(emb)

        # Case 3: numpy array
        if isinstance(emb, np.ndarray):
            return csr_matrix(emb)

        # Case 4: already scipy sparse
        if hasattr(emb, "shape") and hasattr(emb, "tocsr"):
            return emb.tocsr()  # 명시적 변환

        raise TypeError(f"Unknown embedding type: {type(emb)}")

    # =======================================================================

    def get_sparse_embedding(self):
        emd_path = os.path.join(self.data_path, "splade_embedding.npz")

        # Already exists → load
        if os.path.isfile(emd_path):
            with timer("Loading SPLADE embeddings"):
                self.p_embedding = load_npz(emd_path)
                print(f"Loaded embedding shape: {self.p_embedding.shape}")
            return

        print("Building passage embedding (SPLADE)")
        batch_size = 128

        batches = [
            self.contexts[i:i + batch_size]
            for i in range(0, len(self.contexts), batch_size)
        ]

        sparse_rows = None

        with timer("Building SPLADE embeddings"):
            for batch in tqdm(batches, desc="Encoding passages"):
                # IMPORTANT: convert_to_tensor=True로 명시 (기본값이지만 명확성)
                emb = self.encoder.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )

                # Convert SPLADE output → scipy csr
                emb = self._to_scipy_sparse(emb)

                # Accumulate rows
                if sparse_rows is None:
                    sparse_rows = emb
                else:
                    sparse_rows = vstack([sparse_rows, emb])

        self.p_embedding = sparse_rows.tocsr()
        save_npz(emd_path, self.p_embedding)
        print(f"SPLADE embedding saved: {self.p_embedding.shape}")
        print(f"Sparsity: {100 * (1 - self.p_embedding.nnz / np.prod(self.p_embedding.shape)):.2f}%")

    # =======================================================================

    def _encode_query(self, queries):
        """Query encoding 통일"""
        emb = self.encoder.encode(
            queries,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return self._to_scipy_sparse(emb)

    # =======================================================================

    def get_relevant_doc(self, query: str, k: int = 1):
        """단일 쿼리에 대한 검색"""
        if self.p_embedding is None:
            raise ValueError("Passage embeddings not initialized. Call get_sparse_embedding() first.")
        
        query_vec = self._encode_query([query])

        # Sparse matrix multiplication
        result = query_vec @ self.p_embedding.T
        result = result.toarray().squeeze()

        # Top-k 인덱스 추출
        sorted_idx = np.argsort(result)[::-1]
        top_idx = sorted_idx[:k]

        return result[top_idx].tolist(), top_idx.tolist()

    # =======================================================================

    def get_relevant_doc_bulk(self, queries: List[str], k: int = 1):
        """배치 쿼리 검색"""
        if self.p_embedding is None:
            raise ValueError("Passage embeddings not initialized. Call get_sparse_embedding() first.")
        
        query_vec = self._encode_query(queries)
        result = query_vec @ self.p_embedding.T
        result = result.toarray()

        doc_scores, doc_indices = [], []

        for r in result:
            sorted_idx = np.argsort(r)[::-1]
            top_idx = sorted_idx[:k]

            doc_scores.append(r[top_idx].tolist())
            doc_indices.append(top_idx.tolist())

        return doc_scores, doc_indices

    # =======================================================================
    # 질문 하나당 row 하나 + context = top-k 문서 이어붙인 긴 문자열
    # => 질문 하나당 top-k row(id는 같고 context만 다른 passage)
    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: int = 100):
        """
        검색 실행

        Args:
            query_or_dataset: 단일 쿼리(str) 또는 Dataset
            topk: 반환할 문서 수

        Returns:
            - str 입력: (scores, passages) tuple
            - Dataset 입력: DataFrame with one row per (question, passage)
        """
        if self.p_embedding is None:
            raise ValueError("Passage embeddings not initialized. Call get_sparse_embedding() first.")

        # 단일 쿼리일 때는 기존과 동일하게 동작
        if isinstance(query_or_dataset, str):
            scores, idxs = self.get_relevant_doc(query_or_dataset, k=topk)
            passages = [self.contexts[i] for i in idxs]
            return scores, passages

        # Dataset 처리 (질문 하나당 top-k passage를 "각각 한 row"로 생성)
        dataset = query_or_dataset

        with timer(f"Retrieving for {len(dataset)} queries"):
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                dataset["question"], k=topk
            )

        rows = []
        for i, example in enumerate(dataset):
            qid = example["id"]
            qtext = example["question"]

            for rank, (idx, score) in enumerate(zip(doc_indices[i], doc_scores[i])):
                row = {
                    "id": qid,                      # 원래 질문 id (중복 허용)
                    "question": qtext,
                    "context": self.contexts[idx],  # 개별 passage
                    "retrieval_rank": rank,         # 0: top-1, 1: top-2, ...
                    "retrieval_score": float(score),
                }

                # 평가용 정보 유지
                if "context" in example:
                    row["original_context"] = example["context"]
                if "answers" in example:
                    row["answers"] = example["answers"]

                rows.append(row)

        # 질문 개수 × topk 개의 row가 생김
        return pd.DataFrame(rows)


    # =======================================================================
    
    def retrieve_with_scores(self, query_or_dataset: Union[str, Dataset], topk: int = 100):
        """
        검색 결과에 스코어를 포함하여 반환
        """
        if self.p_embedding is None:
            raise ValueError("Passage embeddings not initialized. Call get_sparse_embedding() first.")

        if isinstance(query_or_dataset, str):
            return self.get_relevant_doc(query_or_dataset, k=topk)

        dataset = query_or_dataset
        doc_scores, doc_indices = self.get_relevant_doc_bulk(
            dataset["question"], k=topk
        )

        rows = []
        for i, example in enumerate(dataset):
            row = {
                "id": example["id"],
                "question": example["question"],
                "doc_indices": doc_indices[i],
                "doc_scores": doc_scores[i],
                "contexts": [self.contexts[idx] for idx in doc_indices[i]],
            }

            if "context" in example:
                row["original_context"] = example["context"]
            if "answers" in example:
                row["answers"] = example["answers"]

            rows.append(row)

        return pd.DataFrame(rows)