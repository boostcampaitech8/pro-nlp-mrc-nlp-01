"""Sparse Retrieval 모듈 - TF-IDF 기반 검색 구현."""

import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

# 시드 고정
_SEED = 2024
random.seed(_SEED)
np.random.seed(_SEED)


@contextmanager
def timer(name: str):
    """실행 시간 측정 컨텍스트 매니저.
    
    Args:
        name: 작업 이름
    """
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class SparseRetrieval:
    """TF-IDF 기반 Sparse Retrieval 클래스."""

    def __init__(
        self,
        tokenize_fn: callable,
        data_path: Optional[str] = "data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        """SparseRetrieval 초기화.
        
        Args:
            tokenize_fn: 토큰화 함수
            data_path: 데이터 경로
            context_path: 컨텍스트 파일 경로
        """
        self.data_path = data_path
        context_file = os.path.join(data_path, context_path)
        
        with open(context_file, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        context_texts = [v["text"] for v in wiki.values()]
        self.contexts = list(dict.fromkeys(context_texts))
        print(f"Lengths of unique contexts: {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.p_embedding = None
        self.indexer = None

    def get_sparse_embedding(self) -> None:
        """TF-IDF 임베딩 생성 또는 로드.
        
        캐시된 임베딩이 있으면 로드하고, 없으면 새로 생성하여 저장합니다.
        """
        pickle_name = "sparse_embedding.bin"
        tfidfv_name = "tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        embedding_exists = os.path.isfile(emd_path) and os.path.isfile(tfidfv_path)
        if embedding_exists:
            with open(emd_path, "rb") as f:
                self.p_embedding = pickle.load(f)
            with open(tfidfv_path, "rb") as f:
                self.tfidfv = pickle.load(f)
            print("Embedding pickle loaded.")
        else:
            print("Building passage embedding...")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(f"Embedding shape: {self.p_embedding.shape}")
            with open(emd_path, "wb") as f:
                pickle.dump(self.p_embedding, f)
            with open(tfidfv_path, "wb") as f:
                pickle.dump(self.tfidfv, f)
            print("Embedding pickle saved.")

    def build_faiss(self, num_clusters: int = 64) -> None:
        """FAISS 인덱스 생성 또는 로드.
        
        Args:
            num_clusters: 클러스터 개수
        """
        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        
        if os.path.isfile(indexer_path):
            print("Loading saved FAISS indexer...")
            self.indexer = faiss.read_index(indexer_path)
        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            quantizer = faiss.IndexFlatL2(emb_dim)
            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("FAISS indexer saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 100
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """쿼리 또는 데이터셋에 대한 검색 수행.
        
        Args:
            query_or_dataset: 검색할 쿼리 문자열 또는 Dataset
            topk: 반환할 상위 k개 문서 수
        
        Returns:
            쿼리 문자열인 경우: (점수 리스트, 문서 리스트) 튜플
            Dataset인 경우: 검색 결과 DataFrame
        
        Raises:
            AssertionError: 임베딩이 생성되지 않은 경우
        """
        assert (
            self.p_embedding is not None
        ), "get_sparse_embedding() 메소드를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print(f"[Search query]\n{query_or_dataset}\n")

            top_passages = []
            for idx in range(min(topk, len(doc_scores))):
                print(f"Top-{idx+1} passage with score {doc_scores[idx]:.4f}")
                passage = self.contexts[doc_indices[idx]]
                print(passage)
                top_passages.append(passage)
            return (doc_scores, top_passages)

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                has_ground_truth = "context" in example.keys() and "answers" in example.keys()
                if has_ground_truth:
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """단일 쿼리에 대한 관련 문서 검색.
        
        Args:
            query: 검색 쿼리 문자열
            k: 반환할 상위 k개 문서 수
        
        Returns:
            (점수 리스트, 문서 인덱스 리스트) 튜플
        
        Raises:
            AssertionError: 쿼리에 vocab에 없는 단어만 있는 경우
        """
        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        result_array = result.squeeze()
        sorted_indices = np.argsort(result_array)[::-1]
        top_k_indices = sorted_indices[:k]
        doc_score = result_array[top_k_indices].tolist()
        doc_indices = top_k_indices.tolist()
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List[str], k: Optional[int] = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """여러 쿼리에 대한 관련 문서 일괄 검색.
        
        Args:
            queries: 검색 쿼리 문자열 리스트
            k: 각 쿼리당 반환할 상위 k개 문서 수
        
        Returns:
            (점수 리스트의 리스트, 문서 인덱스 리스트의 리스트) 튜플
        
        Raises:
            AssertionError: 쿼리에 vocab에 없는 단어만 있는 경우
        """
        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        
        doc_scores = []
        doc_indices = []
        num_queries = result.shape[0]
        
        for query_idx in range(num_queries):
            query_result = result[query_idx, :]
            sorted_idx = np.argsort(query_result)[::-1]
            top_k_idx = sorted_idx[:k]
            doc_scores.append(query_result[top_k_idx].tolist())
            doc_indices.append(top_k_idx.tolist())
        
        return doc_scores, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """FAISS 인덱스를 사용한 검색 수행.
        
        Args:
            query_or_dataset: 검색할 쿼리 문자열 또는 Dataset
            topk: 반환할 상위 k개 문서 수
        
        Returns:
            쿼리 문자열인 경우: (점수 리스트, 문서 리스트) 튜플
            Dataset인 경우: 검색 결과 DataFrame
        
        Raises:
            AssertionError: FAISS 인덱스가 생성되지 않은 경우
        """
        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print(f"[Search query]\n{query_or_dataset}\n")

            top_passages = []
            for idx in range(min(topk, len(doc_scores))):
                print(f"Top-{idx+1} passage with score {doc_scores[idx]:.4f}")
                passage = self.contexts[doc_indices[idx]]
                print(passage)
                top_passages.append(passage)
            return (doc_scores, top_passages)

        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                context_list = [self.contexts[pid] for pid in doc_indices[idx]]
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(context_list),
                }
                has_ground_truth = "context" in example.keys() and "answers" in example.keys()
                if has_ground_truth:
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List[float], List[int]]:
        """FAISS를 사용한 단일 쿼리 검색.
        
        Args:
            query: 검색 쿼리 문자열
            k: 반환할 상위 k개 문서 수
        
        Returns:
            (점수 리스트, 문서 인덱스 리스트) 튜플
        
        Raises:
            AssertionError: 쿼리에 vocab에 없는 단어만 있는 경우
        """
        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List[str], k: Optional[int] = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """FAISS를 사용한 여러 쿼리 일괄 검색.
        
        Args:
            queries: 검색 쿼리 문자열 리스트
            k: 각 쿼리당 반환할 상위 k개 문서 수
        
        Returns:
            (점수 리스트의 리스트, 문서 인덱스 리스트의 리스트) 튜플
        
        Raises:
            AssertionError: 쿼리에 vocab에 없는 단어만 있는 경우
        """
        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")

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

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)
