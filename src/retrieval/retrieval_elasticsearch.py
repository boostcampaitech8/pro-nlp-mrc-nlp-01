import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from elasticsearch import Elasticsearch, helpers


seed = 2024
random.seed(seed)
np.random.seed(seed)

"""
elasticsearch 

python retrieval_elasticsearch.py \
    --dataset_name ./data/train_dataset \
    --model_name_or_path bert-base-multilingual-cased \
    --data_path ./data \
    --context_path wikipedia_documents.json \
    --use_elastic True

"""

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        context_texts = [v["text"] for v in wiki.values()]
        self.contexts = list(dict.fromkeys(context_texts))
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.p_embedding = None
        self.indexer = None

    def get_sparse_embedding(self) -> NoReturn:

        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        embedding_exists = os.path.isfile(emd_path) and os.path.isfile(tfidfv_path)
        if embedding_exists:
            with open(emd_path, "rb") as f:
                self.p_embedding = pickle.load(f)
            with open(tfidfv_path, "rb") as f:
                self.tfidfv = pickle.load(f)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as f:
                pickle.dump(self.p_embedding, f)
            with open(tfidfv_path, "wb") as f:
                pickle.dump(self.tfidfv, f)
            print("Embedding pickle saved.")


    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 100
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert (
            self.p_embedding is not None
        ), "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            top_passages = []
            for idx in range(topk):
                print(f"Top-{idx+1} passage with score {doc_scores[idx]:4f}")
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
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

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
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

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


class ElasticSearchRetrieval:
    """
    SparseRetrieval이 TF-IDF 직접 곱셈으로 찾는 버전이라면,
    이 클래스는 Elasticsearch에 인덱싱해두고 BM25로 검색하는 버전.
    인터페이스는 최대한 비슷하게 맞춰서, 평가 코드 재사용 가능하게 구현.
    """

    def __init__(
        self,
        tokenize_fn,  # 일단 시그니처 맞춰두지만 ES에서는 직접 쓰진 않음
        data_path: Optional[str] = "data",
        context_path: Optional[str] = "wikipedia_documents.json",
        es_host: str = "http://localhost:9200",
        index_name: str = "wiki_mrc",
    ) -> NoReturn:

        self.data_path = data_path
        self.index_name = index_name

        # 1) 위키 문서 로딩 (SparseRetrieval과 동일)
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        context_texts = [v["text"] for v in wiki.values()]
        self.contexts = list(dict.fromkeys(context_texts))
        print(f"[ElasticSearchRetrieval] Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # 2) ES 클라이언트
        self.es = Elasticsearch(es_host)

    def build_elasticsearch_index(self, recreate: bool = False, batch_size: int = 500):
        """
        위키 문서들을 ES 인덱스에 넣는 함수.
        - recreate=True 이면 기존 인덱스 삭제 후 다시 생성
        """

        # 인덱스 존재 여부 확인
        index_exists = self.es.indices.exists(index=self.index_name)

        if recreate and index_exists:
            print(f"[ElasticSearchRetrieval] delete existing index: {self.index_name}")
            self.es.indices.delete(index=self.index_name)
            index_exists = False

        if not index_exists:
            # 간단한 매핑 (text 필드만 full-text search)
            body = {
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "integer"},
                        "text":   {"type": "text"},
                    }
                }
            }
            print(f"[ElasticSearchRetrieval] create index: {self.index_name}")
            self.es.indices.create(index=self.index_name, body=body)

            # bulk 인덱싱
            print("[ElasticSearchRetrieval] start bulk indexing to ES...")
            actions = (
                {
                    "_index": self.index_name,
                    "_id": i,
                    "_source": {
                        "doc_id": i,
                        "text": text,
                    },
                }
                for i, text in enumerate(self.contexts)
            )
            helpers.bulk(self.es, actions, chunk_size=batch_size)
            print("[ElasticSearchRetrieval] bulk indexing done.")
        else:
            print(f"[ElasticSearchRetrieval] index already exists: {self.index_name}")

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        단일 쿼리 -> 상위 k개 문서의 (score 리스트, doc_index 리스트) 반환
        doc_index는 self.contexts에서의 인덱스(doc_id)로 맞춰줌
        """

        body = {
            "size": k,
            "query": {
                "match": {
                    "text": query
                }
            }
        }
        res = self.es.search(index=self.index_name, body=body)

        hits = res["hits"]["hits"]
        doc_scores = [hit["_score"] for hit in hits]
        doc_indices = [hit["_source"]["doc_id"] for hit in hits]

        return doc_scores, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List[str], k: Optional[int] = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        여러 쿼리 리스트에 대해 각각 상위 k개 문서의 score, index 반환
        (간단하게 for loop로 돌려도 됨. 나중에 msearch로 최적화 가능)
        """
        all_scores = []
        all_indices = []

        for q in tqdm(queries, desc="ElasticSearch bulk retrieve"):
            scores, indices = self.get_relevant_doc(q, k=k)
            all_scores.append(scores)
            all_indices.append(indices)

        return all_scores, all_indices

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 100
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        if isinstance(query_or_dataset, str):
            # 단일 쿼리
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            top_passages = []
            for idx in range(len(doc_indices)):
                print(f"Top-{idx+1} passage with score {doc_scores[idx]:4f}")
                passage = self.contexts[doc_indices[idx]]
                print(passage)
                top_passages.append(passage)

            return doc_scores, top_passages

        elif isinstance(query_or_dataset, Dataset):
            # Dataset 전체에 대해 retrieval
            total = []
            queries = query_or_dataset["question"]

            with timer("query elasticsearch search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    queries, k=topk
                )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="ElasticSearch retrieval: ")
            ):
                # topk 문서들을 공백으로 이어붙이기 (기존 SparseRetrieval과 동일한 형태)
                context_list = [self.contexts[pid] for pid in doc_indices[idx]]
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(context_list),
                }
                # 정답이 있는 경우 원래 context/answers도 같이 복사
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)




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
    parser.add_argument("--use_elastic", metavar=False, type=bool, help="")

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

    # 1) Elasticsearch 사용
    if getattr(args, "use_elastic", False):
        retriever = ElasticSearchRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
            es_host="http://localhost:9200",
            index_name="wiki_mrc",
        )

        # 인덱스 없으면 만들고, 있으면 그냥 재사용
        retriever.build_elasticsearch_index(recreate=False)

        with timer("bulk query by elasticsearch"):
            df = retriever.retrieve(full_ds, topk=100)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by elasticsearch",
                df["correct"].sum() / len(df),
            )

        with timer("single query by elasticsearch"):
            scores, passages = retriever.retrieve(query, topk=5)

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
