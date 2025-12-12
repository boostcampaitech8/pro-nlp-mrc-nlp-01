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
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

from .metrics import compute_multi_k_metrics, build_experiment_config, log_experiment_console


seed = 2024
random.seed(seed)
np.random.seed(seed)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class BM25Retrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.tokenize_fn = tokenize_fn
        self.bm25 = None

    def process_text(self, text: str) -> List[str]:
        # 1. Lowercasing
        text = text.lower()
        # 2. Tokenization
        tokens = self.tokenize_fn(text)
        # 3. N-grams (1, 2)
        ngrams = tokens + [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
        return ngrams

    def get_sparse_embedding(self) -> NoReturn:

        pickle_name = f"bm25_embedding_v1_2.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as f:
                self.bm25 = pickle.load(f)
            print("BM25 pickle load.")
        else:
            print("Build BM25 object")
            # Use process_text for tokenization + n-grams
            tokenized_contexts = [self.process_text(doc) for doc in tqdm(self.contexts, desc="Tokenizing contexts")]
            self.bm25 = BM25Okapi(tokenized_contexts)
            
            with open(emd_path, "wb") as f:
                pickle.dump(self.bm25, f)
            print("BM25 pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 100
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert (
            self.bm25 is not None
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
                tqdm(query_or_dataset, desc="BM25 retrieval: ")
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

        tokenized_query = self.process_text(query)
        with timer("bm25 search"):
            doc_scores = self.bm25.get_scores(tokenized_query)
        
        sorted_indices = np.argsort(doc_scores)[::-1]
        top_k_indices = sorted_indices[:k]
        
        return doc_scores[top_k_indices].tolist(), top_k_indices.tolist()

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        doc_scores = []
        doc_indices = []
        
        for query in tqdm(queries, desc="Bulk search"):
            tokenized_query = self.process_text(query)
            scores = self.bm25.get_scores(tokenized_query)
            sorted_idx = np.argsort(scores)[::-1]
            top_k_idx = sorted_idx[:k]
            
            doc_scores.append(scores[top_k_idx].tolist())
            doc_indices.append(top_k_idx.tolist())
            
        return doc_scores, doc_indices

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

    retriever = BM25Retrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )
    
    retriever.get_sparse_embedding()

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    with timer("bulk query by bm25"):
        df = retriever.retrieve(full_ds)
        df["correct"] = df["original_context"] == df["context"]
        print(
            "correct retrieval result by bm25",
            df["correct"].sum() / len(df),
        )

    with timer("single query by bm25"):
        scores, indices = retriever.retrieve(query)


"""    # 추가) 25.12.04
    # 1) doc_indices 불러오기
    # retrieve()는 DataFrame만 반환 → doc_indices 다시 계산해야 함
    doc_scores, doc_indices = retriever.get_relevant_doc_bulk(
        full_ds["question"], k=100
    )

    # 2) ground truth ids 생성
    ground_truth_ids = []
    for example in full_ds:
        try:
            gt = retriever.contexts.index(example["context"])
        except:
            gt = -1
        ground_truth_ids.append(gt)

    # 3) multi-k metric 계산
    k_list = [20, 50, 100]
    hit_dict, mrr_dict = compute_multi_k_metrics(
        doc_indices, ground_truth_ids, k_list
    )

    # 4) 실험 config 자동 생성
    config = build_experiment_config(
        retriever=retriever,
        topk=100,
        use_faiss=args.use_faiss,
        k_list=k_list,
    )

    # 5) 콘솔 출력
    log_experiment_console(config, hit_dict, mrr_dict)"""