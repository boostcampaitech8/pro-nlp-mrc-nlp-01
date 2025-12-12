import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from rank_bm25 import BM25Plus
from tqdm.auto import tqdm

seed = 2024
random.seed(seed)
np.random.seed(seed)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

# Global variables for multiprocessing
global_bm25 = None
global_tokenize_fn = None

def query_bm25_wrapper(args):
    """
    Top-level helper function for multiprocessing.
    Uses global variables to avoid pickling large objects.
    """
    query, k = args
    global global_bm25, global_tokenize_fn
    
    # Tokenization logic duplicated from process_text to avoid self dependency
    text = query.lower()
    tokens = global_tokenize_fn(text)
    tokenized_query = tokens + [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
    
    scores = global_bm25.get_scores(tokenized_query)
    sorted_idx = np.argsort(scores)[::-1]
    top_k_idx = sorted_idx[:k]
    return scores[top_k_idx].tolist(), top_k_idx.tolist()

class BM25PlusRetrieval:
    """BM25Plus Retrieval with retrieval accuracy metrics for wandb logging."""
    
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
        
        # Retrieval metrics storage
        self.retrieval_metrics = {}

    def process_text(self, text: str) -> List[str]:
        # 1. Lowercasing
        text = text.lower()
        # 2. Tokenization
        tokens = self.tokenize_fn(text)
        # 3. N-grams (1, 2)
        ngrams = tokens + [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
        return ngrams

    def get_sparse_embedding(self) -> NoReturn:

        # Delta parameter for BM25Plus
        delta = 0.25
        pickle_name = f"bm25plus_delta{delta}_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as f:
                self.bm25 = pickle.load(f)
            print("BM25Plus pickle load.")
        else:
            print(f"Build BM25Plus object with delta={delta}")
            # Use process_text for tokenization + n-grams
            tokenized_contexts = [self.process_text(doc) for doc in tqdm(self.contexts, desc="Tokenizing contexts")]
            self.bm25 = BM25Plus(tokenized_contexts, delta=delta)
            
            with open(emd_path, "wb") as f:
                pickle.dump(self.bm25, f)
            print("BM25Plus pickle saved.")

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
            correct_count = 0
            has_ground_truth = False
            
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="BM25Plus retrieval: ")
            ):
                # Combine retrieved contexts
                retrieved_context = " ".join(
                    [self.contexts[pid] for pid in doc_indices[idx]]
                )
                
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": retrieved_context,
                }
                
                # Check if ground truth exists and calculate retrieval accuracy
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                    has_ground_truth = True
                    
                    # Check if original context is in any of the top-k retrieved contexts
                    original_context = example["context"]
                    retrieved_contexts_list = [self.contexts[pid] for pid in doc_indices[idx]]
                    
                    # Check if original context is contained in any retrieved context
                    is_correct = any(
                        original_context in rc or rc in original_context 
                        for rc in retrieved_contexts_list
                    )
                    if is_correct:
                        correct_count += 1
                        
                total.append(tmp)

            cqas = pd.DataFrame(total)
            
            # Calculate and store retrieval metrics
            if has_ground_truth:
                retrieval_accuracy = correct_count / len(query_or_dataset)
                
                # Calculate MRR
                mrr_sum = 0.0
                for idx, example in enumerate(query_or_dataset):
                    original_context = example["context"]
                    retrieved_contexts_list = [self.contexts[pid] for pid in doc_indices[idx]]
                    
                    # Find the first rank (1-based) where the ground truth appears
                    rank = 0
                    for i, rc in enumerate(retrieved_contexts_list):
                        if original_context in rc or rc in original_context:
                            rank = i + 1
                            break
                    
                    if rank > 0:
                        mrr_sum += 1.0 / rank
                
                mrr = mrr_sum / len(query_or_dataset)

                self.retrieval_metrics = {
                    "retrieval_accuracy": retrieval_accuracy,
                    "mrr": mrr,
                    "correct_count": correct_count,
                    "total_count": len(query_or_dataset),
                    "top_k": topk,
                }
                print(f"\n{'='*50}")
                print(f"Retrieval Accuracy @ {topk}: {retrieval_accuracy:.4f} ({correct_count}/{len(query_or_dataset)})")
                print(f"MRR @ {topk}: {mrr:.4f}")
                print(f"{'='*50}\n")
            else:
                self.retrieval_metrics = {
                    "top_k": topk,
                    "total_count": len(query_or_dataset),
                    "retrieval_accuracy": None,  # Cannot calculate without ground truth
                }
                print(f"\n[INFO] Ground truth not available - retrieval accuracy cannot be calculated")
            
            return cqas
    
    def get_retrieval_metrics(self) -> Dict:
        """Return the retrieval metrics for wandb logging."""
        return self.retrieval_metrics

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        tokenized_query = self.process_text(query)
        with timer("bm25plus search"):
            doc_scores = self.bm25.get_scores(tokenized_query)
        
        sorted_indices = np.argsort(doc_scores)[::-1]
        top_k_indices = sorted_indices[:k]
        
        return doc_scores[top_k_indices].tolist(), top_k_indices.tolist()

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        # Set global variables for multiprocessing
        global global_bm25, global_tokenize_fn
        global_bm25 = self.bm25
        global_tokenize_fn = self.tokenize_fn

        # Use multiprocessing for faster search
        from multiprocessing import Pool, cpu_count
        
        # Use fork to share memory of global_bm25 without pickling
        with Pool(processes=cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(query_bm25_wrapper, zip(queries, [k]*len(queries))),
                    total=len(queries),
                    desc="Bulk search (multiprocessing)"
                )
            )
        
        # Clear globals to free memory reference if needed (optional)
        global_bm25 = None
        global_tokenize_fn = None
        
        doc_scores = [res[0] for res in results]
        doc_indices = [res[1] for res in results]
            
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

    retriever = BM25PlusRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )
    
    retriever.get_sparse_embedding()

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    with timer("bulk query by bm25plus"):
        df = retriever.retrieve(full_ds)
        metrics = retriever.get_retrieval_metrics()
        print(f"Retrieval Metrics: {metrics}")

    with timer("single query by bm25plus"):
        scores, indices = retriever.retrieve(query)
