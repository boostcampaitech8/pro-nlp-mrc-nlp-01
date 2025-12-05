import json
import os
import pickle
import sys

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.training.train_dpr import DPREncoder  # Reuse Encoder class

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class DenseRetrieval:
    def __init__(
        self,
        args,
        dataset,
        model_name_or_path: str,
        data_path: Optional[str] = "data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        self.data_path = data_path
        self.args = args
        self.dataset = dataset
        
        # Load Wikipedia Contexts
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Load Tokenizer & Encoders
        # Check if model_path has specific sub-folders for encoders
        q_path = os.path.join(model_name_or_path, "question_encoder")
        c_path = os.path.join(model_name_or_path, "context_encoder")

        if os.path.isdir(q_path) and os.path.isdir(c_path):
            print(f"Loading encoders from {q_path} and {c_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(q_path)
            self.q_encoder = DPREncoder(q_path).cuda()
            self.c_encoder = DPREncoder(c_path).cuda()
        else:
            print(f"Loading encoders from base model {model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.q_encoder = DPREncoder(model_name_or_path).cuda()
            self.c_encoder = DPREncoder(model_name_or_path).cuda()

        self.p_embedding = None
        self.indexer = None

    def get_dense_embedding(self) -> NoReturn:
        pickle_name = f"dense_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as f:
                self.p_embedding = pickle.load(f)
            print("Dense embedding pickle loaded.")
        else:
            print("Build passage embedding")
            self.c_encoder.eval()
            
            p_embs = []
            batch_size = 128  # Increased from 16 for faster speed (32GB VRAM)
            
            with torch.no_grad():
                for i in tqdm(range(0, len(self.contexts), batch_size), desc="Encoding passages"):
                    batch_contexts = self.contexts[i : i + batch_size]
                    inputs = self.tokenizer(
                        batch_contexts, 
                        padding=True, 
                        truncation=True, 
                        max_length=512, 
                        return_tensors="pt"
                    ).to("cuda")
                    
                    emb = self.c_encoder(inputs["input_ids"], inputs["attention_mask"])
                    p_embs.append(emb.cpu().numpy())
            
            self.p_embedding = np.concatenate(p_embs, axis=0)
            print(self.p_embedding.shape)
            
            with open(emd_path, "wb") as f:
                pickle.dump(self.p_embedding, f)
            print("Dense embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:
        # Save index with a unique name based on clusters or type
        indexer_name = f"faiss_dense_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)
        else:
            p_emb = self.p_embedding.astype(np.float32)
            emb_dim = p_emb.shape[-1]

            # Using Flat Index for exact search (Accuracy > Speed for now)
            # If scaling up, use IVFFlat or IVFPQ
            index = faiss.IndexFlatIP(emb_dim) # Inner Product matching Dot Product of Bi-Encoder
            self.indexer = index
            self.indexer.add(p_emb)
            
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 100
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        
        assert self.indexer is not None, "build_faiss() must be called first."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")
            
            top_passages = []
            for idx in range(topk):
                print(f"Top-{idx+1} passage with score {doc_scores[idx]:.4f}")
                passage = self.contexts[doc_indices[idx]]
                print(passage)
                top_passages.append(passage)
            return (doc_scores, top_passages)

        elif isinstance(query_or_dataset, Dataset):
            # Bulk Retrieval
            queries = query_or_dataset["question"]
            total = []

            with timer("bulk query search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    queries, k=topk
                )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
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
        self.q_encoder.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                [query], 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            ).to("cuda")
            q_emb = self.q_encoder(inputs["input_ids"], inputs["attention_mask"])
            q_emb = q_emb.cpu().numpy().astype(np.float32)

        D, I = self.indexer.search(q_emb, k)
        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        self.q_encoder.eval()
        q_embs = []
        batch_size = 128
        
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
                batch_queries = queries[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_queries, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to("cuda")
                emb = self.q_encoder(inputs["input_ids"], inputs["attention_mask"])
                q_embs.append(emb.cpu().numpy())
        
        q_embs = np.concatenate(q_embs, axis=0).astype(np.float32)
        D, I = self.indexer.search(q_embs, k)
        return D.tolist(), I.tolist()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", metavar="./data/train_dataset", type=str, default="../data/train_dataset")
    parser.add_argument("--model_name_or_path", metavar="./models/dpr_model", type=str, default="klue/bert-base")
    parser.add_argument("--data_path", metavar="./data", type=str, default="./data")
    parser.add_argument("--context_path", metavar="wikipedia_documents", type=str, default="wikipedia_documents.json")
    
    args = parser.parse_args()

    # Load Dataset
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )

    # Init Retriever
    retriever = DenseRetrieval(
        args=args,
        dataset=full_ds,
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    # Embed & Index
    retriever.get_dense_embedding()
    retriever.build_faiss()

    # Test Query
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    with timer("single query"):
        scores, indices = retriever.retrieve(query)

    # Eval (Optional)
    # with timer("bulk query"):
    #     df = retriever.retrieve(full_ds)
    #     df["correct"] = df["original_context"] == df["context"]
    #     print("correct retrieval result", df["correct"].sum() / len(df))

