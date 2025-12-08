import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class KURERetrieval:
    def __init__(
        self,
        data_path: Optional[str] = "data",
        context_path: Optional[str] = "wikipedia_documents.json",
        model_name: str = "nlpai-lab/KURE-v1",
    ) -> NoReturn:

        self.data_path = data_path
        self.model_name = model_name
        
        # Generate unique cache name based on model path
        model_hash = model_name.replace("/", "_").replace("\\", "_")
        self.cache_prefix = f"kure_{model_hash}"
        
        # Load Wikipedia Contexts
        context_file = os.path.join(data_path, context_path)
        print(f"Loading contexts from {context_file}...")
        with open(context_file, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Load Model
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Use GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.p_embedding = None
        self.indexer = None
        self.retrieval_metrics = {}

    def get_dense_embedding(self) -> NoReturn:
        pickle_name = f"{self.cache_prefix}_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as f:
                self.p_embedding = pickle.load(f)
            print(f"Dense embedding loaded from {pickle_name}")
        else:
            print("Build passage embedding (this may take a while)...")
            
            # SentenceTransformer encode helps with batching
            # We treat contexts as queries? No, as documents.
            # KURE might have specific instructions but typically encode works for both.
            # Convert to list ensuring strings
            
            # Using encode with show_progress_bar=True
            self.p_embedding = self.model.encode(
                self.contexts,
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False # Check if normalization is needed? usually dot product for SBERT
            )
            
            print(f"Embedding shape: {self.p_embedding.shape}")
            
            with open(emd_path, "wb") as f:
                pickle.dump(self.p_embedding, f)
            print(f"Dense embedding saved to {pickle_name}")

    def build_faiss(self, num_clusters=64) -> NoReturn:
        indexer_name = f"{self.cache_prefix}_faiss_flat.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        
        if os.path.isfile(indexer_path):
            print(f"Faiss indexer loaded from {indexer_name}")
            self.indexer = faiss.read_index(indexer_path)
        else:
            p_emb = self.p_embedding.astype(np.float32)
            emb_dim = p_emb.shape[-1]

            # Use Flat Inner Product (for Cosine Similarity if normalized, or Dot Product)
            # KURE usually uses Cosine Similarity logic, so we should check normalization.
            # SentenceTransformer docs say: "We recommend to use Cosine Similarity"
            # If so, we should normalize embeddings.
            # Let's re-normalize embeddings just in case for IP search.
            faiss.normalize_L2(p_emb)
            
            index = faiss.IndexFlatIP(emb_dim)
            self.indexer = index
            self.indexer.add(p_emb)
            
            faiss.write_index(self.indexer, indexer_path)
            print(f"Faiss indexer saved to {indexer_name}")

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
            
            with timer("bulk query search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    queries, k=topk
                )
            
            # Since this is primarily used inside a pipeline, we just return the bulk results normally.
            # But adhering to the interface that returns a DataFrame for evaluation/inspection:
            
            total = []
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense result processing")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    # "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]) # Save memory, don't build huge strings
                    "retrieved_indices": doc_indices[idx],
                    "retrieved_scores": doc_scores[idx]
                }
                total.append(tmp)
            
            return pd.DataFrame(total)

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        q_emb = self.model.encode(query, convert_to_numpy=True)
        q_emb = q_emb.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q_emb) # Consistent with build_faiss
        
        D, I = self.indexer.search(q_emb, k)
        return D.tolist()[0], I.tolist()[0]
    
    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        q_embs = self.model.encode(
            queries,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        q_embs = q_embs.astype(np.float32)
        faiss.normalize_L2(q_embs)
        
        D, I = self.indexer.search(q_embs, k)
        return D.tolist(), I.tolist()

import torch
