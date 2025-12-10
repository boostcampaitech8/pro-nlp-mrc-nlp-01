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


class BGEM3KUREHybridRetrieval:
    """
    Hybrid Retrieval:
    - Dense: KURE-v1
    - Sparse: BGE-M3 sparse
    - Re-ranker: BGE reranker

    기존 retrieval_bge_m3_fixed.py 와 동일한 구조 / 동일한 인터페이스 유지
    """

    def __init__(
        self,
        data_path="data",
        context_path="wikipedia_documents.json",

        # Sparse Encoder (BGE-M3)
        sparse_model_name="BAAI/bge-m3",

        # Dense Encoder (KURE)
        kure_model_name="nlpai-lab/KURE-v1",

        # Reranker
        reranker_name="BAAI/bge-reranker-v2-m3",

        use_fp16=True,
        batch_size=8,
        max_length=512,

        use_dense=True,
        use_sparse=True,
        use_reranker=True,

        max_memory_gb=28.0,
    ):
        self.data_path = data_path
        self.base_batch_size = batch_size
        self.max_length = max_length

        self.use_dense = use_dense
        self.use_sparse = use_sparse
        self.use_reranker = use_reranker
        self.max_memory_gb = max_memory_gb

        # Wikipedia Documents
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        wiki_items = list(wiki.items())
        self.ids = [k for k, v in wiki_items]
        self.contexts = [v["text"] for k, v in wiki_items]

        print(f"Number of passages: {len(self.contexts)}")

        # Load Sparse (BGE-M3)
        print(f"Loading BGE-M3 sparse model: {sparse_model_name}")
        from FlagEmbedding import BGEM3FlagModel
        self.sparse_model = BGEM3FlagModel(
            sparse_model_name,
            use_fp16=use_fp16,
            device="cuda"
        )
        print("✅ BGE-M3 sparse loaded")

        # Load Dense (KURE)
        print(f"Loading KURE-v1 dense retriever: {kure_model_name}")
        from transformers import AutoTokenizer, AutoModel
        self.kure_tokenizer = AutoTokenizer.from_pretrained(kure_model_name)
        self.kure_model = AutoModel.from_pretrained(kure_model_name).cuda()
        print("✅ KURE-v1 dense loaded")

        # Load reranker
        self.reranker = None
        if use_reranker:
            print(f"Loading BGE reranker: {reranker_name}")
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(
                reranker_name,
                use_fp16=use_fp16,
                device="cuda"
            )
            print("✅ Reranker loaded")

        # Embeddings
        self.dense_embeddings = None
        self.sparse_embeddings = None


    ######################################
    # Utility
    ######################################

    def _get_adaptive_batch_size(self, texts):
        avg_len = sum(len(t) for t in texts) / len(texts)

        if avg_len < 128:
            return min(self.base_batch_size * 2, 16)
        elif avg_len < 256:
            return self.base_batch_size
        elif avg_len < 512:
            return max(self.base_batch_size // 2, 4)
        else:
            return max(self.base_batch_size // 4, 2)

    def _clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


    ######################################
    # Embedding Building
    ######################################

    def get_embeddings(self):
        dense_path = os.path.join(self.data_path, "kure_dense.npy")
        sparse_path = os.path.join(self.data_path, "bge_sparse.pkl")

        # Try loading
        if self._try_load_embeddings(dense_path, sparse_path):
            return

        print("Building embeddings (KURE dense + BGE sparse)...")

        all_dense = []
        all_sparse = []

        i = 0
        pbar = tqdm(total=len(self.contexts), desc="Encoding passages")

        with timer("Encoding passages"):
            while i < len(self.contexts):
                end_idx = min(i + self.base_batch_size, len(self.contexts))
                batch = self.contexts[i:end_idx]

                batch_size = self._get_adaptive_batch_size(batch)

                # Dense: KURE
                if self.use_dense:
                    tok = self.kure_tokenizer(
                        batch, padding=True, truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to("cuda")

                    with torch.no_grad():
                        out = self.kure_model(**tok)
                        dense_vecs = out.last_hidden_state[:, 0, :].cpu().numpy()
                        all_dense.append(dense_vecs.astype(np.float16))

                # Sparse: BGE-M3
                if self.use_sparse:
                    sparse_out = self.sparse_model.encode(
                        batch,
                        batch_size=batch_size,
                        max_length=self.max_length,
                        return_dense=False,
                        return_sparse=True,
                        return_colbert_vecs=False
                    )
                    all_sparse.extend(sparse_out["lexical_weights"])

                i = end_idx
                pbar.update(len(batch))

                if i % (self.base_batch_size * 10) == 0:
                    self._clear_memory()

        pbar.close()

        # Save
        if self.use_dense:
            self.dense_embeddings = np.vstack(all_dense)
            np.save(dense_path, self.dense_embeddings)
            print(f"✅ Saved dense: {self.dense_embeddings.shape}")

        if self.use_sparse:
            import pickle
            self.sparse_embeddings = all_sparse
            with open(sparse_path, "wb") as f:
                pickle.dump(self.sparse_embeddings, f)
            print(f"✅ Saved sparse: {len(self.sparse_embeddings)} passages")

        self._clear_memory()


    def _try_load_embeddings(self, dense_path, sparse_path):
        ok = True

        if self.use_dense:
            if os.path.exists(dense_path):
                with timer("Loading KURE dense"):
                    self.dense_embeddings = np.load(dense_path)
                    print(f"Dense loaded: {self.dense_embeddings.shape}")
            else:
                ok = False

        if self.use_sparse:
            if os.path.exists(sparse_path):
                import pickle
                with timer("Loading BGE sparse"):
                    with open(sparse_path, "rb") as f:
                        self.sparse_embeddings = pickle.load(f)
                    print(f"Sparse loaded: {len(self.sparse_embeddings)} passages")
            else:
                ok = False

        return ok


    ######################################
    # Query Encoding
    ######################################

    def _encode_query(self, query: str):
        # Dense
        q_tok = self.kure_tokenizer(
            [query], padding=True, truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            out = self.kure_model(**q_tok)
            q_dense = out.last_hidden_state[:, 0, :].cpu().numpy()

        # Sparse
        sparse_out = self.sparse_model.encode(
            [query],
            batch_size=1,
            max_length=self.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False
        )

        q_sparse = sparse_out["lexical_weights"][0]

        return q_dense, q_sparse


    ######################################
    # Score computation
    ######################################

    def _compute_dense_score(self, q, p):
        q = q.astype(np.float32)
        p = p.astype(np.float32)
        return np.dot(p, q.T).squeeze()

    def _compute_sparse_score(self, q_weight, p_weights):
        scores = []
        for w in p_weights:
            s = 0.0
            for tid, wq in q_weight.items():
                if tid in w:
                    s += wq * w[tid]
            scores.append(s)
        return np.array(scores)


    ######################################
    # Retrieval
    ######################################

    def get_relevant_doc(
        self,
        query: str,
        k: int = 1,
        weights: Optional[dict] = None,
        use_rerank: bool = None,
        rerank_top_k=100,
    ):

        if weights is None:
            weights = {
                'dense': 0.5 if self.use_dense else 0.0,
                'sparse': 0.5 if self.use_sparse else 0.0,
            }

        q_dense, q_sparse = self._encode_query(query)

        final_scores = np.zeros(len(self.contexts))

        if self.use_dense and weights["dense"] > 0:
            dense_scores = self._compute_dense_score(q_dense, self.dense_embeddings)
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.ptp() + 1e-8)
            final_scores += weights["dense"] * dense_scores

        if self.use_sparse and weights["sparse"] > 0:
            sparse_scores = self._compute_sparse_score(q_sparse, self.sparse_embeddings)
            sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.ptp() + 1e-8)
            final_scores += weights["sparse"] * sparse_scores

        top_init = np.argsort(final_scores)[::-1][:max(k, rerank_top_k)]

        # Rerank
        if use_rerank and self.reranker is not None:
            pairs = [[query, self.contexts[i]] for i in top_init]
            rerank_scores = self.reranker.compute_score(
                pairs,
                batch_size=16,
                max_length=512
            )
            final_idx = [top_init[i] for i in np.argsort(rerank_scores)[::-1][:k]]
            final_scores_out = [rerank_scores[i] for i in np.argsort(rerank_scores)[::-1][:k]]

        else:
            final_idx = top_init[:k]
            final_scores_out = final_scores[final_idx].tolist()

        return final_scores_out, list(final_idx)



    ######################################
    # Bulk retrieval
    ######################################

    def get_relevant_doc_bulk(
        self,
        queries,
        k=1,
        weights=None,
        use_rerank=None,
        rerank_top_k=100
    ):
        scores, indices = [], []
        for q in tqdm(queries, desc="Retrieving"):
            s, idx = self.get_relevant_doc(
                q, k=k, weights=weights,
                use_rerank=use_rerank,
                rerank_top_k=rerank_top_k
            )
            scores.append(s)
            indices.append(idx)
            self._clear_memory()
        return scores, indices


    ######################################
    # Full dataset retrieval (수정됨)
    ######################################

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk=100,
        weights=None,
        use_rerank=None,
        rerank_top_k=100
    ):

        # Single query
        if isinstance(query_or_dataset, str):
            scores, idxs = self.get_relevant_doc(
                query_or_dataset,
                k=topk,
                weights=weights,
                use_rerank=use_rerank,
                rerank_top_k=rerank_top_k
            )

            # ✅ 여러 context를 하나로 합침
            retrieved_context = " ".join([self.contexts[i] for i in idxs])

            rows = [{
                "id": "0",
                "question": query_or_dataset,
                "context": retrieved_context,  # ✅ 합쳐진 context
            }]

            return pd.DataFrame(rows), {}

        # Dataset retrieval
        dataset = query_or_dataset
        questions = dataset["question"]

        with timer(f"Retrieving for {len(questions)} queries"):
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                questions,
                k=topk,
                weights=weights,
                use_rerank=use_rerank,
                rerank_top_k=rerank_top_k
            )

        # Create rows - ✅ 각 질문당 1개 row만 생성
        rows = []
        has_gt = "context" in dataset.features

        correct = 0
        total = len(questions)
        mrr = 0.0

        for i, ex in enumerate(dataset):
            qid = ex["id"]
            qtext = ex["question"]
            original_context = ex.get("context") if has_gt else None

            # ✅ 여러 context를 하나로 합침
            retrieved_contexts = [self.contexts[idx] for idx in doc_indices[i]]
            retrieved_context = " ".join(retrieved_contexts)

            row = {
                "id": qid,
                "question": qtext,
                "context": retrieved_context,  # ✅ 합쳐진 context
            }

            # Accuracy & MRR 계산
            if has_gt and original_context:
                found = False
                for rank, ctx in enumerate(retrieved_contexts):
                    if original_context in ctx or ctx in original_context:
                        if not found:
                            correct += 1
                            mrr += 1.0 / (rank + 1)
                            found = True
                        break

            if "answers" in ex:
                row["answers"] = ex["answers"]

            rows.append(row)  # ✅ 1개 row만 추가

        metrics = {}
        if has_gt:
            metrics = {
                "retrieval_accuracy": correct / total,
                "mrr": mrr / total,
                "correct_count": correct,
                "total_count": total,
                "top_k": topk,
            }

            print("\n" + "="*50)
            print("[Retrieval Metrics]")
            print(f"Accuracy: {metrics['retrieval_accuracy']:.4f}")
            print(f"MRR: {metrics['mrr']:.4f}")
            print("="*50)

        return pd.DataFrame(rows), metrics