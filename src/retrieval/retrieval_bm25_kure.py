##############################################
# Hybrid Retrieval (BM25 Ensemble + KURE Dense + BGE Reranker)
##############################################

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


class BGEM3KUREHybridRetrievalWithBM25Ensemble:
    """
    Hybrid Retrieval:
    - Sparse: BM25 Ensemble (wandb tokenizer + kiwi tokenizer)
    - Dense: KURE-v1 encoder
    - Reranker: BGE-reranker
    """

    def __init__(
        self,
        data_path="data",
        context_path="wikipedia_documents.json",

        # BM25 Ensemble params
        model_name_or_path="bert-base-multilingual-cased",
        bm25_ensemble_method="weighted_sum",
        bm25_alpha=0.7,

        # Dense
        kure_model_name="models/kure_finetuned/encoder",
        dense_embedding_path=None,

        # Reranker
        reranker_name="BAAI/bge-reranker-v2-m3",

        use_fp16=True,
        batch_size=8,
        max_length=512,

        use_dense=True,
        use_sparse=True,
        use_reranker=True,
    ):
        self.data_path = data_path
        self.use_dense = use_dense
        self.use_sparse = use_sparse
        self.use_reranker = use_reranker

        self.max_length = max_length
        self.batch_size = batch_size

        self.bm25_ensemble_method = bm25_ensemble_method
        self.bm25_alpha = bm25_alpha

        ##########################################
        # Load contexts
        ##########################################
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        wiki_items = list(wiki.items())
        self.ids = [k for k, v in wiki_items]
        self.contexts = [v["text"] for k, v in wiki_items]

        print(f"Number of passages: {len(self.contexts)}")

        ##########################################
        # 1) BM25 Ensemble: wandb + kiwi
        ##########################################
        print("Initializing BM25 retrievers...")

        from transformers import AutoTokenizer
        from kiwipiepy import Kiwi
        from .retrieval_bm25_wandb import BM25RetrievalWithMetrics as BM25RetrievalWandb
        from .retrieval_bm25_morphs import BM25RetrievalWithMetrics as BM25RetrievalMorphs

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.retriever_wandb = BM25RetrievalWandb(
            tokenize_fn=tokenizer.tokenize,
            data_path=data_path,
            context_path=context_path
        )
        self.retriever_wandb.get_sparse_embedding()

        kiwi = Kiwi()
        def kiwi_tokenize(text):
            return [token.form for token in kiwi.tokenize(text)]

        self.retriever_morphs = BM25RetrievalMorphs(
            tokenize_fn=kiwi_tokenize,
            data_path=data_path,
            context_path=context_path
        )
        self.retriever_morphs.get_sparse_embedding()

        print("✓ BM25 Ensemble loaded")

        ##########################################
        # 2) KURE Dense
        ##########################################
        if self.use_dense:
            print(f"Loading KURE dense model: {kure_model_name}")
            from transformers import AutoTokenizer, AutoModel
            self.kure_tokenizer = AutoTokenizer.from_pretrained(kure_model_name)
            self.kure_model = AutoModel.from_pretrained(kure_model_name).cuda()

        ##########################################
        # 3) Reranker (BGE)
        ##########################################
        if use_reranker:
            print(f"Loading reranker: {reranker_name}")
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(
                reranker_name, use_fp16=True, device="cuda"
            )
        else:
            self.reranker = None

        self.dense_embeddings = None


    ##########################################
    # BM25 Ensemble
    ##########################################
    def _ensemble_single(self, scores1, indices1, scores2, indices2, topk):
        combined = {}

        if self.bm25_ensemble_method == "weighted_sum":
            def normalize(x):
                if not x: return []
                mn, mx = min(x), max(x)
                if mx == mn: return [1.0] * len(x)
                return [(v - mn) / (mx - mn) for v in x]

            ns1 = normalize(scores1)
            ns2 = normalize(scores2)

            for idx, did in enumerate(indices1):
                combined[did] = combined.get(did, 0) + self.bm25_alpha * ns1[idx]
            for idx, did in enumerate(indices2):
                combined[did] = combined.get(did, 0) + (1 - self.bm25_alpha) * ns2[idx]

        else:  # RRF
            k = 60
            for r, did in enumerate(indices1):
                combined[did] = combined.get(did, 0) + 1/(k+r+1)
            for r, did in enumerate(indices2):
                combined[did] = combined.get(did, 0) + 1/(k+r+1)

        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:topk]
        final_ids = [d for d, _ in sorted_docs]
        final_scores = [s for _, s in sorted_docs]
        return final_scores, final_ids


    ##########################################
    # Build dense embeddings
    ##########################################
    def get_embeddings(self):
        if self.dense_embeddings is not None:
            return

        print("Building dense embeddings...")

        all_dense = []
        i = 0
        pbar = tqdm(total=len(self.contexts))

        while i < len(self.contexts):
            end = min(i + self.batch_size, len(self.contexts))
            batch = self.contexts[i:end]

            tok = self.kure_tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                out = self.kure_model(**tok)
                vec = out.last_hidden_state[:, 0, :].cpu().numpy()

            all_dense.append(vec.astype(np.float16))
            i = end
            pbar.update(len(batch))

        pbar.close()
        self.dense_embeddings = np.vstack(all_dense)
        print("✓ Dense embeddings built:", self.dense_embeddings.shape)


    ##########################################
    # Dense Query
    ##########################################
    def _encode_query_dense(self, query):
        tok = self.kure_tokenizer(
            [query], padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            out = self.kure_model(**tok)
            return out.last_hidden_state[:, 0, :].cpu().numpy()


    ##########################################
    # Retrieval
    ##########################################
    def get_relevant_doc(self, query, k=1, weights=None, use_rerank=None, rerank_top_k=100):

        ######################################
        # 1) BM25 Ensemble retrieval
        ######################################
        search_k = max(k * 2, rerank_top_k)

        s1, idx1 = self.retriever_wandb.get_relevant_doc(query, k=search_k)
        s2, idx2 = self.retriever_morphs.get_relevant_doc(query, k=search_k)

        bm25_scores, bm25_ids = self._ensemble_single(s1, idx1, s2, idx2, topk=search_k)

        ######################################
        # 2) Dense Retrieval
        ######################################
        if self.use_dense:
            if self.dense_embeddings is None:
                self.get_embeddings()

            q_dense = self._encode_query_dense(query)
            dense_scores = np.dot(self.dense_embeddings, q_dense.T).squeeze()

            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.ptp() + 1e-8)
        else:
            dense_scores = np.zeros(len(self.contexts))

        # BM25 → sparse array
        sparse_array = np.zeros(len(self.contexts))
        for i, did in enumerate(bm25_ids):
            sparse_array[did] = bm25_scores[i]

        if weights is None:
            weights = {"dense": 0.5, "sparse": 0.5}

        final_scores = (
            weights["dense"] * dense_scores +
            weights["sparse"] * sparse_array
        )

        ######################################
        # 3) Reranking
        ######################################
        top_init = np.argsort(final_scores)[::-1][:max(k, rerank_top_k)]

        if use_rerank and self.reranker:
            pairs = [[query, self.contexts[i]] for i in top_init]

            r_scores = self.reranker.compute_score(
                pairs, batch_size=16, max_length=512
            )

            order = np.argsort(r_scores)[::-1][:k]
            final_ids = [top_init[i] for i in order]
            final_vals = [r_scores[i] for i in order]

        else:
            final_ids = top_init[:k]
            final_vals = final_scores[final_ids]

        return final_vals, final_ids


    ##########################################
    # Full dataset retrieval (MRC 호환)
    ##########################################
    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk=100,
        weights=None,
        use_rerank=None,
        rerank_top_k=100
    ):

        ######################################
        # 1) 단일 쿼리
        ######################################
        if isinstance(query_or_dataset, str):
            scores, ids = self.get_relevant_doc(
                query_or_dataset,
                k=topk,
                weights=weights,
                use_rerank=use_rerank,
                rerank_top_k=rerank_top_k
            )

            merged = " ".join([self.contexts[i] for i in ids])

            return pd.DataFrame([{
                "id": "0",
                "question": query_or_dataset,
                "context": merged,
                "retrieval_rank": 1,
                "retrieval_score": float(scores[0]) if len(scores) else 0.0,
            }]), {}

        ######################################
        # 2) 전체 Dataset
        ######################################
        dataset = query_or_dataset
        questions = dataset["question"]

        doc_scores = []
        doc_indices = []

        with timer(f"Retrieving for {len(questions)} queries"):
            for q in tqdm(questions, desc="Retrieving"):
                s, idx = self.get_relevant_doc(
                    q,
                    k=topk,
                    weights=weights,
                    use_rerank=use_rerank,
                    rerank_top_k=rerank_top_k
                )
                doc_scores.append(s)
                doc_indices.append(idx)
                torch.cuda.empty_cache()

        ######################################
        # 결과 생성
        ######################################
        rows = []
        has_gt = "context" in dataset.features

        correct = 0
        mrr = 0.0
        total = len(dataset)

        for i, ex in enumerate(dataset):
            qid = ex["id"]
            qtext = ex["question"]

            original_context = ex.get("context") if has_gt else None

            retrieved_list = [self.contexts[d] for d in doc_indices[i]]
            merged = " ".join(retrieved_list)

            row = {
                "id": qid,
                "question": qtext,
                "context": merged,
                "retrieval_rank": 1,
                "retrieval_score": float(doc_scores[i][0]) if len(doc_scores[i]) else 0.0,
            }

            # 평가
            if has_gt:
                found = False
                for rank, ctx in enumerate(retrieved_list):
                    if original_context in ctx or ctx in original_context:
                        correct += 1
                        mrr += 1.0 / (rank + 1)
                        found = True
                        break

            if "answers" in ex:
                row["answers"] = ex["answers"]

            rows.append(row)

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
            print("Hybrid Retrieval Metrics")
            print(f"Accuracy: {metrics['retrieval_accuracy']:.4f}")
            print(f"MRR: {metrics['mrr']:.4f}")
            print("="*50)

        return pd.DataFrame(rows), metrics
