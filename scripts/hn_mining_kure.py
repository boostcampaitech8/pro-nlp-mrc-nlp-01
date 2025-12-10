import json
import os
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from tqdm.auto import tqdm

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModel,
)

# BM25 (선택)
from rank_bm25 import BM25Okapi

# 프로젝트 경로
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# 코드1의 최적화된 dense retriever
from src.retrieval.retrieval_kure import KURERetrieval



@dataclass
class HardNegativeArguments:
    dataset_name: str = field(
        metadata={"help": "Training dataset path"}
    )
    output_path: str = field(
        default="data/train_with_hard_negatives.json",
        metadata={"help": "Output path for dataset with hard negatives"}
    )
    data_path: str = field(
        default="data",
        metadata={"help": "Data directory for context & embeddings"}
    )
    context_path: str = field(
        default="wikipedia_documents.json",
        metadata={"help": "Wikipedia documents path"}
    )
    kure_model_path: str = field(
        default="nlpai-lab/KURE-v1",
        metadata={"help": "KURE model path (base or fine-tuned)"}
    )
    num_hard_negatives: int = field(
        default=5,
        metadata={"help": "Number of hard negatives per positive example"}
    )
    num_candidates: int = field(
        default=100,
        metadata={"help": "Number of candidates to retrieve for mining"}
    )
    use_bm25: bool = field(
        default=True,
        metadata={"help": "Use BM25 for candidate search"}
    )
    use_dense: bool = field(
        default=True,
        metadata={"help": "Use Dense(KURE) for candidate search"}
    )
    random_negatives: int = field(
        default=2,
        metadata={"help": "Additional random negatives"}
    )




class HybridHardNegativeMiner:
    """
    Hard Negative Miner optimized with:
    - KURE Dense Retriever (FAISS)  ← 코드1 엔진 사용
    - BM25 (optional)
    - GPU memory minimal usage
    """

    def __init__(self, args, contexts: List[str]):
        self.args = args
        self.contexts = contexts

        # ------------------------
        # BM25 Initializing
        # ------------------------
        if args.use_bm25:
            print("Initializing BM25...")
            tokenized_corpus = [doc.split() for doc in contexts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print("✔ BM25 initialized")

        # ------------------------
        # Dense Retriever (KURE) - FAISS + caching
        # ------------------------
        if args.use_dense:
            print("Initializing KURE Retriever...")
            self.retriever = KURERetrieval(
                data_path=args.data_path,
                context_path=args.context_path,
                model_name=args.kure_model_path,
            )

            # 문서 embedding 로드 or 계산
            self.retriever.get_dense_embedding()

            # FAISS 인덱스 생성
            self.retriever.build_faiss()
            print("✔ Dense retriever ready")

    # -------------------------------------------------------------
    # Query encoding은 retriever 내부에서 GPU로 수행
    # BM25는 CPU 기반
    # -------------------------------------------------------------
    def dense_candidates(self, query):
        scores, indices = self.retriever.get_relevant_doc(query, k=self.args.num_candidates)
        return indices.tolist()

    def bm25_candidates(self, query):
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        topk = np.argsort(scores)[::-1][:self.args.num_candidates]
        return topk.tolist()

    # -------------------------------------------------------------
    # Hard Negative Mining
    # -------------------------------------------------------------
    def mine(self, query, positive_context, answers):
        candidate_set = set()

        # BM25
        if self.args.use_bm25:
            candidate_set.update(self.bm25_candidates(query))

        # Dense (FAISS)
        if self.args.use_dense:
            candidate_set.update(self.dense_candidates(query))

        hard_negatives = []

        for idx in candidate_set:
            ctx = self.contexts[idx]

            # positive 제외
            if ctx == positive_context:
                continue

            # answer 포함 문서 제외
            if any(a in ctx for a in answers):
                continue

            hard_negatives.append(ctx)
            if len(hard_negatives) >= self.args.num_hard_negatives:
                break

        # 부족하면 random negative 추가
        while len(hard_negatives) < self.args.num_hard_negatives:
            rnd = random.choice(self.contexts)
            if rnd != positive_context:
                hard_negatives.append(rnd)

        return hard_negatives




def main():
    parser = HfArgumentParser(HardNegativeArguments)
    args = parser.parse_args_into_dataclasses()[0]

    print("="*60)
    print("🚀 Hard Negative Mining (Optimized KURE + BM25)")
    print("="*60)
    print(f"Dataset: {args.dataset_name}")
    print(f"KURE Model: {args.kure_model_path}")
    print(f"HN per example: {args.num_hard_negatives}")
    print(f"Use BM25: {args.use_bm25}")
    print(f"Use Dense (FAISS): {args.use_dense}")
    print("="*60)

    # Load HF dataset
    dataset = load_from_disk(args.dataset_name)
    train_data = dataset["train"]

    # Load Wikipedia docs
    with open(os.path.join(args.data_path, args.context_path), "r", encoding="utf-8") as f:
        wiki = json.load(f)
    contexts = [v["text"] for v in wiki.values()]

    miner = HybridHardNegativeMiner(args, contexts)

    print("\nMining hard negatives...\n")
    results = []

    for example in tqdm(train_data):
        query = example["question"]
        pos = example["context"]
        answers = example["answers"]["text"]

        hard_negs = miner.mine(query, pos, answers)

        # random negatives 추가
        random_negs = []
        for _ in range(args.random_negatives):
            rnd = random.choice(contexts)
            if rnd != pos:
                random_negs.append(rnd)

        results.append({
            "id": example["id"],
            "question": query,
            "positive": pos,
            "hard_negatives": hard_negs,
            "random_negatives": random_negs,
            "answers": example.get("answers", {}),
        })

    # Save
    print(f"\nSaving → {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n🎉 Done! Hard negatives mined successfully.")
    print("="*60)



if __name__ == "__main__":
    main()
