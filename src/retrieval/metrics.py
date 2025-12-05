import hashlib
import json
from datetime import datetime

def compute_hit_at_k(doc_indices, ground_truth_ids, k):
    """
    Hit@k (=Recall@k)
    """
    hits = 0
    total = len(ground_truth_ids)
    for retrieved, gt in zip(doc_indices, ground_truth_ids):
        if gt in retrieved[:k]:
            hits += 1
    return hits / total if total > 0 else 0


def compute_mrr_at_k(doc_indices, ground_truth_ids, k):
    """
    MRR@k
    """
    mrr = 0.0
    total = len(ground_truth_ids)

    for retrieved, gt in zip(doc_indices, ground_truth_ids):
        rank = 0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id == gt:
                rank = i + 1  # 1-based rank
                break
        if rank > 0:
            mrr += 1.0 / rank

    return mrr / total if total > 0 else 0


def compute_multi_k_metrics(doc_indices, ground_truth_ids, k_list):
    """
    여러 k를 한 번에 평가
    """
    hit_dict = {}
    mrr_dict = {}
    for k in k_list:
        hit_dict[k] = compute_hit_at_k(doc_indices, ground_truth_ids, k)
        mrr_dict[k] = compute_mrr_at_k(doc_indices, ground_truth_ids, k)
    return hit_dict, mrr_dict


def generate_exp_id(config):
    """
    Exp_ID 자동 생성
    """
    s = json.dumps(config, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8].upper()
    

def detect_approach(retriever):
    # Sparse
    if hasattr(retriever, "tfidfv"):
        return "TF-IDF"
    if hasattr(retriever, "bm25"):
        return "BM25"

    # True Bi-Encoder (dual encoders)
    if hasattr(retriever, "dense_query_encoder") and hasattr(retriever, "dense_passage_encoder"):
        return "Bi-Encoder"

    # Single dense encoder
    if hasattr(retriever, "dense_encoder"):
        return "Dense-Encoder"

    # Cross Encoder
    if hasattr(retriever, "cross_encoder"):
        return "Cross-Encoder"

    return "Unknown"


def detect_category(retriever):
    sparse = hasattr(retriever, "tfidfv") or hasattr(retriever, "bm25")
    dense = (
        hasattr(retriever, "dense_encoder")
        or hasattr(retriever, "dense_query_encoder")
        or hasattr(retriever, "dense_passage_encoder")
    )

    cross = hasattr(retriever, "cross_encoder")

    if sparse and dense:
        return "Hybrid"
    if dense:
        return "Dense"
    if sparse:
        return "Sparse"
    if cross:
        return "Reranker"
    return "Unknown"


def extract_hyperparams(retriever):
    hyper = {}

    # TF-IDF
    if hasattr(retriever, "tfidfv"):
        hyper["ngram_range"] = retriever.tfidfv.ngram_range
        hyper["max_features"] = retriever.tfidfv.max_features

    # BM25
    if hasattr(retriever, "bm25"):
        hyper["k1"] = retriever.bm25.k1
        hyper["b"] = retriever.bm25.b

    # Dense encoder
    if hasattr(retriever, "dense_encoder"):
        hyper["embedding_dim"] = getattr(retriever.dense_encoder, "output_dim", None)

    return hyper


def build_experiment_config(retriever, topk, use_faiss, k_list):
    config = {}

    # Category
    config["Category"] = detect_category(retriever)

    # Approach
    config["Approach"] = detect_approach(retriever)

    # Stage1
    config["Stage1_Method"] = config["Approach"]
    config["Stage1_Model"] = getattr(retriever, "model_name", config["Approach"])
    config["Stage1_TopK"] = topk

    # Stage2 (none by default)
    config["Stage2_Method"] = "None"
    config["Stage2_Model"] = "None"

    # Fusion (none by default)
    config["Fusion_Type"] = "None"
    config["Fusion_Params"] = "-"

    # Backend / Index
    if use_faiss:
        config["Index"] = "FAISS"
    else:
        config["Index"] = "MatrixMul"

    # Hyperparams
    config["Key_Hyperparams"] = extract_hyperparams(retriever)

    # Evaluation Range
    config["Eval_K_list"] = k_list

    # Auto Exp ID
    config["Exp_ID"] = generate_exp_id(config)

    return config


def log_experiment_console(config, hit_dict, mrr_dict):
    print("\n========== Experiment Configuration ==========")
    for key, value in config.items():
        print(f"{key:15}: {value}")
    
    print("\n------------- Retrieval Metrics --------------")
    for k in config["Eval_K_list"]:
        print(f"Hit@{k:<3}: {hit_dict[k]:.4f}")
    for k in config["Eval_K_list"]:
        print(f"MRR@{k:<3}: {mrr_dict[k]:.4f}")
    print("=============================================\n")