"""설정 관련 모듈 - 모델 및 데이터 학습 인자 정의."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """모델 관련 인자 클래스."""

    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    retriever_name_or_path: Optional[str] = field(
        default="outputs/dpr_test",
        metadata={
            "help": "Path to pretrained retriever model or model identifier from huggingface.co/models"
        },
    )

@dataclass
class DataTrainingArguments:
    """데이터 및 학습 관련 인자 클래스."""

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=100,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    use_wandb: bool = field(
        default=False, metadata={"help": "Whether to use Weights & Biases for logging"}
    )
    wandb_project: str = field(
        default="retrieval", metadata={"help": "WandB project name"}
    )
    wandb_run_name: str = field(
        default="run", metadata={"help": "WandB run name"}
    )
    context_file: str = field(
        default="wikipedia_documents.json", metadata={"help": "Path to context file"}
    )

    bge_use_reranker: bool = field(
        default=False, metadata={"help": "Use BGE reranker after retrieval"}
    )
    bge_rerank_top_k: int = field(
        default=30, metadata={"help": "Number of top passages to keep after reranking"}
    )
    bge_batch_size: int = field(
        default=8, metadata={"help": "Batch size for reranking"}
    )

    bge_use_dense: bool = field(
        default=True, metadata={"help": "Use dense retrieval"}
    )
    bge_use_sparse: bool = field(
        default=True, metadata={"help": "Use sparse retrieval"}
    )
    bge_use_colbert: bool = field(
        default=False, metadata={"help": "Use colbert retrieval"}
    )
    bge_dense_weight: float = field(
        default=0.5, metadata={"help": "Weight for dense score"}
    )
    bge_sparse_weight: float = field(
        default=0.5, metadata={"help": "Weight for sparse score"}
    )
    bge_colbert_weight: float = field(
        default=0.0, metadata={"help": "Weight for colbert score"}
    )
    bge_max_length: int = field(
        default=512, metadata={"help": "Max length for BGE"}
    )

    sparse_embedding_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to sparse embedding pickle file (optional override)"}
    )

    dense_embedding_path: Optional[str] = field(
        default=None, metadata={"help": "Path to save/load dense embeddings"}
    )
    temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature for scaling similarity scores"}
    )

    # Retrieval Caching
    save_retrieval_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save retrieval results for caching (e.g., data/cache/retrieval_k5.json)"}
    )
    load_retrieval_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to load cached retrieval results (skip retrieval if provided)"}
    )