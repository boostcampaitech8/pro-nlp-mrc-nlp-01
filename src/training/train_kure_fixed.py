"""
python -m src.training.train_kure_fixed \
    --output_dir models/kure_temp_005 \
    --model_name_or_path nlpai-lab/KURE-v1 \
    --dataset_name data/train_dataset_hn_kure \
    --do_train \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --max_seq_length 384 \
    --temperature 0.05
"""

import logging
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import random
from typing import NoReturn, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    Trainer,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from src.config.arguments import DataTrainingArguments, ModelArguments

logger = logging.getLogger(__name__)

# Bi-Encoder for KURE (Siamese Network with Shared Weights)
class BiEncoder(nn.Module):
    def __init__(self, model_name_or_path, config=None, temperature=0.05):
        super(BiEncoder, self).__init__()
        # Shared Encoder
        if config:
            self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path)
        self.temperature = temperature

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(
        self,
        q_input_ids,
        q_attention_mask,
        c_input_ids,
        c_attention_mask,
        q_token_type_ids=None,
        c_token_type_ids=None,
        labels=None
    ):
        # Question Encoding
        q_outputs = self.model(
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
            token_type_ids=q_token_type_ids
        )
        q_emb = self.mean_pooling(q_outputs, q_attention_mask)

        # Context Encoding
        c_outputs = self.model(
            input_ids=c_input_ids,
            attention_mask=c_attention_mask,
            token_type_ids=c_token_type_ids
        )
        c_emb = self.mean_pooling(c_outputs, c_attention_mask)

        # Similarity Score (Dot Product)
        # Note: If KURE calculates Cosine Similarity in inference, we might want to normalize here.
        # But usually Dot Product is fine for training if we don't enforce unit norm.
        # SBERT usually uses CosineSimilarityLoss -> which normalizes.
        # BiEncoder with In-Batch Negatives (NLL Loss) uses dot product logits.
        
        q_emb = F.normalize(q_emb, p=2, dim=1)
        c_emb = F.normalize(c_emb, p=2, dim=1)

        sim_scores = torch.matmul(q_emb, c_emb.transpose(0, 1))/self.temperature # (B, 2B)

        # Labels setup
        batch_size = q_emb.shape[0]
        num_contexts = c_emb.shape[0]
        
        if labels is None:
            if num_contexts == batch_size:
                target = torch.arange(batch_size, device=sim_scores.device)
            else:
                # Assuming [P1, HN1, P2, HN2, ...]
                stride = num_contexts // batch_size
                target = torch.arange(0, num_contexts, step=stride, device=sim_scores.device)
        else:
            target = labels
            
        loss = F.cross_entropy(sim_scores, target)

        return SequenceClassifierOutput(loss=loss, logits=sim_scores)


class DPRDataCollator:
    def __init__(self, tokenizer, max_seq_length=384):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, features):
        questions = [f["question"] for f in features]
        contexts = []
        for f in features:
            contexts.append(f["context"])
            if "hard_negative_context" in f:
                contexts.append(f["hard_negative_context"])

        # Question Tokenization
        q_batch = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )

        # Context Tokenization
        c_batch = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        batch = {
            "q_input_ids": q_batch["input_ids"],
            "q_attention_mask": q_batch["attention_mask"],
            "c_input_ids": c_batch["input_ids"],
            "c_attention_mask": c_batch["attention_mask"],
        }
        
        if "token_type_ids" in q_batch:
            batch["q_token_type_ids"] = q_batch["token_type_ids"]
        if "token_type_ids" in c_batch:
            batch["c_token_type_ids"] = c_batch["token_type_ids"]
            
        return batch

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    temperature = getattr(model_args, 'temperature', 0.05)
    # Custom DataCollator에서 raw columns을 사용하므로 False로 설정
    training_args.remove_unused_columns = False

    print(f"Model is from {model_args.model_name_or_path}")
    print(f"Data is from {data_args.dataset_name}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    train_dataset = datasets["train"]
    print(f"Train dataset size: {len(train_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    model = BiEncoder(model_args.model_name_or_path, temperature=temperature)

    # Prepare Dataset
    def prepare_features(example):
        features = {
            "question": example["question"],
            "context": example["context"]
        }
        if "hard_negative_context" in example:
            features["hard_negative_context"] = example["hard_negative_context"]
        return features

    train_dataset = train_dataset.map(
        prepare_features,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False
    )
    
    data_collator = DPRDataCollator(tokenizer, max_seq_length=data_args.max_seq_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        
        # Save underlying shared encoder
        if training_args.output_dir is not None:
            encoder_path = os.path.join(training_args.output_dir, "encoder")
            if not os.path.exists(encoder_path):
                os.makedirs(encoder_path)
            
            model.model.save_pretrained(encoder_path)
            tokenizer.save_pretrained(encoder_path)
            print(f"Shared encoder saved to {encoder_path}")

if __name__ == "__main__":
    main()
