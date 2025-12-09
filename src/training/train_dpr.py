import logging
import os
import os
import sys

# Add project root to sys.path to allow importing from src
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

# DPR 모델을 위한 Bi-Encoder 클래스 정의
class DPREncoder(nn.Module):
    def __init__(self, model_name_or_path, config=None):
        super(DPREncoder, self).__init__()
        if config:
            self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # RoBERTa는 token_type_ids를 사용하지 않을 수 있음
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.pooler_output  # [CLS] token representation


class BiEncoder(nn.Module):
    def __init__(self, model_name_or_path, config=None):
        super(BiEncoder, self).__init__()
        # Question Encoder와 Context Encoder는 독립적인 가중치를 가짐 (DPR 논문 기준)
        # 만약 Siamese Network처럼 공유하고 싶다면 하나만 생성해서 공유하면 됨.
        self.question_encoder = DPREncoder(model_name_or_path, config)
        self.context_encoder = DPREncoder(model_name_or_path, config)

    def forward(
        self,
        q_input_ids,
        q_attention_mask,
        c_input_ids,
        c_attention_mask,
        q_token_type_ids=None,
        c_token_type_ids=None,
        labels=None # Trainer compatibility
    ):
        q_outputs = self.question_encoder(q_input_ids, q_attention_mask, q_token_type_ids)
        c_outputs = self.context_encoder(c_input_ids, c_attention_mask, c_token_type_ids)  # (batch_size, embed_dim)

        # sim_scores: (batch_size, batch_size) if 1-1 mapping
        # If we have hard negatives, c_outputs might be larger than q_outputs.
        # Assuming c_outputs has (batch_size * (1 + num_negatives)) embeddings
        
        sim_scores = torch.matmul(q_outputs, c_outputs.transpose(0, 1)) # (batch_size, num_contexts)

        # Labels
        # If we have [P1, N1, P2, N2, ...], the targets for Q1, Q2, .. are 0, 2, ...
        # If we just have [P1, P2, ...], targets are 0, 1, ...
        
        batch_size = q_outputs.shape[0]
        num_contexts = c_outputs.shape[0]
        
        if labels is None:
            if num_contexts == batch_size:
                target = torch.arange(batch_size, device=sim_scores.device)
            else:
                # Assuming 1 positive per question and rest are hard negatives interleaving or appended?
                # The DataCollator below interleaves: [P1, HN1, P2, HN2, ...]
                # So target indices are 0, 2, 4, ... (stride = num_contexts // batch_size)
                stride = num_contexts // batch_size
                target = torch.arange(0, num_contexts, step=stride, device=sim_scores.device)
        else:
            target = labels
            
        loss = F.cross_entropy(sim_scores, target)

        return SequenceClassifierOutput(loss=loss, logits=sim_scores)


# Dictionary 입력을 받아서 처리하는 Data Collator
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

    print(f"Model is from {model_args.model_name_or_path}")
    print(f"Data is from {data_args.dataset_name}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set seed
    set_seed(training_args.seed)

    # Load Dataset
    datasets = load_from_disk(data_args.dataset_name)
    train_dataset = datasets["train"]
    print(f"Original train dataset size: {len(train_dataset)}")
    print(f"Original features: {train_dataset.features}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Define Model (Bi-Encoder)
    # config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = BiEncoder(model_args.model_name_or_path)

    # Prepare Dataset (Raw Text -> Collator handles tokenization)
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
    print(f"Processed train dataset size: {len(train_dataset)}")
    if len(train_dataset) > 0:
        print(f"Processed sample: {train_dataset[0]}")
    
    # Data Collator
    data_collator = DPRDataCollator(tokenizer, max_seq_length=data_args.max_seq_length)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        
        # Save separate encoders for retrieval later
        if training_args.output_dir is not None:
            q_path = os.path.join(training_args.output_dir, "question_encoder")
            c_path = os.path.join(training_args.output_dir, "context_encoder")
            
            if not os.path.exists(q_path): os.makedirs(q_path)
            if not os.path.exists(c_path): os.makedirs(c_path)
            
            model.question_encoder.model.save_pretrained(q_path)
            model.context_encoder.model.save_pretrained(c_path)
            tokenizer.save_pretrained(q_path)
            tokenizer.save_pretrained(c_path)
            print(f"Encoders saved to {q_path} and {c_path}")

if __name__ == "__main__":
    main()
