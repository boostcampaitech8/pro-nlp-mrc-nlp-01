"""
KURE Fine-tuning with Hard Negative Sampling

Contrastive Learning with In-batch Negatives + Hard Negatives

Usage:
    # Step 1: Mine hard negatives
    python -m src.training.hard_negative_mining \
        --dataset_name data/train_dataset \
        --output_path data/train_with_hard_negatives.json \
        --num_hard_negatives 5 \
        --use_bm25 \
        --use_current_model
    
    # Step 2: Fine-tune KURE
    python -m src.training.train_kure_with_hard_negatives \
        --train_data data/train_with_hard_negatives.json \
        --output_dir models/kure_finetuned_hard_neg \
        --model_name nlpai-lab/KURE-v1 \
        --num_epochs 3 \
        --batch_size 16 \
        --learning_rate 2e-5 \
        --temperature 0.05
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
    get_linear_schedule_with_warmup,
)


@dataclass
class TrainingArguments:
    train_data: str = field(
        metadata={"help": "Path to training data with hard negatives"}
    )
    output_dir: str = field(
        default="models/kure_finetuned_hard_neg",
        metadata={"help": "Output directory for fine-tuned model"}
    )
    model_name: str = field(
        default="nlpai-lab/KURE-v1",
        metadata={"help": "Base KURE model name or path"}
    )
    num_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Training batch size"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate"}
    )
    warmup_steps: int = field(
        default=500,
        metadata={"help": "Number of warmup steps"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature for contrastive loss"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Gradient accumulation steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every N steps"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Evaluate every N steps"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )


class HardNegativeDataset(Dataset):
    """Hard Negative가 포함된 데이터셋"""
    
    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["question"],
            "positive": item["positive"],
            "hard_negatives": item["hard_negatives"],
            "random_negatives": item.get("random_negatives", []),
        }


def collate_fn(batch, tokenizer, max_length=512):
    """
    Batch를 구성합니다.
    
    Returns:
        queries: [batch_size, seq_len]
        positives: [batch_size, seq_len]
        negatives: [batch_size, num_negatives, seq_len]
    """
    queries = [item["query"] for item in batch]
    positives = [item["positive"] for item in batch]
    
    # 모든 negative를 모음
    all_negatives = []
    for item in batch:
        negatives = item["hard_negatives"] + item["random_negatives"]
        all_negatives.append(negatives)
    
    # Tokenize queries
    query_inputs = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Tokenize positives
    positive_inputs = tokenizer(
        positives,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Tokenize negatives (flatten → tokenize → reshape)
    flat_negatives = []
    neg_counts = []
    for negs in all_negatives:
        flat_negatives.extend(negs)
        neg_counts.append(len(negs))
    
    negative_inputs = tokenizer(
        flat_negatives,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        "query_input_ids": query_inputs["input_ids"],
        "query_attention_mask": query_inputs["attention_mask"],
        "positive_input_ids": positive_inputs["input_ids"],
        "positive_attention_mask": positive_inputs["attention_mask"],
        "negative_input_ids": negative_inputs["input_ids"],
        "negative_attention_mask": negative_inputs["attention_mask"],
        "neg_counts": neg_counts,
    }


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss with Hard Negatives
    
    InfoNCE Loss: -log(exp(sim(q,p)/τ) / Σ exp(sim(q,n)/τ))
    """
    
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query_emb: torch.Tensor,  # [batch_size, hidden_dim]
        positive_emb: torch.Tensor,  # [batch_size, hidden_dim]
        negative_emb: torch.Tensor,  # [total_negatives, hidden_dim]
        neg_counts: List[int],  # negative 개수 리스트
    ):
        """
        Args:
            query_emb: Query embeddings
            positive_emb: Positive document embeddings
            negative_emb: Negative document embeddings (flattened)
            neg_counts: Number of negatives per query
        """
        batch_size = query_emb.size(0)
        
        # Normalize embeddings
        query_emb = F.normalize(query_emb, p=2, dim=1)
        positive_emb = F.normalize(positive_emb, p=2, dim=1)
        negative_emb = F.normalize(negative_emb, p=2, dim=1)
        
        # Positive similarity: [batch_size]
        pos_sim = (query_emb * positive_emb).sum(dim=1) / self.temperature
        
        # Negative similarities
        losses = []
        neg_start = 0
        
        for i, neg_count in enumerate(neg_counts):
            # 현재 query의 negative embeddings
            neg_end = neg_start + neg_count
            curr_neg_emb = negative_emb[neg_start:neg_end]  # [neg_count, hidden_dim]
            
            # Query와 negative 간의 similarity
            neg_sim = torch.matmul(
                query_emb[i:i+1],  # [1, hidden_dim]
                curr_neg_emb.T  # [hidden_dim, neg_count]
            ) / self.temperature  # [1, neg_count]
            
            # In-batch negatives도 추가 (다른 sample의 positive를 negative로 사용)
            in_batch_neg_emb = torch.cat([
                positive_emb[:i],
                positive_emb[i+1:]
            ], dim=0)  # [batch_size-1, hidden_dim]
            
            if in_batch_neg_emb.size(0) > 0:
                in_batch_sim = torch.matmul(
                    query_emb[i:i+1],
                    in_batch_neg_emb.T
                ) / self.temperature
                
                # 모든 negative similarity 합치기
                all_neg_sim = torch.cat([neg_sim, in_batch_sim], dim=1)
            else:
                all_neg_sim = neg_sim
            
            # InfoNCE Loss
            logits = torch.cat([
                pos_sim[i:i+1].unsqueeze(0),  # [1, 1]
                all_neg_sim  # [1, num_negatives]
            ], dim=1)  # [1, 1 + num_negatives]
            
            labels = torch.zeros(1, dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)
            losses.append(loss)
            
            neg_start = neg_end
        
        return torch.stack(losses).mean()


def encode(model, input_ids, attention_mask):
    """문서/쿼리를 인코딩합니다."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # [CLS] token embedding
    return outputs.last_hidden_state[:, 0, :]


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    gradient_accumulation_steps=1,
):
    """1 epoch 학습"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(pbar):
        # Move to device
        query_input_ids = batch["query_input_ids"].to(device)
        query_attention_mask = batch["query_attention_mask"].to(device)
        positive_input_ids = batch["positive_input_ids"].to(device)
        positive_attention_mask = batch["positive_attention_mask"].to(device)
        negative_input_ids = batch["negative_input_ids"].to(device)
        negative_attention_mask = batch["negative_attention_mask"].to(device)
        neg_counts = batch["neg_counts"]
        
        # Encode
        query_emb = encode(model, query_input_ids, query_attention_mask)
        positive_emb = encode(model, positive_input_ids, positive_attention_mask)
        negative_emb = encode(model, negative_input_ids, negative_attention_mask)
        
        # Compute loss
        loss = criterion(query_emb, positive_emb, negative_emb, neg_counts)
        loss = loss / gradient_accumulation_steps
        
        # Backward
        loss.backward()
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
    return total_loss / len(dataloader)


def main():
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    print("="*60)
    print("KURE Fine-tuning with Hard Negative Sampling")
    print("="*60)
    print(f"Base Model: {args.model_name}")
    print(f"Training Data: {args.train_data}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print(f"Num Epochs: {args.num_epochs}")
    print("="*60)
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load tokenizer and model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    print("✅ Model loaded")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = HardNegativeDataset(args.train_data)
    print(f"Training examples: {len(dataset)}")
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_length),
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Loss function
    criterion = ContrastiveLoss(temperature=args.temperature)
    
    # Training loop
    print("\nStarting training...")
    best_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*60}")
        
        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            output_path = os.path.join(args.output_dir, "best")
            os.makedirs(output_path, exist_ok=True)
            
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            print(f"✅ Best model saved to {output_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Training completed!")
    print(f"Final model saved to {final_path}")
    print(f"Best model saved to {os.path.join(args.output_dir, 'best')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()