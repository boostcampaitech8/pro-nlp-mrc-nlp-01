import os
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset


def main():
    # ======================================
    # 0. 기본 설정
    # ======================================
    max_seq_length = 2048
    load_in_4bit = True
    model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_TOKEN 환경변수를 먼저 export 하세요.")

    # ======================================
    # 1. 모델 로드
    # ======================================
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        token = hf_token,
    )


    # ======================================
    # 1.5 ChatML 템플릿 적용 (★ 여기 넣어야 함)
    # ======================================
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen2.5")
    print("Chat template head:", tokenizer.chat_template[:60])


    # ======================================
    # 2. LoRA 적용
    # ======================================
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0.0,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
    )

    # ======================================
    # 3. 데이터 로드
    # ======================================
    train_path = "data/train_chatml.json"
    dataset = load_dataset("json", data_files=train_path, split="train")
    # ======================================
    # 4. TrainingArguments
    # ======================================
    bfloat16 = is_bfloat16_supported()

    training_args = TrainingArguments(
        output_dir = "./outputs_qwen3_8b_lora",
        num_train_epochs = 1,

        gradient_checkpointing = True, 
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        learning_rate = 2e-4,

        logging_steps = 10,
        logging_first_step = True,        # ★ wandb 첫 스텝부터 로깅
        warmup_ratio = 0.03,              # ★ 학습 안정성 UP
        dataloader_num_workers = 2,       # ★ 학습 속도 증가

        save_strategy = "no",
        save_total_limit = None,   # 의미 없어짐

        fp16 = not bfloat16,
        bf16 = bfloat16,

        optim = "paged_adamw_32bit",
        report_to = ["wandb"],     # ← 여기서 wandb 로그 기록됨
    )

    # ======================================
    # 5. Trainer
    # ======================================
    trainer = SFTTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        packing = False,
    )

    # ======================================
    # 6. Train
    # ======================================
    trainer.train()

    # ======================================
    # 7. 최종 저장
    # ======================================
    save_dir = "./outputs_qwen3_8b_lora/final"

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
