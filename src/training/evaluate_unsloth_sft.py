import os
import re
import json
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from tqdm import tqdm
from peft import PeftModel

# -----------------------------
# 텍스트 정규화 & EM/F1
# -----------------------------
def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^0-9a-z가-힣 ]+", "", text)
    return text

def compute_em_f1(pred: str, gold: str):
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)

    # EM
    em = 1 if pred_n == gold_n else 0

    # F1
    pred_tokens = pred_n.split()
    gold_tokens = gold_n.split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return em, 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return em, 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1


# -----------------------------
# ChatML 한 샘플 파싱
# -----------------------------
def parse_chatml(sample_text: str):
    """
    train/val_chatml.json의 text 하나에서
    system, user, assistant(gold) 를 뽑아준다.
    """
    # DOTALL: 줄바꿈 포함해서 매칭
    sys_match = re.search(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>",
        sample_text,
        re.DOTALL,
    )
    user_match = re.search(
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>",
        sample_text,
        re.DOTALL,
    )
    asst_match = re.search(
        r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>",
        sample_text,
        re.DOTALL,
    )

    if not (sys_match and user_match and asst_match):
        raise ValueError("ChatML 포맷이 예상과 다릅니다. system/user/assistant 중 일부 누락.")

    system_text = sys_match.group(1).strip()
    user_text = user_match.group(1).strip()
    gold_answer = asst_match.group(1).strip()

    return system_text, user_text, gold_answer


# -----------------------------
# 메인 평가 함수
# -----------------------------
def main():
    # ===== 0. 기본 설정 =====
    max_seq_length = 2048
    load_in_4bit = True
    base_model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    lora_path = "outputs_qwen3_8b_lora/final"
    val_path = "data/val_chatml.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_TOKEN 환경변수를 먼저 export 하세요.")

    # ===== 1. 모델 & 토크나이저 로드 =====
    print("Loading base model...")
    bfloat16 = is_bfloat16_supported()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        dtype = torch.bfloat16 if bfloat16 else torch.float16,
        token = hf_token,
    )

    # Chat 템플릿 qwen2.5 로 다시 세팅 (train 코드와 동일하게)
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template = "qwen2.5")
    print("Chat template head:", tokenizer.chat_template[:60])

    # LoRA 어댑터 로드
    print("Applying LoRA adapter from:", lora_path)
    model = PeftModel.from_pretrained(
        model,
        lora_path,
    )

    model.eval()
    model.to(device)

    # ===== 2. 검증 데이터 로드 =====
    print("Loading validation dataset from:", val_path)
    dataset = load_dataset("json", data_files=val_path, split="train")

    # ===== 3. 평가 루프 =====
    total_em = 0.0
    total_f1 = 0.0
    results = []

    print("Start evaluating...")
    for idx, item in enumerate(tqdm(dataset)):
        text = item["text"]

        try:
            system_text, user_text, gold_answer = parse_chatml(text)
        except ValueError as e:
            print(f"[WARN] idx={idx} 파싱 실패: {e}")
            continue

        # messages 구조로 다시 만들고, chat_template을 통해 프롬프트 생성
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]

        # messages 기반 템플릿 → Tensor만 반환
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 생성된 토큰에서 입력 길이를 빼서 실제 생성된 답변 부분만 추출
        input_len = input_ids.shape[1]
        gen_ids = output[0][input_len:]
        pred_answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        em, f1 = compute_em_f1(pred_answer, gold_answer)
        total_em += em
        total_f1 += f1

        results.append({
            "index": idx,
            "system": system_text,
            "user": user_text,
            "gold": gold_answer,
            "pred": pred_answer,
            "em": em,
            "f1": f1,
        })

    n = len(results)
    final_em = total_em / n if n > 0 else 0.0
    final_f1 = total_f1 / n if n > 0 else 0.0

    print("\n==============================")
    print(f" #examples : {n}")
    print(f" Final EM  : {final_em:.4f}")
    print(f" Final F1  : {final_f1:.4f}")
    print("==============================\n")

    # ===== 4. 결과 저장 =====
    os.makedirs("eval_outputs", exist_ok=True)
    save_path = "eval_outputs/qwen3_8b_lora_val_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "final_em": final_em,
                "final_f1": final_f1,
                "results": results,
            },
            f,
            ensure_ascii = False,
            indent = 2,
        )

    print(f"Saved detailed results → {save_path}")


if __name__ == "__main__":
    main()
