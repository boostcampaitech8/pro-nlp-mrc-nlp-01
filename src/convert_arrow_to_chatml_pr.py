# convert_arrow_to_chatml.py
# HF Arrow dataset → Qwen3 ChatML dataset 변환 스크립트 (train + validation)
import unsloth  

import os
import json
from datasets import load_from_disk

# =======================================
# 0. 설정
# =======================================
INPUT_ARROW_DIR = "data/train_dataset"   # load_from_disk() 경로
TRAIN_JSON_OUT = "data/train_chatml.json"      
VAL_JSON_OUT   = "data/val_chatml.json"         

# 대회 규칙 기반 최적화된 SYSTEM_PROMPT
SYSTEM_PROMPT = """
당신은 Open-Domain Question Answering(ODQA) 시스템의 Reader 모델입니다.
주어진 문맥(context)만을 사용하여 질문에 대한 정답을 정확히 추출해야 합니다.

규칙:
1) 정답은 반드시 문맥에 포함된 표현 그대로 출력합니다.
2) 문맥에 없는 정보는 절대 추론하거나 생성하지 않습니다.
3) 설명, 이유, 분석, 사족을 절대 덧붙이지 않습니다.
4) 출력은 정답 문자열만 단독으로 반환합니다.
5) 하나의 질문에는 하나의 짧은 정답만 출력합니다.
6) EM 기준 평가를 위해 불필요한 공백, 문장부호, 말을 추가하지 않습니다.
""".strip()


# =======================================
# 1. ChatML 템플릿 로딩 (Qwen3)
# =======================================
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template

# 1) 아무 qwen tokenizer 하나 로드
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")  # 아무 Qwen 호환 모델 OK

# 2) 여기에 ChatML 템플릿 적용
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")


# =======================================
# 2. HF Dataset 로딩
# =======================================
print(f"📂 Loading HF dataset from: {INPUT_ARROW_DIR}")

dataset = load_from_disk(INPUT_ARROW_DIR)

# korquad 형식 DatasetDict(train=..., validation=...)
if "train" not in dataset or "validation" not in dataset:
    raise RuntimeError("❌ dataset에 train / validation split이 없습니다.")

train_split = dataset["train"]
val_split   = dataset["validation"]

print(f"🔍 train samples: {len(train_split)}")
print(f"🔍 val samples:   {len(val_split)}")


# =======================================
# 3. 변환 함수
# =======================================
def convert_item(item):
    question = item.get("question", "")
    context  = item.get("context", "")
    answers  = item.get("answers", {})

    # 정답 추출
    answer_text = ""
    if answers and "text" in answers and len(answers["text"]) > 0:
        answer_text = answers["text"][0]

    # 필수 필드 없으면 스킵
    if not question or not context or not answer_text:
        return None

    # ChatML message 구성
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"질문: {question}\n\n문맥: {context}"},
        {"role": "assistant", "content": answer_text},
    ]

    # ChatML 문자열 생성
    chatml_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    return {"text": chatml_text}


# =======================================
# 4. 전체 split 변환
# =======================================
def convert_split(split):
    converted = []
    for item in split:
        out = convert_item(item)
        if out:
            converted.append(out)
    return converted


print("🔄 Converting train split to ChatML...")
train_data = convert_split(train_split)

print("🔄 Converting validation split to ChatML...")
val_data = convert_split(val_split)


print(f"✨ 변환 완료! train={len(train_data)}, val={len(val_data)}")


# =======================================
# 5. JSON 파일 저장
# =======================================
print(f"💾 Saving to {TRAIN_JSON_OUT}")
with open(TRAIN_JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

print(f"💾 Saving to {VAL_JSON_OUT}")
with open(VAL_JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print("🎉 All Done! ChatML 변환이 모두 완료되었습니다.")
