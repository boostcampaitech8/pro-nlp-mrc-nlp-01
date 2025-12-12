import json
import os
from glob import glob
from collections import defaultdict
import argparse
from datasets import load_from_disk
from difflib import SequenceMatcher

# -------------------------------------------------------
# 앙상블할 모델 파일과 가중치 설정
# -------------------------------------------------------
# 형식: (파일명, 가중치)
# 가중치가 높을수록 해당 모델의 예측이 더 큰 영향을 미침
ENSEMBLE_MODELS = [
    ("prediction_for_ensemble_curtis.json", 3.0),
    ("prediction_for_ensemble_hantaek2.json", 2.0),
    ("prediction_for_ensemble_potato.json", 2.0),
    ("prediction_for_ensemble_sunbear.json", 1.0),
    ("prediction_for_ensemble_uomnf97.json", 3.0),
    # ("prediction_for_ensemble_model2.json", 1.0),
    # ("prediction_for_ensemble_model3.json", 0.8),
    # 필요한 모델 추가...
]

# 기본 폴더 경로
DEFAULT_FOLDER = "predictions_for_ensemble"


# -------------------------------------------------------
# 텍스트 정규화 (원본 보존, 비교용 최소 정규화)
# -------------------------------------------------------
def normalize_text(text: str):
    return text.strip()   # 공백 제거 외 변형 없음


# -------------------------------------------------------
# 유사도 판단 (threshold는 0.80 권장)
# -------------------------------------------------------
def is_similar(a: str, b: str, threshold: float = 0.75):
    return SequenceMatcher(None, a, b).ratio() >= threshold


def soft_ensemble(folder_path: str = None,
                  json_out: str = "predictions.json",
                  csv_out: str = "predictions_submit.csv",
                  model_weights: list = None):
    """
    여러 앙상블 JSON 파일에서 text별 probability를 합산하여
    최종 답변 1개 선택 후 JSON, TAB-CSV 두 가지 출력 파일 생성
    
    Args:
        folder_path: JSON 파일들이 있는 폴더 경로 (기본값: predictions_for_ensemble)
        json_out: 출력 JSON 파일명
        csv_out: 출력 CSV 파일명
        model_weights: (파일명, 가중치) 튜플 리스트 (기본값: ENSEMBLE_MODELS)
    """
    
    # 기본값 설정
    if folder_path is None:
        folder_path = DEFAULT_FOLDER
    if model_weights is None:
        model_weights = ENSEMBLE_MODELS

    output_dir = "outputs/ensemble_output"
    os.makedirs(output_dir, exist_ok=True)
    json_out = os.path.join(output_dir, "predictions.json")
    csv_out = os.path.join(output_dir, "predictions_submit.csv")
    
    # 모델 파일 및 가중치 출력
    print("\n========== 앙상블 모델 설정 ==========")
    for idx, (file_name, weight) in enumerate(model_weights):
        file_path = os.path.join(folder_path, file_name)
        exists = "✓" if os.path.exists(file_path) else "✗ (파일 없음)"
        print(f"  [{idx}] {file_name}: 가중치 = {weight} {exists}")
    print("======================================\n")

    all_candidates = defaultdict(list)

    # ------------------------------------------------
    # 1. 지정된 모델 파일들의 후보를 가중치 적용하여 모은다
    # ------------------------------------------------
    for file_name, weight in model_weights:
        file_path = os.path.join(folder_path, file_name)
        
        if not os.path.exists(file_path):
            print(f"[경고] 파일이 존재하지 않아 건너뜀: {file_path}")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            preds = json.load(f)

        for qid, cand_list in preds.items():
            for cand in cand_list:  # cand = {"text": ..., "probability": ...}
                text = cand["text"]
                prob = float(cand["probability"]) * weight  # 가중치 적용
                all_candidates[qid].append((text, prob))
        
        print(f"[로드 완료] {file_name} (가중치: {weight})")

    # ------------------------------------------------
    # 2. 유사도 그룹핑 + 그룹 확률 합산 + 그룹 내 대표 텍스트 선정
    # ------------------------------------------------
    final_json = {}

    for qid, cand_list in all_candidates.items():

        # 2-1. grouping: {정규화된 키: [(원본텍스트, prob), ...]}
        groups = {}

        for text, prob in cand_list:
            norm = normalize_text(text)

            matched = False
            # 기존 그룹 중 유사한 representative 찾기
            for rep_text in groups.keys():
                if is_similar(norm, rep_text):
                    groups[rep_text].append((text, prob))
                    matched = True
                    break

            if not matched:
                groups[norm] = [(text, prob)]

     # 2-2. 그룹별 확률 합산 & 그룹 내 최고 확률 텍스트 선택
        group_scores = {}      # {rep : total_prob}
        group_best_text = {}   # {rep : best_original_text}

        for rep, items in groups.items():
            total_prob = sum(prob for _, prob in items)
            group_scores[rep] = total_prob

            best_original_text = max(items, key=lambda x: x[1])[0]
            group_best_text[rep] = best_original_text

        # 2-3. 가장 점수 높은 그룹 선택
        best_group = max(group_scores.items(), key=lambda x: x[1])[0]

        # 해당 그룹의 최고 확률 텍스트를 최종 답변으로 선정
        final_json[qid] = group_best_text[best_group]


    # ------------------------------------------------
    # 3. JSON output 저장 (형식 1)
    # ------------------------------------------------
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)

    print(f"[완료] JSON 파일 생성 → {json_out}")

    # ------------------------------------------------
    # 4. CSV(TAB) output 저장 (형식 2)
    # ------------------------------------------------
    with open(csv_out, "w", encoding="utf-8") as f:
        for qid, text in final_json.items():
            f.write(f"{qid}\t{text}\n")

    print(f"[완료] CSV(TAB) 파일 생성 → {csv_out}")
    return final_json



def compute_em(pred_dict, val_dataset_path):
    """
    predictions.json(dict)을 validation 데이터의 정답과 비교하여 EM 계산
    """

    dataset = load_from_disk(val_dataset_path)["validation"]

    total = len(dataset)
    correct = 0

    for example in dataset:
        qid = example["id"]

        pred_text = pred_dict.get(qid, "").strip()

        gold_list = example["answers"]["text"]
        if len(gold_list) == 0:
            gold_text = ""
        else:
            gold_text = gold_list[0].strip()

        if pred_text == gold_text:
            correct += 1

    em = correct / total * 100

    print("\n========== EM SCORE ==========")
    print(f"Total examples : {total}")
    print(f"Correct        : {correct}")
    print(f"EM Score       : {em:.2f}%")
    print("================================\n")

    return em


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder_path", type=str, default=DEFAULT_FOLDER,
                        help="앙상블할 JSON 파일들이 있는 폴더 경로")
    parser.add_argument("--json_out", type=str, default="predictions.json")
    parser.add_argument("--csv_out", type=str, default="predictions_submit.csv")
    parser.add_argument("--val_dataset_path", type=str,
                        default="/data/ephemeral/git/pro-nlp-mrc-nlp-01/data/train_dataset")
    parser.add_argument("--compute_em", action="store_true",
                        help="EM 점수 계산 여부")
    args = parser.parse_args()

    # 코드 상단의 ENSEMBLE_MODELS 설정 사용
    pred_dict = soft_ensemble(
        folder_path=args.folder_path, 
        json_out=args.json_out, 
        csv_out=args.csv_out,
        model_weights=ENSEMBLE_MODELS  # 상단에서 정의한 모델/가중치 사용
    )
    
    # EM 계산 수행 (옵션)
    if args.compute_em:
        compute_em(pred_dict, args.val_dataset_path)
