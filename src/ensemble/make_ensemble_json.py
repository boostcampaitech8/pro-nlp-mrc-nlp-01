#############
# 실행 명령어
# python -m src.ensemble.make_ensemble_json \
#     --input_file outputs/temp2/nbest_predictions.json\
#     --run_name test                

# 기본 상위 10개로 불러옴
# 실제 파일명 입력해야함.
####################33



import json
import collections
import argparse
import os
import time
from tqdm import tqdm

# [고정 경로 설정] 사용자가 요청한 디렉토리
FIXED_SAVE_DIR = "/data/ephemeral/git/pro-nlp-mrc-nlp-01/predictions_for_ensemble"

def convert_nbest_to_ensemble(input_path, run_name=None, top_k=10):
    """
    nbest_predictions.json 파일을 읽어서 지정된 고정 경로에 앙상블용 json 포맷으로 저장합니다.
    """
    
    # 1. nbest 파일 로드
    print(f"Loading nbest predictions from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        nbest_data = json.load(f)

    all_ensemble_json = collections.OrderedDict()
    
    print(f"Processing {len(nbest_data)} examples...")

    # 2. 데이터 처리 루프 (확률 합산 및 정렬)
    for example_id, nbest_list in tqdm(nbest_data.items()):
        ensemble_candidates = collections.defaultdict(float)

        for pred in nbest_list:
            if "text" in pred and "probability" in pred:
                ensemble_candidates[pred["text"]] += pred["probability"]
        
        ensemble_list = [
            {"text": k, "probability": v} 
            for k, v in ensemble_candidates.items()
        ]
        
        ensemble_list.sort(key=lambda x: x["probability"], reverse=True)
        final_ensemble_list = ensemble_list[:top_k]
        
        all_ensemble_json[example_id] = final_ensemble_list

    # 3. 저장 경로 및 파일명 결정
    if not os.path.exists(FIXED_SAVE_DIR):
        print(f"Creating directory: {FIXED_SAVE_DIR}")
        os.makedirs(FIXED_SAVE_DIR, exist_ok=True)

    if run_name:
        # 사용자가 run_name을 입력한 경우
        safe_run_name = run_name.replace("/", "_")
        file_name = f"prediction_for_ensemble_{safe_run_name}.json"
    else:
        # run_name이 없는 경우: 입력 파일명이나 타임스탬프를 활용
        base_name = os.path.basename(input_path)
        name_without_ext = os.path.splitext(base_name)[0]
        # "nbest_predictions_" 접두사가 있다면 제거해서 깔끔하게 만듦
        if name_without_ext.startswith("nbest_predictions_"):
            clean_name = name_without_ext.replace("nbest_predictions_", "")
        else:
            clean_name = name_without_ext
        
        file_name = f"prediction_for_ensemble_{clean_name}.json"

    # 최종 저장 경로
    output_path = os.path.join(FIXED_SAVE_DIR, file_name)

    print(f"Saving ensemble json to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(all_ensemble_json, indent=4, ensure_ascii=False) + "\n"
        )
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Convert nbest_predictions to ensemble format in fixed directory")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input nbest_predictions.json file"
    )
    
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=None, 
        help="Name of the run (used for filename). If not provided, input filename is used."
    )
    
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=10, 
        help="Number of candidates to keep (default: 10)"
    )

    args = parser.parse_args()

    convert_nbest_to_ensemble(args.input_file, args.run_name, args.top_k)

if __name__ == "__main__":
    main()