"""
틀린 케이스 JSON 파일에서 ground_truth와 prediction만 남기는 스크립트
"""
import json
from pathlib import Path

# 프로젝트 루트 경로 설정
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent

# 입력 파일 경로
input_file = project_root / "notebooks" / "negative_passage" / "wrong_cases_random_1to1.json"
# 출력 파일 경로 (같은 파일에 덮어쓰기)
output_file = input_file

# 파일 읽기
print(f"파일 읽기 중: {input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ground_truth와 prediction만 남기기
print(f"필터링 중... (총 {len(data)}개 항목)")
filtered_data = [
    {
        "ground_truth": item["ground_truth"],
        "prediction": item["prediction"]
    }
    for item in data
]

# 파일 저장
print(f"파일 저장 중: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"완료! {len(filtered_data)}개 항목이 저장되었습니다.")

