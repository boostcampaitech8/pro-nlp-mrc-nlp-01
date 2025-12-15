"""
틀린 케이스 JSON 파일에서 ground_truth와 prediction만 남기는 스크립트
"""
import json
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트 경로 설정
from notebooks.utils import setup_project_path

project_root = setup_project_path()

# 입력 파일 경로
input_file = project_root / "notebooks" / "negative_passage" / "wrong_cases_random_1to1.json"
# 출력 파일 경로 (같은 파일에 덮어쓰기)
output_file = input_file

# 파일 읽기
print(f"파일 읽기 중: {input_file}")
if not input_file.exists():
    print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
    exit(1)

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data: List[Dict] = json.load(f)
except Exception as e:
    print(f"❌ 파일 읽기 실패: {e}")
    exit(1)

# ground_truth와 prediction만 남기기
print(f"필터링 중... (총 {len(data)}개 항목)")
filtered_data: List[Dict[str, str]] = [
    {
        "ground_truth": item.get("ground_truth", ""),
        "prediction": item.get("prediction", "")
    }
    for item in data
    if "ground_truth" in item and "prediction" in item
]

# 파일 저장
print(f"파일 저장 중: {output_file}")
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 완료! {len(filtered_data)}개 항목이 저장되었습니다.")
except Exception as e:
    print(f"❌ 파일 저장 실패: {e}")
    exit(1)

