"""
train_qa.json에서 answer 길이가 정확히 20자인 데이터만 필터링하는 스크립트
"""

import json
from pathlib import Path

# 프로젝트 루트 경로 설정
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent

# 입력 파일 경로
input_file = project_root / "notebooks" / "negative_passage" / "train_qa.json"
# 출력 파일 경로
output_file = project_root / "notebooks" / "negative_passage" / "train_qa_answer_length_20.json"

# 파일 읽기
print(f"파일 읽기 중: {input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"총 {len(data)}개 항목 로드 완료")

# answer 길이가 정확히 20자인 항목만 필터링
print("\n필터링 중... (answer 길이 = 20자)")
filtered_data = [
    item for item in data
    if len(item.get('answer', '')) == 20
]

print(f"필터링 완료: {len(filtered_data)}개 항목 (전체의 {len(filtered_data)/len(data)*100:.2f}%)")

# 파일 저장
print(f"\n파일 저장 중: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"✅ 저장 완료!")
print(f"\n요약:")
print(f"  - 원본: {len(data)}개 항목")
print(f"  - 필터링 후: {len(filtered_data)}개 항목")
print(f"  - 저장 위치: {output_file}")
