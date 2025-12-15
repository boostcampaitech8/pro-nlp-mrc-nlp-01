import json
import os

# 파일 경로 설정 (경로가 다르면 수정해주세요)
wiki_path = "/data/ephemeral/git/pro-nlp-mrc-nlp-01/data/wikipedia_documents.json"
# 만약 data 폴더 안에 있다면: "data/wikipedia_documents.json"

if os.path.exists(wiki_path):
    print(f"Loading {wiki_path}...")
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki_data = json.load(f)

    print("-" * 40)
    print(f"1. 데이터 타입: {type(wiki_data)}")
    print(f"2. 총 문서 개수: {len(wiki_data)}")

    # 데이터 샘플 출력
    if isinstance(wiki_data, dict):
        first_key = list(wiki_data.keys())[0]
        print(f"3. 구조: DICTIONARY (Key-Value)")
        print(f"   [샘플 Key]: {first_key}")
        print(f"   [샘플 Value]: {wiki_data[first_key]}")
    elif isinstance(wiki_data, list):
        print(f"3. 구조: LIST")
        print(f"   [샘플 Index 0]: {wiki_data[0]}")
    
    print("-" * 40)
else:
    print(f"파일을 찾을 수 없습니다: {wiki_path}")
    print("경로를 다시 확인해주세요.")