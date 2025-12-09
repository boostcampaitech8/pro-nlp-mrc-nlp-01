"""
Python 스크립트를 노트북으로 변환하는 스크립트
"""
import json
from pathlib import Path

# 기존 노트북 메타데이터 구조 참고
metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'codemirror_mode': {'name': 'ipython', 'version': 3},
        'file_extension': '.py',
        'mimetype': 'text/x-python',
        'name': 'python',
        'nbconvert_exporter': 'python',
        'pygments_lexer': 'ipython3',
        'version': '3.10.19'
    }
}

# Python 스크립트 읽기
script_path = Path(__file__).parent / 'create_random_negative_1to1.py'
with open(script_path, 'r', encoding='utf-8') as f:
    script_content = f.read()

# 스크립트를 셀로 분할
cells = []

# 첫 번째 마크다운 셀
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '# 7. Random Negative Passage 1:1 비율 생성\n',
        '\n',
        '본 노트북은 **1:1 비율**로 Random Negative Passage를 생성합니다.\n',
        '\n',
        '**목표:**\n',
        '- 각 Positive 예시마다 정확히 1개의 Random Negative Passage 생성\n',
        '- Positive:Negative = 1:1 비율 유지\n',
        '- 생성된 데이터셋을 저장하여 재사용 가능하도록 구성\n'
    ]
})

# 환경 설정 셀
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## 7.1. 환경 설정 및 라이브러리 Import\n']
})

# 함수 정의 부분 (if __name__ 이전)
main_start = script_content.find('if __name__ == "__main__":')
if main_start > 0:
    function_part = script_content[:main_start].strip()
    # docstring 제거
    if function_part.startswith('"""'):
        doc_end = function_part.find('"""', 3)
        if doc_end > 0:
            function_part = function_part[doc_end + 3:].strip()
    
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': function_part.split('\n')
    })
    
    # 메인 실행 부분
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## 7.2. 데이터셋 생성 및 저장\n']
    })
    
    main_part = script_content[main_start:].strip()
    # if __name__ == "__main__": 제거
    main_part = main_part.replace('if __name__ == "__main__":', '').strip()
    # 들여쓰기 제거
    main_lines = main_part.split('\n')
    main_lines = [line[4:] if line.startswith('    ') else line for line in main_lines if line.strip()]
    
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': main_lines
    })

# 노트북 생성
notebook = {
    'cells': cells,
    'metadata': metadata,
    'nbformat': 4,
    'nbformat_minor': 2
}

# 저장
output_path = Path(__file__).parent / '07_create_random_negative_1to1.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'노트북 생성 완료: {output_path}')

