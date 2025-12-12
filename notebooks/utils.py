"""
Notebooks 공통 유틸리티 함수
"""
import re
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def get_project_root() -> Path:
    """
    프로젝트 루트 경로를 반환합니다.
    
    Returns:
        프로젝트 루트 Path 객체
    """
    # notebooks/utils.py 기준으로 프로젝트 루트 찾기
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return project_root


def setup_project_path(project_root: Optional[Path] = None) -> Path:
    """
    프로젝트 경로를 sys.path에 추가합니다.
    
    Args:
        project_root: 프로젝트 루트 경로 (None이면 자동 탐색)
        
    Returns:
        프로젝트 루트 Path 객체
    """
    if project_root is None:
        project_root = get_project_root()
    
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root


def normalize_answer(text: str) -> str:
    """
    정답 텍스트를 정규화합니다 (비교를 위해).
    
    Args:
        text: 정규화할 텍스트
        
    Returns:
        정규화된 텍스트
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def is_exact_match(prediction: str, ground_truth: str) -> bool:
    """
    예측값과 정답이 정확히 일치하는지 확인합니다.
    
    Args:
        prediction: 예측 텍스트
        ground_truth: 정답 텍스트
        
    Returns:
        Exact Match 여부
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def extract_answer_from_example(example: Dict) -> Optional[str]:
    """
    예시 딕셔너리에서 정답 텍스트를 추출합니다.
    
    Args:
        example: 데이터셋 예시 딕셔너리
        
    Returns:
        정답 텍스트 (없으면 None)
    """
    answers = example.get('answers', {})
    
    # answers가 딕셔너리인 경우
    if isinstance(answers, dict):
        answer_texts = answers.get('text', [])
    # answers가 리스트인 경우
    elif isinstance(answers, list):
        answer_texts = answers
    else:
        answer_texts = []
    
    # 첫 번째 정답 반환
    if answer_texts and len(answer_texts) > 0:
        return answer_texts[0].strip()
    
    return None


def find_checkpoint_directory(model_path: Path) -> Path:
    """
    모델 경로에서 checkpoint 디렉토리를 자동으로 찾습니다.
    
    Args:
        model_path: 모델 경로
        
    Returns:
        checkpoint 경로 또는 원본 경로
    """
    if model_path.is_dir():
        checkpoint_dirs = [
            d for d in model_path.iterdir()
            if d.is_dir() and d.name.startswith('checkpoint-')
        ]
        if checkpoint_dirs:
            # 가장 최근 checkpoint 사용 (숫자가 큰 것)
            checkpoint_dirs.sort(
                key=lambda x: int(x.name.split('-')[1]) if x.name.split('-')[1].isdigit() else 0,
                reverse=True
            )
            return checkpoint_dirs[0]
    
    return model_path


def setup_venv_path(project_root: Path) -> bool:
    """
    가상환경(.venv) 경로를 sys.path에 추가합니다.
    
    Args:
        project_root: 프로젝트 루트 경로
        
    Returns:
        가상환경 추가 성공 여부
    """
    import sys
    
    venv_path = project_root / ".venv"
    
    if not venv_path.exists():
        return False
    
    # Python 버전에 맞는 site-packages 경로 찾기
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    venv_site_packages = venv_path / "lib" / f"python{python_version}" / "site-packages"
    
    # site-packages가 없으면 다른 가능한 경로 시도
    if not venv_site_packages.exists():
        # lib/python3.x/site-packages 경로들 확인
        lib_dir = venv_path / "lib"
        if lib_dir.exists():
            for py_dir in lib_dir.iterdir():
                if py_dir.is_dir() and py_dir.name.startswith("python"):
                    site_packages = py_dir / "site-packages"
                    if site_packages.exists():
                        venv_site_packages = site_packages
                        break
    
    if venv_site_packages.exists():
        venv_path_str = str(venv_site_packages)
        if venv_path_str not in sys.path:
            sys.path.insert(0, venv_path_str)
        return True
    
    return False


def setup_notebook_environment(
    notebook_path: Optional[Path] = None,
    add_to_path: bool = True,
    setup_venv: bool = True
) -> Dict[str, Path]:
    """
    노트북 실행 환경을 설정합니다.
    
    Args:
        notebook_path: 현재 노트북 파일 경로 (None이면 자동 탐색)
        add_to_path: sys.path에 프로젝트 루트 추가 여부
        setup_venv: 가상환경(.venv) 경로 추가 여부
        
    Returns:
        경로 딕셔너리: {
            'project_root': 프로젝트 루트,
            'data_dir': 데이터 디렉토리,
            'notebook_dir': 노트북 디렉토리,
            'venv_path': 가상환경 경로 (존재하는 경우)
        }
    """
    # 프로젝트 루트 설정
    if notebook_path is None:
        # 노트북 실행 시 현재 작업 디렉토리에서 프로젝트 루트 찾기
        current_dir = Path(os.getcwd())
        project_root = get_project_root()
    else:
        # 노트북 파일 경로 기준으로 프로젝트 루트 찾기
        project_root = notebook_path.parent.parent
    
    if add_to_path:
        setup_project_path(project_root)
    
    # 가상환경 설정
    venv_path = None
    if setup_venv:
        venv_path = project_root / ".venv"
        if setup_venv_path(project_root):
            print(f"✅ 가상환경(.venv) 경로 추가됨")
        elif venv_path.exists():
            print(f"⚠️ 가상환경(.venv)이 존재하지만 site-packages를 찾을 수 없습니다")
        else:
            print(f"ℹ️ 가상환경(.venv)이 없습니다. 시스템 Python을 사용합니다")
    
    # 경로 설정
    data_dir = project_root / "data"
    notebook_dir = project_root / "notebooks"
    
    result = {
        'project_root': project_root,
        'data_dir': data_dir,
        'notebook_dir': notebook_dir
    }
    
    if venv_path and venv_path.exists():
        result['venv_path'] = venv_path
    
    return result


def load_dataset_safely(
    dataset_path: Path,
    split: Optional[str] = None
) -> Any:
    """
    데이터셋을 안전하게 로드합니다.
    
    Args:
        dataset_path: 데이터셋 경로
        split: 로드할 split 이름 (None이면 전체)
        
    Returns:
        데이터셋 객체
    """
    from datasets import load_from_disk
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {dataset_path}")
    
    dataset = load_from_disk(str(dataset_path))
    
    if split:
        if split not in dataset:
            raise ValueError(f"Split '{split}'을 찾을 수 없습니다. 사용 가능한 splits: {list(dataset.keys())}")
        return dataset[split]
    
    return dataset


def load_json_safely(json_path: Path) -> Dict:
    """
    JSON 파일을 안전하게 로드합니다.
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        JSON 데이터 딕셔너리
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

