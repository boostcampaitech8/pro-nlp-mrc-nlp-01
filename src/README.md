# Source Code

Open-Domain Question Answering (ODQA) 프로젝트의 실제 작업용 소스 코드입니다.

## 폴더 구조

```
src/
├── config/              # 설정 관련
│   ├── __init__.py
│   └── arguments.py     # 모델 및 데이터 학습 인자 정의
├── retrieval/           # 검색/리트리벌 관련
│   ├── __init__.py
│   └── retrieval.py     # Sparse Retrieval 구현 (TF-IDF 기반)
├── training/            # 학습 관련
│   ├── __init__.py
│   ├── train.py         # 모델 학습 스크립트
│   └── trainer_qa.py    # Question Answering을 위한 커스텀 Trainer
├── inference/           # 추론 관련
│   ├── __init__.py
│   └── inference.py     # 모델 추론 스크립트
├── utils/               # 유틸리티 함수
│   ├── __init__.py
│   └── utils_qa.py      # 전처리 및 후처리 유틸리티 함수
└── requirements.txt     # 필요한 패키지 목록 (루트의 requirements.txt와 동일)
```

## 사용 방법

**주의**: 모든 명령은 프로젝트 루트 디렉토리(`/data/ephemeral/git/pro-nlp-mrc-nlp-01`)에서 실행해야 합니다.

### 학습

#### 방법 1: Python 모듈로 실행
```bash
# 프로젝트 루트에서 실행
python -m src.training.train \
  --model_name_or_path klue/bert-base \
  --dataset_name data/train_dataset \
  --output_dir models/train_dataset \
  --do_train \
  --do_eval
```

#### 방법 2: 스크립트 사용   
```bash
# 프로젝트 루트에서 실행
./scripts/train.sh \
  --model_name_or_path klue/bert-base \
  --dataset_name data/train_dataset \
  --output_dir models/train_dataset \
  --do_train \
  --do_eval
```

### 추론

**주의**: 프로젝트 루트에서 실행하세요.

#### 방법 1: Python 모듈로 실행
```bash
# 프로젝트 루트에서 실행
python -m src.inference.inference \
  --output_dir outputs/test_dataset/ \
  --dataset_name data/test_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_predict
```

#### 방법 2: 스크립트 사용
```bash
# 프로젝트 루트에서 실행
./scripts/inference.sh \
  --output_dir outputs/test_dataset/ \
  --dataset_name data/test_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_predict
```

## 모듈 구조

### config
- `ModelArguments`: 모델 관련 인자 (모델 경로, config, tokenizer 등)
- `DataTrainingArguments`: 데이터 및 학습 관련 인자

### retrieval
- `SparseRetrieval`: TF-IDF 기반 Sparse Retrieval 클래스

### training
- `train.py`: 학습 메인 스크립트
- `QuestionAnsweringTrainer`: QA를 위한 커스텀 Trainer 클래스

### inference
- `inference.py`: 추론 메인 스크립트

### utils
- `set_seed`: 시드 고정 함수
- `postprocess_qa_predictions`: QA 예측 후처리 함수
- `check_no_error`: 에러 체크 함수

## 참고

- `basecode/` 폴더는 참고용 베이스 코드입니다.
- 실제 작업은 이 `src/` 폴더에서 진행합니다.
- `scripts/` 폴더에 실행 스크립트가 있습니다.
 