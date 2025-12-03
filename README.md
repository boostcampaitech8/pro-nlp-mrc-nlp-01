# Open-Domain Question Answering (ODQA) Project

P stage 3 대회를 위한 Open-Domain Question Answering 프로젝트입니다.

## 프로젝트 구조

```
.
├── src/                 # 소스 코드 (실제 작업 코드)
│   ├── config/          # 설정 관련
│   ├── data/            # 데이터 처리
│   ├── training/        # 학습 관련
│   ├── inference/       # 추론 관련
│   └── utils/           # 유틸리티
├── basecode/            # 참고용 베이스라인 코드
├── scripts/             # 실행 스크립트
├── data/                # 데이터셋 (gitignore)
├── models/              # 학습된 모델 (gitignore)
├── outputs/             # 출력 결과 (gitignore)
├── checkpoints/         # 체크포인트 (gitignore)
├── cache/               # 캐시 (gitignore)
├── logs/                # 로그 (gitignore)
└── experiments/         # 실험 결과 (gitignore)
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd pro-nlp-mrc-nlp-01
```

### 2. 가상 환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 데이터 준비

데이터는 `data/` 폴더에 위치해야 합니다:
- `data/train_dataset/`: 학습 데이터셋
- `data/test_dataset/`: 테스트 데이터셋
- `data/wikipedia_documents.json`: 위키피디아 문서 집합

## 사용 방법

### 학습

```bash
# 방법 1: Python 모듈로 실행
python -m src.training.train \
  --model_name_or_path klue/bert-base \
  --dataset_name data/train_dataset \
  --output_dir models/train_dataset \
  --do_train \
  --do_eval

# 방법 2: 스크립트 사용
./scripts/train.sh \
  --model_name_or_path klue/bert-base \
  --dataset_name data/train_dataset \
  --output_dir models/train_dataset \
  --do_train \
  --do_eval
```

### 추론

```bash
# 방법 1: Python 모듈로 실행
python -m src.inference.inference \
  --output_dir outputs/test_dataset/ \
  --dataset_name data/test_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_predict

# 방법 2: 스크립트 사용
./scripts/inference.sh \
  --output_dir outputs/test_dataset/ \
  --dataset_name data/test_dataset/ \
  --model_name_or_path models/train_dataset/ \
  --do_predict
```

## 주요 모듈

### src/config
- `ModelArguments`: 모델 관련 인자
- `DataTrainingArguments`: 데이터 및 학습 관련 인자

### src/data
- `SparseRetrieval`: TF-IDF 기반 Sparse Retrieval

### src/training
- `train.py`: 학습 메인 스크립트
- `QuestionAnsweringTrainer`: QA를 위한 커스텀 Trainer

### src/inference
- `inference.py`: 추론 메인 스크립트

### src/utils
- `set_seed`: 시드 고정
- `postprocess_qa_predictions`: QA 예측 후처리
- `check_no_error`: 에러 체크

## 참고

- `basecode/` 폴더는 참고용 베이스라인 코드입니다.
- 실제 작업은 `src/` 폴더에서 진행합니다.
- 자세한 사용법은 `src/README.md`를 참고하세요.

## 라이선스

[라이선스 정보를 여기에 추가하세요]

