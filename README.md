# Open-Domain Question Answering (ODQA) Project

P stage 3 대회를 위한 Open-Domain Question Answering 프로젝트입니다. 이 프로젝트는 질문에 대한 답변을 찾기 위해 문서 검색(Retrieval)과 기계 독해(Machine Reading Comprehension) 두 단계로 구성됩니다.

## 📋 목차

- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [주요 모듈](#주요-모듈)
- [데이터 형식](#데이터-형식)
- [실험 및 평가](#실험-및-평가)
- [참고 자료](#참고-자료)

## 📁 프로젝트 구조

```
.
├── src/                    # 소스 코드
│   ├── config/             # 설정 관련
│   │   ├── __init__.py
│   │   └── arguments.py    # 모델 및 데이터 학습 인자 정의
│   ├── retrieval/          # 검색/리트리벌
│   │   ├── __init__.py
│   │   ├── retrieval.py    # Sparse Retrieval (TF-IDF 기반)
│   │   ├── metrics.py      # 검색 평가 메트릭
│   │   └── reranker.py     # 리랭커 구현
│   ├── training/           # 학습 관련
│   │   ├── __init__.py
│   │   ├── train.py        # 모델 학습 메인 스크립트
│   │   ├── trainer_qa.py   # QA를 위한 커스텀 Trainer
│   │   └── eval.py         # 평가 스크립트
│   ├── inference/          # 추론 관련
│   │   ├── __init__.py
│   │   └── inference.py    # 모델 추론 메인 스크립트
│   └── utils/              # 유틸리티
│       ├── __init__.py
│       └── utils_qa.py     # 전처리 및 후처리 함수
├── scripts/                # 실행 스크립트
│   ├── train.sh            # 학습 스크립트
│   ├── inference.sh        # 추론 스크립트
│   └── run_all.sh          # 전체 파이프라인 실행
├── notebooks/              # Jupyter 노트북
│   ├── eda/                # 탐색적 데이터 분석
│   ├── data_agument/       # 데이터 증강
│   └── negative_passage/   # 네거티브 샘플링
├── configs/                # 설정 파일
│   └── sweep_config.yaml   # 하이퍼파라미터 스윕 설정
├── requirements.txt        # Python 패키지 의존성
└── README.md              # 프로젝트 문서
```

## 🚀 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd pro-nlp-mrc-nlp-01
```

### 2. 가상 환경 생성 및 활성화

```bash
# 가상 환경 생성
python -m venv .venv

# 활성화 (Linux/Mac)
source .venv/bin/activate

# 활성화 (Windows)
.venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

주요 패키지:
- `torch`: PyTorch 딥러닝 프레임워크
- `transformers`: Hugging Face Transformers 라이브러리
- `datasets`: 데이터셋 로딩 및 처리
- `faiss-gpu`: 효율적인 벡터 검색 (GPU 지원)
- `sentence-transformers`: 문장 임베딩
- `rank_bm25`: BM25 검색 알고리즘
- `kiwipiepy`: 한국어 형태소 분석기
- `wandb`: 실험 추적 및 로깅

### 4. 데이터 준비

데이터는 `data/` 폴더에 위치해야 합니다:

```
data/
├── train_dataset/          # 학습 데이터셋 (Hugging Face datasets 형식)
├── test_dataset/           # 테스트 데이터셋
└── wikipedia_documents.json # 위키피디아 문서 집합
```

## 💻 사용 방법

### 학습

#### 기본 학습

```bash
python -m src.training.train \
  --model_name_or_path klue/bert-base \
  --dataset_name data/train_dataset \
  --output_dir models/my_model \
  --do_train \
  --do_eval \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --warmup_steps 500 \
  --save_strategy epoch \
  --evaluation_strategy epoch
```

#### 스크립트 사용

```bash
./scripts/train.sh \
  --model_name_or_path klue/bert-base \
  --dataset_name data/train_dataset \
  --output_dir models/my_model \
  --do_train \
  --do_eval
```

#### WandB 로깅 활성화

```bash
python -m src.training.train \
  --model_name_or_path klue/bert-base \
  --dataset_name data/train_dataset \
  --output_dir models/my_model \
  --do_train \
  --do_eval \
  --report_to wandb \
  --run_name my_experiment
```

### 추론

#### 기본 추론

```bash
python -m src.inference.inference \
  --model_name_or_path models/my_model \
  --dataset_name data/test_dataset \
  --output_dir outputs/test_results \
  --do_predict
```

#### Retrieval 포함 추론

```bash
python -m src.inference.inference \
  --model_name_or_path models/my_model \
  --dataset_name data/test_dataset \
  --output_dir outputs/test_results \
  --do_predict \
  --eval_retrieval \
  --top_k_retrieval 100 \
  --use_faiss
```

### Retrieval만 실행

```bash
python -m src.retrieval.retrieval \
  --dataset_name data/train_dataset \
  --model_name_or_path klue/bert-base \
  --data_path data \
  --context_path wikipedia_documents.json \
  --use_faiss
```

## 🔧 주요 모듈

### config

- **ModelArguments**: 모델 관련 설정
  - `model_name_or_path`: 모델 경로 또는 Hugging Face 모델 ID
  - `config_name`: 설정 파일 경로 (선택)
  - `tokenizer_name`: 토크나이저 경로 (선택)
  - `retriever_name_or_path`: 리트리버 모델 경로

- **DataTrainingArguments**: 데이터 및 학습 관련 설정
  - `dataset_name`: 데이터셋 경로
  - `max_seq_length`: 최대 시퀀스 길이 (기본값: 384)
  - `doc_stride`: 문서 분할 시 스트라이드 (기본값: 128)
  - `max_answer_length`: 최대 답변 길이 (기본값: 30)
  - `top_k_retrieval`: 검색할 상위 k개 문서 (기본값: 100)
  - `use_faiss`: FAISS 인덱스 사용 여부

### retrieval

- **SparseRetrieval**: TF-IDF 기반 Sparse Retrieval 클래스
  - `get_sparse_embedding()`: TF-IDF 임베딩 생성/로드
  - `build_faiss()`: FAISS 인덱스 생성
  - `retrieve()`: 검색 수행
  - `retrieve_faiss()`: FAISS를 사용한 검색

### training

- **train.py**: 모델 학습 메인 스크립트
  - 데이터 로딩 및 전처리
  - 모델 초기화
  - 학습 및 평가 실행

- **QuestionAnsweringTrainer**: QA를 위한 커스텀 Trainer
  - 표준 Trainer를 확장하여 QA 작업에 특화
  - 예측 후처리 및 평가 메트릭 계산

### inference

- **inference.py**: 모델 추론 메인 스크립트
  - Retrieval 단계 (선택)
  - MRC 단계
  - 결과 저장

### utils

- **set_seed()**: 시드 고정 함수
- **postprocess_qa_predictions()**: QA 예측 후처리
  - n-best 후보 생성
  - 확률 계산
  - 앙상블용 JSON 생성
- **check_no_error()**: 설정 검증 및 체크포인트 확인

## 📊 데이터 형식

### 입력 데이터셋 형식

Hugging Face `datasets` 형식으로 저장된 데이터셋:

```python
{
    "id": "question_id",
    "question": "질문 텍스트",
    "context": "문서 컨텍스트",
    "answers": {
        "text": ["답변 텍스트"],
        "answer_start": [시작 위치]
    }
}
```

### 출력 형식

- **predictions.json**: 최종 예측 결과
  ```json
  {
    "question_id": "답변 텍스트"
  }
  ```

- **nbest_predictions.json**: 상위 n개 후보
  ```json
  {
    "question_id": [
      {
        "text": "답변 텍스트",
        "probability": 0.95,
        "start_logit": 2.3,
        "end_logit": 2.1
      }
    ]
  }
  ```

- **predictions_submit.csv**: 제출용 CSV 파일
  ```
  id	answer
  question_id	답변 텍스트
  ```

## 📈 실험 및 평가

### 평가 메트릭

- **Exact Match (EM)**: 정확히 일치하는 답변의 비율
- **F1 Score**: 토큰 단위 F1 점수
- **Hit@K**: 상위 K개 문서 중 정답이 포함된 비율
- **MRR@K**: Mean Reciprocal Rank

### WandB 실험 추적

```bash
# WandB 로그인
wandb login

# 실험 실행
python -m src.training.train \
  --report_to wandb \
  --run_name experiment_001 \
  ...
```

## 📚 참고 자료

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

## 📝 라이선스

[라이선스 정보를 여기에 추가하세요]

## 👥 기여자

[기여자 목록을 여기에 추가하세요]

## 🔗 관련 링크

- 프로젝트 저장소: [GitHub 링크]
- 이슈 트래커: [이슈 링크]
- 문서: [문서 링크]
