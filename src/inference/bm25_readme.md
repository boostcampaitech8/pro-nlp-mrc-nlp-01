# BM25 + Wandb Inference

BM25 기반 Retrieval과 wandb 로깅을 지원하는 추론 스크립트입니다.

## 파일 구조

| 파일 | 설명 |
|------|------|
| `inference_bm25_wandb.py` | wandb 로깅 지원 추론 스크립트 |
| `../retrieval/retrieval_bm25_wandb.py` | Retrieval accuracy 계산 지원 |

---

## 사용법

### 1. Eval 모드 (성능 확인)
validation 데이터로 EM/F1 점수 확인 + wandb 기록

```bash
python -m src.inference.inference_bm25_wandb \
  --output_dir outputs/eval_k100_bm25/ \
  --dataset_name data/train_dataset/ \
  --model_name_or_path baseline/models/train_dataset/ \
  --do_eval \
  --use_wandb \
  --wandb_project "retrieval" \
  --wandb_run_name "bm25_k100_eval"
```


### 2. Predict 모드 (제출 파일 생성)
test 데이터로 predictions_submit.csv 생성

```bash
python -m src.inference.inference_bm25_wandb \
  --output_dir outputs/test_bm25_wandb/ \
  --dataset_name data/test_dataset/ \
  --model_name_or_path baseline/models/train_dataset/ \
  --do_predict \
  --use_wandb \
  --wandb_project "retrieval" \
  --wandb_run_name "bm25_wandb_submit"
```

---

## 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--output_dir` | (필수) | 출력 폴더 |
| `--dataset_name` | (필수) | 데이터셋 경로 |
| `--model_name_or_path` | (필수) | Reader 모델 경로 |
| `--do_predict` | False | 예측 모드 (제출용) |
| `--do_eval` | False | 평가 모드 (EM/F1 확인) |
| `--top_k_retrieval` | 100 | 검색할 문서 수 |
| `--use_wandb` | False | wandb 로깅 사용 |
| `--wandb_project` | "retrieval" | wandb 프로젝트명 |
| `--wandb_run_name` | 자동생성 | wandb 실행 이름 |

---

## Wandb 기록 항목

### Config (모든 모드)
- `retrieval_method`: BM25
- `top_k_retrieval`: k 값
- `model_name_or_path`, `dataset_name` 등

### Metrics (Eval 모드에서만)
- `retrieval/accuracy`: 정답 문서가 top-k에 포함된 비율
- `mrc/exact_match`: EM 점수
- `mrc/f1`: F1 점수

---

## 출력 파일

`--output_dir` 폴더에 생성됨:

| 파일 | 설명 |
|------|------|
| `predictions_submit.csv` | 제출용 파일 |
| `predictions.json` | 예측 결과 |
| `nbest_predictions.json` | N-best 예측 |

---

## 참고

- Eval 모드: `data/train_dataset` 사용 (정답 있음)
- Predict 모드: `data/test_dataset` 사용 (정답 없음)
- 두 모드는 다른 데이터셋을 사용하므로 동시에 쓸 수 없음
