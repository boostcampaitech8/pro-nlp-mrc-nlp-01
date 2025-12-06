# Hard Negative Mining 설명

## 1. Hard Negative Passage란?

### 1.1 개념
Hard Negative는 질문과 **의미적으로 유사하지만 정답이 아닌 문서(Passage)**를 의미합니다. 이는 모델이 구분하기 어려운 "까다로운 오답"으로, Dense Retrieval 모델 학습에 매우 중요한 역할을 합니다.

### 1.2 필요성
기존 DPR 학습에서 사용하는 **In-batch Negative**는 같은 배치 내 다른 질문의 정답 문서를 Negative로 사용합니다. 예를 들어:
- Q1: "미국 대통령의 임기는?" → P1 (정답): "미국 대통령 임기 4년..."
- Q2: "프랑스의 수도는?" → P2 (정답): "파리는 프랑스의 수도..."

이 경우 Q1 입장에서 P2는 Negative가 되지만, **주제가 완전히 다르기 때문에 모델이 쉽게 구분**할 수 있습니다.

반면, Hard Negative는:
- Q1: "미국 대통령의 임기는?" 
- Hard Negative: "미국 부통령의 임기는 대통령과 같이..." (주제는 비슷하지만 정답 아님)

처럼 모델이 혼동하기 쉬운 문서를 제공하여, 모델이 **진짜 정답과 유사한 오답을 구분하는 능력**을 학습하게 합니다.

## 2. Hard Negative 찾기 방법

### 2.1 BM25 기반 (본 프로젝트 채택)
- **장점**: 구현이 간단하고 빠름. 별도 모델 학습 불필요.
- **방법**: 
  1. BM25로 각 질문에 대해 Top-K 문서 검색
  2. 검색 결과 중 정답이 아닌 문서들 선별
  3. 그중 BM25 점수가 가장 높은 문서를 Hard Negative로 선택

### 2.2 기존 DPR 모델 기반
- **장점**: 이미 학습된 DPR의 "약점"을 집중 공략 가능.
- **방법**: 학습된 DPR로 검색했을 때 높은 점수를 받았지만 오답인 문서 사용.
- **단점**: 초기 DPR이 없으면 사용 불가 (Iterative 학습 필요).

### 2.3 Cross-Encoder 기반
- **장점**: 가장 정교한 Hard Negative 선별 가능.
- **방법**: Question-Passage 쌍을 입력받아 관련성을 직접 점수화하는 모델 사용.
- **단점**: 계산 비용이 높고 별도 모델 학습 필요.

## 3. 현재 코드 구현 방식

### 3.1 전체 파이프라인
```
[입력] train_dataset (question, context, answers)
    ↓
[1단계] scripts/mine_hard_negatives.py
    - BM25로 각 질문마다 Top-100 검색
    - 정답 제외하고 가장 높은 점수의 문서 선택
    ↓
[출력] train_dataset_hn (+ hard_negative_context 컬럼 추가)
    ↓
[2단계] src/training/train_dpr.py
    - (Q, Positive, Hard Negative) 형태로 학습
    - Cross-Entropy Loss로 Positive를 가장 높게 랭킹하도록 학습
    ↓
[결과] outputs/train_dataset_hn (학습된 DPR 모델)
```

### 3.2 Mining 스크립트 핵심 로직 (`scripts/mine_hard_negatives.py`)

#### (1) BM25 검색
```python
# BM25 Retriever 초기화 (전역)
global_retriever = BM25RetrievalWandB(...)
global_retriever.get_sparse_embedding()

# 각 질문에 대해 Top-K 검색
scores = global_retriever.bm25.get_scores(tokenized_query)
sorted_indices = np.argsort(scores)[::-1]
doc_indices = sorted_indices[:top_k]  # Top-100
```

#### (2) Hard Negative 선별
```python
for rank, doc_idx in enumerate(doc_indices):
    retrieved_context = global_retriever.contexts[doc_idx]
    
    # 1. 정답 문서 제외
    if retrieved_context == original_context:
        continue
    
    # 2. 정답 문자열 포함 문서 제외
    has_answer = any(ans in retrieved_context for ans in answer_texts)
    if has_answer:
        continue
    
    # 3. 첫 번째로 조건을 만족하는 문서 = Hard Negative
    hn_context = retrieved_context
    break
```

#### (3) 병렬 처리 최적화
```python
# Multiprocessing으로 속도 향상
with Pool(processes=args.num_proc) as pool:
    results = list(tqdm(
        pool.imap(worker_retrieve_and_mine, chunks),
        total=len(chunks)
    ))
```

### 3.3 학습 코드 핵심 로직 (`src/training/train_dpr.py`)

#### (1) 데이터 준비
```python
def prepare_features(example):
    features = {
        "question": example["question"],
        "context": example["context"]  # Positive
    }
    if "hard_negative_context" in example:
        features["hard_negative_context"] = example["hard_negative_context"]
    return features
```

#### (2) Batch 구성 (DataCollator)
```python
def __call__(self, features):
    questions = [f["question"] for f in features]
    contexts = []
    for f in features:
        contexts.append(f["context"])  # Positive
        if "hard_negative_context" in f:
            contexts.append(f["hard_negative_context"])  # Hard Negative
    
    # Tokenize
    q_batch = self.tokenizer(questions, ...)
    c_batch = self.tokenizer(contexts, ...)  # [P1, HN1, P2, HN2, ...]
```

#### (3) Loss 계산 (BiEncoder)
```python
def forward(self, q_input_ids, c_input_ids, ...):
    q_outputs = self.question_encoder(...)  # (batch_size, dim)
    c_outputs = self.context_encoder(...)   # (batch_size * 2, dim) if HN exists
    
    sim_scores = torch.matmul(q_outputs, c_outputs.T)  # (batch, batch*2)
    
    # Target: [0, 2, 4, ...] (Positive 위치만)
    stride = c_outputs.shape[0] // q_outputs.shape[0]
    target = torch.arange(0, c_outputs.shape[0], step=stride)
    
    loss = F.cross_entropy(sim_scores, target)
```

## 4. 학습 결과 분석

| 구분 | In-batch Only | + Hard Negatives | 개선폭 |
|:---|:---:|:---:|:---:|
| Accuracy @ 100 | 0.42 | **0.91** | +0.49 |
| MRR @ 100 | 0.06 | **0.48** | +0.42 |

Hard Negative 추가만으로도 **검색 정확도가 2배 이상 상승**했으며, 이는 모델이 "유사하지만 다른" 문서를 효과적으로 구분하게 되었음을 의미합니다.

## 5. 추가 개선 가능성

### 5.1 Iterative Mining
현재 학습된 DPR을 사용하여 다시 Hard Negative를 찾는 방식입니다.
- 1차: BM25 Hard Negative로 DPR 학습
- 2차: 학습된 DPR로 Hard Negative 재채굴
- 3차: 재학습 반복

### 5.2 Multiple Hard Negatives
현재는 1개의 Hard Negative만 사용하지만, 2~5개로 늘려 학습 난이도를 높일 수 있습니다.

### 5.3 Hard Negative 품질 개선
- **Difficulty Filtering**: 너무 쉽거나 너무 어려운 Negative 제외
- **Diversity**: 비슷한 주제의 Hard Negative 여러 개보다 다양한 주제 선호
