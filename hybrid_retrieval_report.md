# Hybrid Retrieval 성능 보고서: BM25 + KURE

## 성능 요약

### 전체 모델 비교

| 모델 | Correct Count | Accuracy | MRR | 개선율 (vs DPR) |
|------|---------------|----------|-----|-----------------|
| **DPR** (Hard Negative Mining) | 219/240 | 91.25% | 0.4800 | - |
| **KURE** (Hard Negative Mining) | 232/240 | 96.67% | 0.7126 | +5.42%p |
| **Hybrid (BM25 + KURE)** | 239/240 | **99.58%** | **0.8491** | **+8.33%p** |

---

## 주요 성과

Hybrid Retrieval이 단일 모델 대비 큰 폭의 성능 향상을 달성했습니다:

### vs DPR
- **정확도 향상**: +8.33%p (219 → 239 correct)
- **MRR 향상**: +76.9% (0.48 → 0.85)
- **추가 정답 발견**: +20개 질문

### vs KURE
- **정확도 향상**: +2.92%p (232 → 239 correct)
- **MRR 향상**: +19.2% (0.71 → 0.85)
- **추가 정답 발견**: +7개 질문

> [!IMPORTANT]
> Hybrid 방식은 240개 질문 중 **239개에서 정답을 찾아** 거의 완벽한 검색 정확도를 달성했습니다. 단 1개 질문만 실패했습니다.

---

## Hybrid Retrieval 구조

### 아키텍처

```
Query
  ├─→ BM25 (Sparse Retrieval)    ──┐
  │                                 │
  └─→ KURE (Dense Retrieval)     ──┤
                                    ├─→ Hybrid Score Fusion
                                    │   (alpha * BM25 + (1-alpha) * KURE)
                                    │
                                    ├─→ Top-K Selection
                                    │
                                    └─→ (Optional) Cross-encoder Reranker
                                        │
                                        └─→ Final Results
```

### 핵심 원리

Hybrid Retrieval은 **Sparse 방식**과 **Dense 방식**의 장점을 결합합니다:

#### 1. **BM25 (Sparse Retrieval)**

**특징**
- 키워드 기반 검색 (Lexical Matching)
- TF-IDF 개선 버전
- 통계적 기반의 빠른 검색

**장점**
- 정확한 키워드 매칭에 강함
- Out-of-vocabulary (OOV) 단어 처리 우수
- 고유명사, 숫자, 날짜 등 구체적 정보 검색에 유리

**단점**
- 동의어나 유사 표현 처리 약함
- 의미적 유사성 파악 불가

#### 2. **KURE (Dense Retrieval)**

**특징**
- 의미 기반 검색 (Semantic Matching)
- 한국어 특화 임베딩 모델
- FAISS를 활용한 벡터 검색

**장점**
- 의미적으로 유사한 문서 검색 우수
- 동의어, 패러프레이즈 처리 가능
- 문맥 이해 능력

**단점**
- 정확한 키워드 매칭이 필요한 경우 약점
- 특정 고유명사나 숫자 검색에서 부정확할 수 있음

#### 3. **Hybrid Fusion**

두 모델의 점수를 가중 평균으로 결합:

```python
final_score = alpha * bm25_score + (1 - alpha) * kure_score
```

**Alpha 파라미터**
- `alpha = 0.0`: KURE만 사용 (Dense only)
- `alpha = 0.5`: BM25와 KURE 동일 비중
- `alpha = 1.0`: BM25만 사용 (Sparse only)

---

## 성능 분석

### Accuracy 비교

```
DPR:    ████████████████████████████████████████████  91.25%
KURE:   ███████████████████████████████████████████████  96.67%
Hybrid: ██████████████████████████████████████████████████  99.58%
```

Hybrid가 단 **1개 질문만 실패**하여 거의 완벽한 정확도 달성

### Mean Reciprocal Rank (MRR) 비교

MRR은 정답이 검색 결과에서 얼마나 상위에 위치하는지를 측정합니다.

```
DPR:    ████████████████████                          0.4800
KURE:   ████████████████████████████████              0.7126
Hybrid: ███████████████████████████████████████       0.8491
```

**평균 정답 순위**
- **DPR**: 약 2.08번째
- **KURE**: 약 1.40번째  
- **Hybrid**: 약 1.18번째

> [!TIP]
> Hybrid는 정답 문서를 평균 **1.18번째** 순위에 배치하여, 대부분의 경우 정답이 검색 결과 1~2위 안에 위치합니다.

---

## 왜 Hybrid가 더 효과적인가?

### 상호 보완적 강점

1. **키워드 + 의미 검색의 결합**
   - BM25: "2020년 도쿄 올림픽" 같은 정확한 키워드 매칭
   - KURE: "여름 스포츠 대회"를 "올림픽"으로 연결하는 의미 이해

2. **Robustness 향상**
   - 한 모델이 놓친 문서를 다른 모델이 포착
   - 질문 유형에 따라 adaptive하게 작동

3. **검색 안정성**
   - 특정 쿼리 패턴에 대한 의존도 감소
   - 다양한 질문 유형에 대해 일관된 성능

### 실제 예시 (가설)

**질문**: "한국의 수도는 어디인가?"

```
BM25 검색:
- "한국", "수도" 키워드로 정확히 매칭
→ "대한민국의 수도는 서울이다." (High Score)

KURE 검색:
- "한국의 수도"와 "대한민국의 수도" 의미적으로 동일
- "서울은 한반도 중부에 위치한 도시" (의미 유사)
→ 관련 문서들 검색

Hybrid:
- BM25의 정확한 키워드 매칭 + KURE의 의미적 확장
→ 최적의 문서 선택
```

---

## 기술 구현 세부사항

### BM25 Ensemble

본 프로젝트에서는 단순 BM25가 아닌 **BM25 Ensemble**을 사용합니다:

**특징**
- 여러 BM25 인스턴스를 조합
- 다양한 토크나이징 전략 적용 가능
- 검색 안정성과 커버리지 향상

### KURE 최적화

```python
Model: nlpai-lab/KURE-v1 (SentenceTransformer)
Index: FAISS IndexFlatIP
Precision: FP16 (GPU memory efficiency)
Normalization: L2 normalization for cosine similarity
Batch Size: 64~128 (optimized for throughput)
```

### Hybrid Fusion Strategy

```python
# Score Normalization
bm25_scores = normalize(bm25_raw_scores)
kure_scores = normalize(kure_raw_scores)

# Weighted Fusion
for doc_id in all_candidate_docs:
    hybrid_score = alpha * bm25_scores[doc_id] + (1 - alpha) * kure_scores[doc_id]

# Re-ranking by hybrid score
ranked_docs = sort_by_score(hybrid_score, descending=True)
```

---

## 결론

### 핵심 성과

1. **거의 완벽한 검색 정확도**: 99.58% (239/240)
2. **높은 순위 품질**: MRR 0.8491로 대부분의 정답이 상위 1~2위에 배치
3. **단일 모델 대비 우수성**: DPR, KURE 단독 사용보다 명확한 성능 향상

### 핵심 인사이트

> [!NOTE]
> **Sparse + Dense Hybrid 전략**은 단순히 성능을 더하는 것이 아니라, 두 패러다임의 약점을 상호 보완하여 **시너지 효과**를 창출합니다.

**Hybrid Retrieval의 장점**:
- Lexical과 Semantic의 완벽한 조화
- 다양한 질문 유형에 대한 안정적 성능
- 높은 정확도와 동시에 높은 순위 품질

### 모델 선택 가이드

| 상황 | 권장 모델 |
|------|----------|
| **최고 성능 필요** | Hybrid (BM25 + KURE) |
| 한국어 전용, 리소스 제한적 | KURE |
| 빠른 속도 우선, 키워드 검색 | BM25 |
| 영어 데이터셋 | DPR |

### 추가 최적화 방향

**현재 단계**: BM25 + KURE Hybrid

**다음 단계 고려사항**:
1. **Cross-encoder Reranking**: Hybrid 결과에 재순위화 추가
2. **Alpha Tuning**: 데이터셋 특성에 맞는 최적 가중치 탐색
3. **Adaptive Fusion**: 쿼리 타입에 따라 alpha를 동적으로 조정
4. **Query Expansion**: 쿼리를 확장하여 검색 커버리지 향상

---

## 참고 정보

### 실험 설정

- **Dataset**: 한국어 MRC (Machine Reading Comprehension)
- **Total Questions**: 240
- **Corpus**: Wikipedia documents
- **Training**: Hard Negative Mining 적용
- **Evaluation Metrics**: Accuracy (Correct Count), MRR

### 코드베이스

- **Hybrid Implementation**: `src/retrieval/retrieval_hybrid_kure.py`
- **BM25 Retrieval**: BM25 Ensemble
- **KURE Retrieval**: `src/retrieval/retrieval_kure.py`
- **Config**: `src/config/arguments.py`

### Mean Reciprocal Rank (MRR) 설명

MRR은 각 쿼리에 대해 첫 번째 정답이 나타나는 순위의 역수를 평균낸 값입니다.

```
MRR = (1/N) * Σ(1 / rank_i)

예시:
Query 1: 정답 순위 1위 → 1/1 = 1.0
Query 2: 정답 순위 2위 → 1/2 = 0.5
Query 3: 정답 순위 1위 → 1/1 = 1.0
MRR = (1.0 + 0.5 + 1.0) / 3 = 0.83
```

높은 MRR은 정답이 검색 결과 상위에 일관되게 배치됨을 의미합니다.

---

**보고서 생성일**: 2025-12-10  
**평가 데이터**: 총 240개 질문  
**설정**: BM25 Ensemble + KURE (Hard Negative Mining)
