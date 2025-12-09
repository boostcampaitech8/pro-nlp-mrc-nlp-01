# Negative Passage 성능 실험 보고서

## 1. 실험 개요

### 1.1 연구 목적
본 실험은 Machine Reading Comprehension (MRC) 모델 학습 시 다양한 **Negative Passage 샘플링 전략**이 모델 성능에 미치는 영향을 체계적으로 분석하고, 최적의 Negative Passage 전략을 도출하는 것을 목적으로 합니다.

### 1.2 연구 배경
MRC 모델 학습 시 Positive Passage(정답이 포함된 문서)만으로 학습하는 것보다, Negative Passage(정답이 포함되지 않은 문서)를 함께 학습에 포함하면 모델의 판별 능력이 향상될 수 있습니다. 특히 Hard Negative(질문과 유사하지만 정답이 없는 문서)는 모델이 미묘한 차이를 학습하는 데 도움이 됩니다.

### 1.3 실험 범위
- **Baseline**: Negative Passage 없이 Positive만 사용
- **Random Negative**: 질문과 무관하게 랜덤하게 선택한 Negative Passage
- **BM25 Hard Negative**: BM25 검색으로 질문과 유사하지만 정답이 없는 Negative Passage
- **Mixed Negative**: Hard Negative와 Random Negative를 혼합

## 2. 실험 설정

### 2.1 모델 및 데이터
- **모델**: `klue/bert-base` (한국어 BERT 기반)
- **학습 데이터**: 3,952개 샘플
- **검증 데이터**: 240개 샘플
- **Wikipedia 문서**: 56,737개 (중복 제거 후)

### 2.2 하이퍼파라미터
| 파라미터 | 값 |
|---------|-----|
| 학습 에폭 | 2 |
| 학습률 | 3e-5 |
| 배치 크기 (학습) | 16 |
| 배치 크기 (평가) | 32 |
| 최대 시퀀스 길이 | 384 |
| 문서 스트라이드 | 128 |
| 최대 정답 길이 | 30 |
| Warmup 비율 | 0.1 |
| Weight Decay | 0.01 |

### 2.3 실험 전략
| 실험명 | 전략 | Negative 개수 | 설명 |
|--------|------|---------------|------|
| baseline | none | 0 | Negative Passage 없음 |
| random_neg_1 | random | 1 | Random Negative 1개 |
| random_neg_3 | random | 3 | Random Negative 3개 |
| hard_neg_1 | hard | 1 | BM25 Hard Negative 1개 |
| hard_neg_3 | hard | 3 | BM25 Hard Negative 3개 |
| mixed_neg_2 | mixed | 2 | Hard 1개 + Random 1개 |

## 3. 실험 방법론

### 3.1 BM25 Retriever
- **알고리즘**: BM25 (Best Matching 25)
- **파라미터**: k1=1.5, b=0.75
- **역할**: 질문에 대해 유사한 문서를 검색하여 Hard Negative 후보 생성

### 3.2 Negative Passage 샘플링 전략

#### 3.2.1 Random Negative
- Wikipedia 문서 전체에서 랜덤하게 선택
- 조건: Positive Context와 다르고, 정답이 포함되지 않은 문서
- 장점: 다양한 도메인의 Negative 샘플 제공
- 단점: 질문과 무관한 문서가 많아 학습 효율이 낮을 수 있음

#### 3.2.2 BM25 Hard Negative
- BM25로 질문에 대해 상위 k개 문서 검색
- 그 중 정답이 포함되지 않은 문서 선택
- 장점: 질문과 유사하지만 정답이 없는 문서로 모델의 판별 능력 향상
- 단점: 검색 품질에 의존적

#### 3.2.3 Mixed Negative
- Hard Negative와 Random Negative를 혼합
- Hard Negative의 판별 능력 향상 + Random Negative의 일반화 능력 결합

### 3.3 데이터 증강 방식
각 Positive 예시에 대해:
1. Positive 예시 추가 (정답 포함)
2. 전략에 따라 Negative 예시 추가 (정답 없음, 빈 answers)

## 4. 실험 결과

### 4.1 전체 실험 결과 요약

| 실험명 | 전략 | Negative 개수 | 학습 샘플 수 | 학습 Loss | Exact Match (%) | F1 Score (%) |
|--------|------|---------------|-------------|-----------|-----------------|--------------|
| **baseline** | none | 0 | 7,978 | 1.259 | **57.50** | 67.60 |
| **random_neg_1** | random | 1 | 14,696 | 0.675 | **60.00** | **69.02** ⭐ |
| random_neg_3 | random | 3 | 28,006 | 0.368 | 56.25 | 66.34 |
| hard_neg_1 | hard | 1 | 13,676 | 0.767 | 58.33 | 68.35 |
| hard_neg_3 | hard | 3 | 25,384 | 0.443 | 56.67 | 66.21 |
| mixed_neg_2 | mixed | 2 | 20,394 | 0.528 | 56.25 | 66.80 |

### 4.2 성능 비교

#### 4.2.1 F1 Score 비교
```
Baseline:           67.60%
Random Negative 1:  69.02% (+1.41%) ⭐ 최고 성능
Hard Negative 1:    68.35% (+0.75%)
Mixed Negative 2:   66.80% (-0.80%)
Hard Negative 3:    66.21% (-1.39%)
Random Negative 3:  66.34% (-1.26%)
```

#### 4.2.2 Exact Match 비교
```
Baseline:           57.50%
Random Negative 1:  60.00% (+2.50%) ⭐ 최고 성능
Hard Negative 1:    58.33% (+0.83%)
Hard Negative 3:    56.67% (-0.83%)
Random Negative 3:  56.25% (-1.25%)
Mixed Negative 2:   56.25% (-1.25%)
```

### 4.3 주요 발견사항

#### ✅ 긍정적 발견
1. **Random Negative 1개가 최고 성능**
   - F1 Score: 69.02% (Baseline 대비 +1.41%)
   - Exact Match: 60.00% (Baseline 대비 +2.50%)
   - 적절한 수준의 Negative 샘플링이 성능 향상에 기여

2. **Hard Negative 1개도 효과적**
   - F1 Score: 68.35% (Baseline 대비 +0.75%)
   - Hard Negative가 모델의 판별 능력을 향상시킴

3. **학습 Loss 감소**
   - Negative 샘플 추가 시 학습 Loss가 크게 감소
   - Random Negative 1: 0.675 (Baseline: 1.259)

#### ⚠️ 부정적 발견
1. **Negative 개수 증가 시 성능 저하**
   - Random Negative 3개: F1 66.34% (Baseline보다 낮음)
   - Hard Negative 3개: F1 66.21% (Baseline보다 낮음)
   - **과도한 Negative 샘플링은 오히려 성능을 저하시킴**

2. **Mixed Negative의 기대 이하 성능**
   - F1 Score: 66.80% (Baseline보다 낮음)
   - Hard와 Random의 단순 혼합이 최적이 아닐 수 있음

## 5. 결과 분석

### 5.1 Negative 개수에 따른 성능 변화

#### Random Negative 전략
- **1개**: F1 69.02% ⭐ 최고
- **3개**: F1 66.34% (성능 저하)

**분석**: Random Negative는 1개일 때 최적이며, 3개로 증가하면 성능이 저하됩니다. 이는 과도한 Negative 샘플이 모델 학습을 방해할 수 있음을 시사합니다.

#### Hard Negative 전략
- **1개**: F1 68.35% (양호)
- **3개**: F1 66.21% (성능 저하)

**분석**: Hard Negative도 1개일 때가 최적이며, 3개로 증가하면 성능이 저하됩니다. Hard Negative가 많아질수록 모델이 혼란스러워질 수 있습니다.

### 5.2 전략별 비교

#### Random vs Hard Negative
- **Random Negative 1개**: F1 69.02% (최고)
- **Hard Negative 1개**: F1 68.35%

**분석**: 
- Random Negative가 약간 더 우수한 성능을 보임
- Hard Negative는 질문과 유사한 문서를 제공하지만, 때로는 너무 유사하여 모델이 혼란스러울 수 있음
- Random Negative는 다양한 도메인의 Negative를 제공하여 일반화 능력을 향상시킬 수 있음

### 5.3 학습 샘플 수와 성능의 관계

| 실험 | 학습 샘플 수 | F1 Score | 비고 |
|------|-------------|----------|------|
| baseline | 7,978 | 67.60% | - |
| random_neg_1 | 14,696 | **69.02%** | 최적 |
| random_neg_3 | 28,006 | 66.34% | 과도한 증강 |
| hard_neg_1 | 13,676 | 68.35% | 적절 |
| hard_neg_3 | 25,384 | 66.21% | 과도한 증강 |

**분석**: 
- 학습 샘플 수가 많다고 항상 성능이 좋은 것은 아님
- **적절한 수준의 Negative 샘플링(1개)이 최적**
- 과도한 증강(3개 이상)은 오히려 성능을 저하시킴

### 5.4 학습 Loss 분석

| 실험 | 학습 Loss | F1 Score |
|------|-----------|----------|
| baseline | 1.259 | 67.60% |
| random_neg_1 | 0.675 | **69.02%** |
| random_neg_3 | 0.368 | 66.34% |
| hard_neg_1 | 0.767 | 68.35% |
| hard_neg_3 | 0.443 | 66.21% |

**분석**:
- Negative 샘플 추가 시 학습 Loss가 크게 감소
- 하지만 Loss가 너무 낮을수록(0.3~0.4) 오히려 성능이 저하
- **적절한 수준의 Loss(0.6~0.7)가 최적 성능과 연관**

## 6. 결론 및 권장사항

### 6.1 주요 결론

1. **Random Negative 1개 전략이 최고 성능**
   - F1 Score: 69.02% (Baseline 대비 +1.41%p)
   - Exact Match: 60.00% (Baseline 대비 +2.50%p)
   - 다양한 도메인의 Negative 샘플이 모델의 일반화 능력을 향상시킴

2. **적절한 수준의 Negative 샘플링이 중요**
   - Negative 1개: 성능 향상
   - Negative 3개: 성능 저하
   - **과도한 Negative 샘플링은 오히려 해로울 수 있음**

3. **Hard Negative도 효과적이지만 Random보다 약간 낮음**
   - Hard Negative 1개: F1 68.35% (양호)
   - 질문과 유사한 문서를 제공하지만, 때로는 너무 유사하여 혼란을 줄 수 있음

4. **Mixed Negative는 기대 이하**
   - Hard와 Random의 단순 혼합이 최적이 아닐 수 있음
   - 더 정교한 혼합 비율이나 전략이 필요할 수 있음

### 6.2 권장사항

#### ✅ 즉시 적용 가능한 권장사항
1. **Random Negative 1개 전략 채택**
   - 현재 실험에서 최고 성능을 보임
   - 구현이 간단하고 효과적
   - 다양한 도메인의 Negative 샘플 제공

2. **Negative 개수는 1개로 제한**
   - 3개 이상의 Negative는 성능 저하
   - 학습 효율과 성능의 균형을 고려

#### 🔬 추가 연구 제안
1. **Hard Negative 품질 개선**
   - 현재 BM25 기반 Hard Negative가 Random보다 약간 낮은 성능
   - 더 정교한 검색 알고리즘(예: Dense Retrieval) 적용 검토
   - Hard Negative 필터링 기준 개선

2. **Mixed Negative 전략 재검토**
   - 현재 Mixed 전략이 기대 이하
   - Hard와 Random의 최적 비율 탐색
   - 동적 혼합 전략 연구

3. **Negative 샘플링 품질 평가**
   - Negative 샘플의 난이도 분석
   - 모델이 실제로 학습하는 Negative 패턴 분석

4. **하이퍼파라미터 최적화**
   - 현재 실험은 2 epoch, 더 긴 학습 시 성능 변화 확인
   - 학습률, 배치 크기 등과의 상호작용 분석

### 6.3 실무 적용 가이드

#### 단계별 적용 방법
1. **1단계: Baseline 모델 학습**
   - Negative 없이 Positive만으로 학습
   - Baseline 성능 측정

2. **2단계: Random Negative 1개 추가**
   - 각 Positive에 Random Negative 1개 추가
   - 성능 향상 확인

3. **3단계: (선택) Hard Negative 1개 시도**
   - BM25 기반 Hard Negative 추가
   - Random과 비교하여 더 나은 전략 선택

4. **4단계: 성능 모니터링**
   - 검증 세트에서 지속적인 성능 확인
   - 과적합 여부 확인

#### 주의사항
- ⚠️ **Negative 개수는 1~2개로 제한**
- ⚠️ **과도한 증강은 성능 저하를 초래할 수 있음**
- ⚠️ **데이터셋 특성에 따라 최적 전략이 다를 수 있음**

## 7. 부록

### 7.1 실험 환경
- **GPU**: NVIDIA GeForce RTX 2070
- **Python**: 3.10.19
- **Framework**: PyTorch, Transformers
- **시드**: 42 (재현성 보장)

### 7.2 평가 메트릭
- **F1 Score**: 예측과 정답 간의 F1 점수 (주 메트릭)
- **Exact Match**: 완전히 일치하는 비율
- **학습 Loss**: 학습 과정의 손실값

### 7.3 참고 자료
- 실험 결과 JSON: `performance_experiment_report.json`
- 실험 노트북: `05_negative_passage_performance_experiment.ipynb`
- 실험 모델 체크포인트: `experiments/` 디렉토리

---

**보고서 작성일**: 2025년  
**작성자**: MRC 실험 팀  
**버전**: 1.0






