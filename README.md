# 🔍 Open-Domain Question Answering (ODQA)
> **1차 도메인 프로젝트 Wrap-Up Report 기반 프로젝트 문서화**  
> **프로젝트 기간: 2025.11.24 ~ 2025.12.15**

---

## 👥 1. 프로젝트 팀 구성 및 역할

| 파트 | 이름 | 주요 역할 및 활동 |
| :--- | :--- | :--- |
| **Data** | **김지환 (T8052)** | EDA 수행, 데이터 품질 관리, 서버 복구 매뉴얼 작성 |
| **Data** | **배민석 (T8094)** | LLM 기반 Synthetic QA 데이터 생성, 데이터 증강 전략 수립 |
| **Retrieval** | **강진영 (T8007)** | 점진적 Retrieval 개선 로직 설계, 실험 프로세스 체계화 |
| **Retrieval** | **배주연 (T8096)** | Hybrid Retrieval (BM25+KURE) 구현, Hard Negative Mining |
| **Reader** | **박준하 (T8086)** | Reader 베이스라인 탐색, Qwen 기반 LLM Reader 파인튜닝 실험 |
| **Reader** | **정제원 (T8184)** | 하이퍼파라미터 튜닝 (Bayesian Optimization), CNN 레이어 추가 실험 |

---

## 📝 2. 프로젝트 개요

본 프로젝트는 사용자의 질문에 대해 방대한 문서군(Corpus)에서 관련 문서를 찾아내고(**Retriever**), 해당 문서 내에서 정답 구간을 정확히 추출하는(**Reader**) 두 단계 기반의 **ODQA 시스템**을 구축하는 프로젝트입니다.

- **목표**: 자연어 질문(Query)에 대해 Wikipedia 데이터베이스에서 가장 적절한 답변을 도출
- **핵심 파이프라인**: Data Preprocessing → Sparse/Dense Retrieval → Reranking → MRC (Reader) → Ensemble

---

## 🚀 3. 주요 수행 전략 및 결과

### 📊 Data 파트
- **문제점**: Negative Passage 부재로 인한 모델의 환각 현상(Hallucination) 및 Corpus 활용도 저조
- **해결 방안**:
  - **Negative Sampling**: Random 및 Hard Negative를 도입하여 "정답 없음" 판별 능력 강화
  - **Data Augmentation**: LLM을 활용해 학습에 미사용된 Context 기반 Synthetic QA 생성
- **인사이트**: Negative Sampling은 "양보다 질"이 중요하며, 적절한 난이도의 오답 예시(1개 수준)가 가장 효과적임을 확인

### 🔍 Retrieval 파트
- **Sparse Retrieval**: BM25 Ensemble (Subword + Morphs)을 통해 최고 정확도 확보
- **Dense Retrieval**: In-batch Negative의 한계를 Hard Negative Mining으로 극복하고, 한국어 특화 모델인 **KURE** 도입
- **Hybrid Strategy**: 키워드 기반 BM25와 의미 기반 KURE를 결합하여 **Top-100 정확도 99.58%** 달성
- **Reranker**: `bge-reranker-v2-m3`를 적용하여 검색 결과 재순위화 및 MRR 향상

### 📖 Reader 파트
- **Baseline**: `HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned` 선정
- **Optimization**: Bayesian Optimization을 통해 하이퍼파라미터 안전 구간(Learning Rate 3e-5 등) 도출
- **실험 및 시도**:
  - **CNN Layer**: 지역적 특징 강화를 위해 시도했으나 성능 향상 미미
  - **LLM Reader**: Qwen3 모델 LoRA 파인튜닝 시도. 문맥 이해는 우수하나 Span 추출 정밀도에서 RoBERTa 대비 한계 확인
- **Ensemble**: 상위 20개 n-best 후보의 텍스트 기반 Soft Voting 및 가중치 조정을 통해 최종 성능 극대화

---

## 📈 4. 최종 성능 변화 (Retrieval 중심)

| 단계 | 모델 전략 | Top-100 Accuracy | MRR | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| Phase 1 | BM25 Ensemble | 96.67% | 0.7008 | Sparse 최고 성능 |
| Phase 2 | DPR (In-batch) | 42.00% | 0.0600 | 초기 실패 사례 |
| Phase 2 | DPR (Hard Neg) | 91.25% | 0.4813 | 변별력 확보 |
| Phase 3 | KURE (Hard Neg) | 96.67% | 0.7126 | 한국어 특화 우위 |
| Phase 4 | **Hybrid (BM25+KURE)** | **99.58%** | 0.8491 | 거의 완벽한 검색 |
| Phase 5 | + Reranker | 95.00% | **0.8794** | 정교한 재순위화 |

---

## 📁 5. 프로젝트 구조

```text
.
├── src/                    # 소스 코드
│   ├── retrieval/          # Sparse/Dense/Hybrid Retrieval 및 Reranker
│   ├── training/           # Reader 모델 학습 (QA Trainer 포함)
│   ├── inference/          # 추론 파이프라인 (Retriever + Reader)
│   └── utils/              # 전처리, 후처리 및 앙상블 로직
├── notebooks/              # EDA 및 데이터 실험 (Augmentation, Negative Sampling)
├── scripts/                # 학습 및 추론 실행 스크립트
└── README.md              # 프로젝트 문서
```

---

## 💡 6. 자체 평가 및 시사점

### ✅ 잘한 점
- **단계적 개선**: Sparse → Dense → Hybrid → Reranker로 이어지는 체계적인 성능 향상
- **철저한 기록**: WandB와 Notion을 연동하여 실험 가설과 결과를 자산화

### ⚠️ 아쉬운 점
- **일반화 성능**: 검증 데이터(Private)에 최적화된 전략이 Public 데이터에서 다소 편향됨
- **K-Fold 미적용**: 데이터 분포에 대한 보다 강건한 검증 체계 부족

### 🌟 배운 점
- "한 번에 하나의 가설만 검증하는 원칙"의 중요성
- 팀 공통의 실험 컨벤션 정립이 협업 효율성에 미치는 지대한 영향

---

## 🛠️ 7. 설치 및 실행 방법

### 설치
```bash
git clone <repository-url>
pip install -r requirements.txt
```

### 실행 (Train & Inference)
```bash
# 학습 실행
bash scripts/train.sh
# 추론 실행
bash scripts/inference.sh
```

---
**NLP-01 Team T800** | 강진영, 김지환, 박준하, 배민석, 배주연, 정제원