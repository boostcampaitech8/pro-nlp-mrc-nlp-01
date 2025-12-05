# EDA (Exploratory Data Analysis)

MRC(Machine Reading Comprehension) 데이터 탐색적 분석 노트북입니다.

## 📁 폴더 구조

```
EDA/
├── 01_dataset_analysis.ipynb       # 데이터셋 분석
├── 02_corpus_analysis.ipynb        # Wikipedia corpus 분석
├── 03_question_type_6W2H.ipynb     # 6W2H 질문 분류
├── 04_question_clustering.ipynb    # 질문 클러스터링
├── 05_similarity_lexical.ipynb     # Lexical 유사도
├── 06_similarity_semantic.ipynb    # Semantic 유사도
├── _archive/                       # 참고용 원본 노트북
└── README.md
```

---

## 📊 노트북 설명

### 01_dataset_analysis.ipynb
**Train/Validation 데이터셋의 품질과 통계 분석**

| 분석 항목 | 설명 |
|-----------|------|
| 질문 분석 | 길이 분포, 중복 확인 |
| 정답 분석 | 길이 분포, 위치 분포, exact match 검증 |
| Context 분석 | 길이 분포 |
| 데이터 품질 점검 | 중복 ID, annotation 불일치, 이상치 등 |

---

### 02_corpus_analysis.ipynb
**Retrieval에 사용되는 Wikipedia corpus 분석**

| 분석 항목 | 설명 |
|-----------|------|
| 문서 개수 | ~57k개 |
| 토큰 길이 분포 | 512 토큰 초과 비율 |
| [UNK] 토큰 분포 | 토크나이저 커버리지 |
| 언어 비율 | 한국어/특수문자 비율 |

---

### 03_question_type_6W2H.ipynb
**규칙 기반 6W2H 프레임 질문 분류**

- Who (누가/누구)
- What (무엇/무슨/어떤)
- When (언제)
- Where (어디)
- Why (왜)
- How (어떻게)
- How many/much (몇/숫자)
- 유형별 정답 길이 분석

---

### 04_question_clustering.ipynb
**NLP 기반 질문 주제 클러스터링**

- 한국어 전처리 (명사 추출)
- TF-IDF 벡터화
- KMeans 클러스터링 (Elbow Method)
- 클러스터별 상위 키워드 분석
- PCA 기반 2D 시각화

---

### 05_similarity_lexical.ipynb
**Lexical Overlap 기반 질문-Context 유사도**

- Jaccard Similarity 계산
- 질문-정답 간 lexical overlap
- 유사도 구간별 분포 분석

---

### 06_similarity_semantic.ipynb
**임베딩 기반 질문-Context 유사도**

- Sentence Transformers 모델 사용
- Cosine Similarity 계산
- 낮은 유사도 샘플 분석 (Retrieval 어려움 예상)
- Lexical vs Semantic Similarity 비교

---

## 📂 _archive/
정리 전 원본 노트북들 (참고용)
