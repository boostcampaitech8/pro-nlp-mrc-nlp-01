# EDA (Exploratory Data Analysis)

현재 프로젝트에서 사용되고 있는 데이터셋은 MRC (Machine Reading Comprehension) 형태의 데이터셋이고, Retrieval + Reader 구조로 풀게 되는 전형적인 위키 기반 QA 데이터셋이다.
이를 위한 EDA 체크리스트 설계

[ ] 질문(Question) 분석
- 질문 길이 분포 (토큰/문자 단위)
- 질문 유형 분석 ("what", "when", "why", "how" 등)
- 중복 질문 존재 여부 
- 질문 속 stopword 비율
- 질문이 어떤 주제를 많이 다루는지 (예:TF-IDF 기반 클러스터링)

목적: 질문 패턴을 알면 모델의 약점을 조기에 파악 가능

[ ] 정답(answer) 분석
- answers 구조 분석
- answer 길이 분포
- answer_start가 context 길이를 넘어가는 이상치 있는지
- 동일 question에 multiple answers가 존재하는지 확인
- answer가 context에서 exact match 되는지 검증

목적: annotation 오류 확인, span extraction이 가능한지 체크

[ ] 정답