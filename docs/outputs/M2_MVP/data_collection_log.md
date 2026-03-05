# AI Hub 데이터 수집 로그

- **수집 일시**: 2026-03-05
- **사용 데이터셋**:
  - [71852] 공공 민원 상담 LLM 데이터 (주 학습 데이터)
  - [71844] 민간 민원 상담 LLM 데이터 (보조 학습 데이터)
- **전처리 통계**:
  - 총 샘플 수: ~10,000건 (1 epoch 학습용)
  - 구성: Train (80%), Val (10%), Test (10%)
  - 포맷: JSONL (EXAONE Chat Template 형식)
- **전처리 내용**:
  - 개인정보(PII) 마스킹 처리 (정규표현식)
  - 중복 데이터 및 20자 미만 짧은 민원 필터링
  - EXAONE Chat Template(`[|system|]`, `[|user|]`, `[|assistant|]`) 적용
- **데이터 위치**: `data/processed/civil_complaint_train.jsonl`
