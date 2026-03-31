# ADR-007: 행정 특화 모델(v3) 파인튜닝 전략

**상태**: Proposed  
**작성일**: 2026-03-31

---

## 1. Context
범용 LLM은 대한민국 행정 문서 특유의 **개조식(Short sentences, bullet points)** 표현과 전문 법률/행정 용어 처리에 한계가 있습니다. 고품질 공문서 작성을 위해서는 실제 정부 문서를 학습한 특화 모델이 필요합니다.

## 2. Decision
행정안전부 **'정부 공문서 AI 학습데이터'**를 활용하여 EXAONE-Deep-7.8B를 파인튜닝합니다.

### A. 학습 데이터셋 구성 (3-Way Strategy)
1. **Corpus Tuning**: 보도자료, 정책보고서 등 3,885건의 원문을 학습하여 행정 문체(Style)와 도메인 지식을 내재화.
2. **Task Tuning (Q&A)**: 정부가 공식 제작한 4만 건의 Q&A 쌍을 학습하여 정확한 정보 추출 및 답변 능력 강화.
3. **Task Tuning (Rewrite)**: "평서문 -> 개조식" 변환 데이터셋을 학습하여 공문서 초안 작성 기능 최적화.

### B. 학습 방식
- **방법**: 4-bit QLoRA (Memory-efficient Fine-tuning)
- **Target**: Attention layers & Output head
- **Template**: EXAONE Chat Template (v2 호환 유지)

## 3. Consequences
- **장점**: 
  - 공무원이 수정할 필요가 거의 없는 '진짜 공문서 체' 결과물 생성 가능.
  - 행정 전문 용어에 대한 할루시네이션(환각) 감소.
- **단점**: 
  - 특화 데이터에 과적합(Overfitting)되어 일상 대화 능력이 저하될 우려가 있음 (범용 데이터 혼합 학습 권장).
