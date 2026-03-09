# M1: 기획 및 설계 (Week 1-4)

**기간**: 2026-03-03 ~ 2026-03-28
**상태**: 완료 (90%)
**최종 수정**: 2026-03-09

---

## 진행 현황 요약

M1 단계에서는 프로젝트의 기반이 되는 기획, 설계, 데이터 수집 환경을 구축했습니다.
킥오프 문서 작성, 시스템 아키텍처 설계, 크롤러 프로토타입 개발이 완료되었으며,
일부 문서(킥오프 회의록, 기술 스택 보고서)와 데이터 전처리 관련 산출물이 미완성 상태입니다.

---

## 산출물 체크리스트

### Week 1: 프로젝트 킥오프
- [x] 01_Kickoff/draft.md - 문제 정의서 (Draft)
- [x] 01_Kickoff/toc.md - PRD 표준 목차 (TOC)
- [x] 01_Kickoff/exaone_analysis.md - EXAONE-Deep-7.8B 모델 분석
- [ ] 01_Kickoff/kickoff_meeting_notes.md - 킥오프 회의록
- [ ] 01_Kickoff/tech_stack_report.md - 기술 스택 선정 보고서

### Week 2: 시스템 설계
- [x] 02_System_Design/architecture_diagram.png - 시스템 아키텍처
- [x] 02_System_Design/database_erd.png - ERD
- [x] 02_System_Design/api_specification.md - API 명세서

### Week 3: 데이터 수집
- [x] 03_Data_Collection/crawling_targets.md - 크롤링 대상 목록
- [x] 03_Data_Collection/crawler_prototype/ - 크롤러 초안
- [x] 03_Data_Collection/dataset_and_environment_specs.md - 데이터셋 및 학습 환경 명세서
- [ ] 03_Data_Collection/preprocessing_script.py - 전처리 스크립트
- [ ] 03_Data_Collection/data_quality_report.md - 데이터 품질 리포트

### Week 4: 데이터 수집 실행
- [x] src/data_collection_preprocessing/ - 데이터 수집 및 전처리 모듈 (PR #9)
  - aihub_collector.py - AI Hub 데이터 수집기
  - data_preprocessor.py - 데이터 전처리기
  - pii_masking.py - 개인정보 비식별화
  - calibration_dataset.py - 캘리브레이션 데이터셋 생성
  - pipeline.py - 통합 파이프라인

---

## 완료 기준

- [x] 10,000건+ 민원 데이터 수집 (AI Hub 71852, 71844 데이터셋 활용)
- [x] 개인정보 비식별화 처리 완료 (pii_masking.py)
- [x] 시스템 아키텍처 문서 작성 완료

---

## 관련 GitHub 이력

| 항목 | 참조 |
|------|------|
| PR #9 | feat: 데이터 수집 및 전처리 모듈 추가 |
| PR #8 | feat(m1-planning): 시스템 아키텍처, ERD, API 명세 추가 |
| PR #7 | PRD, WBS, Outputs 구조 |
| PR #5 | PRD 추가 및 프로젝트 구조 업데이트 |
| Issue #6 | On-Device AI 민원 분석 및 처리 시스템 PRD (Closed) |
| Issue #15 | Preprocessing Pipeline: PII Masking (Closed) |

---

## 남은 과제

1. **킥오프 회의록 작성** - 형식적 문서, 우선순위 낮음
2. **기술 스택 선정 보고서** - PRD v2.0과 exaone_analysis.md로 대체 가능
3. **데이터 품질 리포트** - M2에서 실험 과정 중 보완

---

**작성일**: 2026-03-05
**최종 수정일**: 2026-03-09
