# M3: 고도화 및 최적화 (Week 9-12)

**기간**: 2026-04-28 ~ 2026-05-22
**상태**: 부분 완료 (40%) - 모델 최적화 완료, RAG/FAISS/Docker/백엔드 미구현
**최종 수정**: 2026-03-09

---

## 진행 현황 요약

M3 단계에서는 M2 MVP에서 미달했던 핵심 KPI를 대폭 개선했습니다.
vLLM 0.17.0 도입으로 추론 속도를 9.29초에서 2.43초로 단축하고,
분류 정확도를 2%에서 90%로 끌어올렸습니다.
그러나 벡터 검색(FAISS), RAG 파이프라인, 전용 분류기(KR-ELECTRA),
백엔드 API, 프론트엔드 UI, Docker 배포는 아직 미구현 상태입니다.

---

## 주요 성과 (M2 대비 개선)

| 지표 | M2 MVP | M3 최적화 후 | 개선폭 |
|------|--------|-------------|--------|
| 분류 정확도 | 2.00% | **90.0%** | +88.0%p |
| 추론 속도 (Avg) | 9.291s | **2.43s** | -6.86s |
| BERTScore F1 | - | **46.05** | 신규 확보 |
| GPU VRAM | 4.95 GB | **4.17 GB** | -0.78 GB |

---

## 산출물 체크리스트

### Week 9: 벡터 검색 시스템 구축
- [ ] embedding_model_selection.md - 임베딩 모델 선정 (계획: multilingual-e5-large)
- [ ] embeddings/ - 민원 데이터 임베딩 생성
- [ ] faiss_index/ - FAISS 벡터 인덱스 구축
- [ ] search_api_test.md - 검색 API 테스트 결과

### Week 10: RAG 파이프라인 및 분류 기능
- [ ] rag_pipeline.py - RAG 파이프라인 코드
- [ ] classification_model/ - KR-ELECTRA 기반 전용 분류기
- [ ] classification_accuracy.md - 분류 정확도 리포트

### Week 11: 모델 최적화 및 vLLM 서빙
- [x] vLLM 0.17.0 서버 구축 및 검증 (PR #24)
- [x] src/inference/vllm_stabilizer.py - vLLM 안정화 스크립트
- [x] src/evaluation/evaluate_m3_vllm_final.py - M3 최종 평가 스크립트
- [x] FINAL_M3_COMPLETION_REPORT.md - M3 완료 리포트
- [ ] benchmark_report.md - 정식 벤치마크 리포트
- [ ] memory_usage.md - 메모리 사용량 상세 리포트

### Week 12: 백엔드/프론트엔드 및 Docker 배포
- [ ] FastAPI 백엔드 서버 구축
- [ ] Figma UI/UX 디자인 (동서대 디자인학부 협업)
- [ ] Figma MCP 기반 React/Next.js 웹 UI 구현
- [ ] Dockerfile - Docker 빌드 파일
- [ ] docker-compose.yml - Docker Compose 설정
- [ ] deployment_test.md - 배포 테스트 리포트

---

## 완료 기준

- [x] 분류 정확도 >= 85% (실측 90.0%)
- [ ] 유사 사례 검색 기능 (Recall@5 >= 80%) - FAISS 미구현
- [ ] 답변 생성 속도 < 5초 (p95) - vLLM 기반 2.43초 달성, 정식 벤치마크 필요
- [ ] Docker 컨테이너 배포 성공 - 미구현
- [ ] 모든 핵심 기능 구현 완료 - 백엔드/프론트엔드(Figma MCP 기반) 미구현

---

## 기술적 이슈 및 해결

1. **vLLM 토크나이저 충돌**: ExaoneTokenizer 인식 오류 -> PreTrainedTokenizerFast 강제 매핑
2. **구조적 NoneType 에러**: rope_parameters 및 get_interface 누락 -> 로컬 소스 하드 패치
3. **분류 정확도 비정상**: `<thought>` 태그 완벽 분리 파서 + `add_generation_prompt=True` 적용

---

## 관련 GitHub 이력

| 항목 | 참조 |
|------|------|
| PR #24 | M2 MVP Final Report & M3 Optimization Complete |
| Issue #20 | 추론 속도 개선: vLLM 배포로 p50 < 2초 달성 (Closed) |
| Issue #21 | 민원 분류 정확도 평가 방법론 개선 (Closed) |
| Issue #22 | 답변 생성 품질 개선 (Closed) |
| Issue #23 | M2 MVP KPI 미달 지표 개선 트래킹 (Closed) |
| Issue #25 | 로드맵 최종 달성 및 최적화 완료 (Closed) |
| Issue #26 | EXAONE-Deep-7.8B 로드맵 최종 달성 (Closed) |
| Issue #27 | RAG 통합, FAISS 벡터 검색 및 전용 분류기 구축 (Open) |

---

## 남은 핵심 과제 (M3 잔여)

### 기존 과제 (인프라/시스템)
1. **FAISS 벡터 검색 시스템** (#53): multilingual-e5-large 임베딩 + FAISS 인덱스
2. **RAG 파이프라인** (#54): 검색 결과 기반 답변 생성 통합
3. **KR-ELECTRA 전용 분류기** (#55): 경량 분류기로 전처리 단계 분류
4. **FastAPI 백엔드** (#56): vLLM + RAG 통합 API 서버
5. **Figma MCP 기반 프론트엔드** (#57): 동서대 디자인학부 협업 Figma 디자인 -> React/Next.js 웹 UI
6. **Docker 컨테이너화** (#58): vLLM + FastAPI + React/Next.js 멀티 컨테이너

### 신규 추가 과제 (2026-03-10) - 모델 품질 고도화

M2 미달성 KPI 갭 분석을 기반으로 다음 고도화 이슈가 추가되었습니다:

7. **QLoRA 하이퍼파라미터 최적화** (#67): Rank(8/32/64), LR(1e-4/5e-5), Epoch(2/3) 체계적 탐색
   - 근거: M2에서 EXP-001 Baseline만 실행, EXP-002/003 미실행
   - 기대: BLEU/ROUGE-L 개선의 핵심 레버

8. **답변 생성 품질 고도화** (#68): BLEU >=30, ROUGE-L >=40 달성 전략
   - 프롬프트 엔지니어링, 디코딩 전략 최적화, 평가 방법론 개선
   - 근거: BLEU 17.32 (목표 대비 42.3% 미달), ROUGE-L 18.28 (54.3% 미달)

9. **추론 속도 추가 최적화** (#69): Avg 2.43s -> p50 < 2s 달성
   - vLLM 파라미터 튜닝, Speculative Decoding, Prefix Caching
   - 근거: PRD KPI-001 목표 p50 < 1s (AWQ 기준) 미달

10. **학습 데이터 증강 및 테스트셋 품질 개선** (#70): 데이터 편향 해소
    - 카테고리별 균형 테스트셋 재구성, PII 마스킹 영향 분석
    - 근거: 테스트셋이 other(금융) 카테고리에 편중, 평가 신뢰도 저하

---

## 미달성 KPI 갭 분석 (M2 -> M3 현재)

| 지표 | M2 측정값 | M3 현재값 | 목표 | 달성 여부 | 추가 고도화 |
|------|-----------|-----------|------|-----------|------------|
| 분류 정확도 | 2% | **90%** | >= 85% | 달성 | #55 전용 분류기로 추가 향상 |
| BLEU | 17.32 | 미재측정 | >= 30 | **미달** | #67, #68, #70 |
| ROUGE-L | 18.28 | 미재측정 | >= 40 | **미달** | #67, #68, #70 |
| BERTScore F1 | - | **46.05** | 베이스라인 | 확보 | #68 추가 향상 |
| 추론 속도 Avg | 9.29s | **2.43s** | < 2s | **근접 미달** | #69 |
| GPU VRAM | 4.95GB | **4.17GB** | < 8GB | 달성 | - |
| 모델 크기 | 4.94GB | - | < 5GB | 달성 | - |

### 고도화 의존성 흐름

```
#70 데이터 증강 ──┐
                  ├──> #67 QLoRA HP 최적화 ──> #68 답변 품질 고도화
                  │                              │
#54 RAG 파이프라인 ──────────────────────────────┘
                                                  │
#69 추론 속도 최적화 <────────────────────────────┘
```

---

**작성일**: 2026-03-05
**최종 수정일**: 2026-03-10
