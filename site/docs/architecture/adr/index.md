# ADR (Architecture Decision Records)

GovOn 프로젝트에서 내린 주요 아키텍처 결정 사항을 기록합니다. 각 ADR은 결정의 배경, 검토 후보, 선정 근거, 그리고 결정에 따른 영향을 포함합니다.

---

## ADR이란

Architecture Decision Record(ADR)는 소프트웨어 아키텍처에서 중요한 결정을 내릴 때 그 맥락, 대안, 근거를 기록하는 문서입니다. GovOn은 온프레미스 AI 시스템이라는 특수한 환경에서 다수의 기술 선택을 해야 했으며, 각 결정의 이유와 트레이드오프를 투명하게 남겨두고 있습니다.

---

## ADR 목록

| ADR | 제목 | 상태 | 핵심 결정 |
|-----|------|------|-----------|
| [ADR-001](#adr-001-exaone-deep-78b-모델-선정) | EXAONE-Deep-7.8B 모델 선정 | Accepted | 한국어 특화 + Apache 2.0 + 추론 체인 내장 |
| [ADR-002](#adr-002-awq-w4a16g128-양자화-방식-선정) | AWQ W4A16g128 양자화 방식 선정 | Accepted | 15.6GB에서 4.94GB로 68.3% 감소, vLLM 네이티브 지원 |
| [ADR-003](#adr-003-vllm-추론-서빙-엔진-선정) | vLLM 추론 서빙 엔진 선정 | Accepted | PagedAttention + AsyncLLMEngine + AWQ 네이티브 로드 |
| [ADR-004](#adr-004-faiss-기반-벡터-검색-엔진-선정) | FAISS 기반 벡터 검색 엔진 선정 | Accepted | 폐쇄망 적합, pip 설치만으로 사용, CPU 전용 동작 |

---

## ADR-001: EXAONE-Deep-7.8B 모델 선정

**상태**: Accepted

GovOn의 기반 LLM을 선정하는 결정입니다. EXAONE-Deep-7.8B, Qwen2.5-7B, Llama-3-8B-Korean 세 후보를 비교 검토했습니다.

**핵심 선정 근거**:

- 한국어 사전학습 데이터 비중이 높아 민원 도메인 파인튜닝 시 수렴 속도와 최종 품질이 우수
- `<thought>` 태그 기반 추론 체인이 민원 분류 근거 제시에 적합 (공무원이 AI 답변의 논리 과정을 확인 가능)
- Apache 2.0 라이선스로 공공기관 배포에 법적 제약 없음
- 7.8B 파라미터는 AWQ 양자화 시 약 4~5GB VRAM으로 서빙 가능

**트레이드오프**: EXAONE 모델 생태계가 Llama/Qwen 대비 작아 커뮤니티 지원이 제한적이며, `trust_remote_code=True`로 인해 transformers 버전 업데이트 시 호환성 패치(`vllm_stabilizer.py`)가 필요합니다.

---

## ADR-002: AWQ W4A16g128 양자화 방식 선정

**상태**: Accepted

파인튜닝된 EXAONE 모델을 16~24GB GPU 환경에서 서빙하기 위한 양자화 방식을 결정합니다. AWQ, GPTQ, BitsAndBytes INT8/NF4를 비교했습니다.

**핵심 선정 근거**:

- 활성화 분포를 고려한 양자화로 품질 저하 최소화
- vLLM이 AWQ 모델을 네이티브로 로드하여 GEMM 커널로 추론 (별도 변환 불필요)
- 도메인 캘리브레이션 데이터(512샘플)로 민원 텍스트에 특화된 양자화 품질 확보
- 모델 크기 15.6GB에서 4.94GB로 약 3.5배 축소

**트레이드오프**: AutoAWQ CUDA 빌드 의존성, 4비트 양자화로 인한 소폭의 품질 저하 (특히 숫자 추론, 법령 조항 인용)

---

## ADR-003: vLLM 추론 서빙 엔진 선정

**상태**: Accepted

AWQ 양자화 모델을 서빙할 추론 엔진을 결정합니다. vLLM, Ollama, TGI, TorchServe를 비교했습니다.

**핵심 선정 근거**:

- PagedAttention으로 16GB GPU에서도 KV 캐시를 효율적으로 관리하여 동시 요청 처리
- `AsyncLLMEngine`이 FastAPI의 `async/await` 패턴과 직접 통합
- AWQ 모델 네이티브 지원으로 양자화 출력물을 변환 없이 바로 서빙
- Continuous batching으로 요청 도착 시 즉시 처리 시작

**트레이드오프**: CUDA 강한 의존성, EXAONE 모델과의 호환성을 위한 런타임 패치 필요, `enforce_eager=True`로 인한 CUDA graph 최적화 비활성화

---

## ADR-004: FAISS 기반 벡터 검색 엔진 선정

**상태**: Accepted

RAG 파이프라인의 벡터 검색 엔진을 결정합니다. FAISS, Chroma, Qdrant, Milvus, pgvector를 비교했습니다.

**핵심 선정 근거**:

- `pip install faiss-cpu` 단일 패키지로 설치 완료, 외부 서버/프로세스 의존 없음
- `.faiss` 파일 + JSON 메타데이터만으로 완전한 오프라인 검색 가능
- `faiss-cpu`로 GPU 없이 동작하여 vLLM에 GPU 자원을 온전히 할당
- IndexFlatIP(brute-force)로 정확도 100% 보장, 10만건 이상 시 IndexIVFFlat 자동 전환

**트레이드오프**: 개별 벡터 삭제/수정 불가 (전체 재빌드 필요), 메타데이터 기반 필터링은 애플리케이션 레벨 후처리

---

## ADR 작성 가이드

새로운 아키텍처 결정을 기록할 때는 `docs/adr/` 디렉토리에 다음 형식으로 파일을 생성합니다.

**파일명**: `ADR-NNN-짧은-설명.md`

**필수 섹션**:

1. **Status**: Proposed / Accepted / Deprecated / Superseded
2. **Context**: 결정이 필요한 배경과 제약 조건
3. **검토 후보**: 비교 대상과 각각의 장단점
4. **Decision**: 최종 결정과 핵심 근거
5. **Consequences**: 긍정적 영향, 부정적 영향, 향후 고려사항
