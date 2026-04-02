# API Specification (v1.0)
# On-Device AI 민원 분석 및 처리 시스템

**문서 정보**
- **작성일**: 2026-03-05
- **기반 문서**: PRD v2.0
- **인프라**: FastAPI + vLLM + FAISS (Docker)

---

## 1. 개요
본 시스템은 지자체 공무원의 민원 처리를 지원하는 AI 백엔드 서비스를 제공합니다. 모든 API는 폐쇄망(On-Premise) 환경에서 구동되며, 보안 및 성능 최적화를 위해 설계되었습니다.

- **Base URL**: `/api/v1`
- **Response Format**: `application/json`

---

## 2. API 명세

### 2.1 민원 분석 및 답변 (Complaints & Generation)

#### [POST] 민원 카테고리 자동 분류
민원 텍스트를 분석하여 시스템에 정의된 가장 적합한 카테고리를 예측합니다.
- **Endpoint**: `/complaints/analyze/category`
- **Request Body**:
  ```json
  {
    "content": "string (민원 본문, 최소 20자)"
  }
  ```
- **Response**:
  ```json
  {
    "category_id": 12,
    "category_name": "도로/교통",
    "confidence": 0.94,
    "reasoning": "보도블록 파손 및 안전 우려 사항이 핵심 요청 사항임"
  }
  ```

#### [POST] 공문서 초안 생성
EXAONE-Deep-7.8B 모델을 사용하여 법률, 매뉴얼, 공시정보를 참고한 공문서 초안을 생성합니다.
- **Endpoint**: `/complaints/analyze/generate-public-doc`
- **Request Body**:
  ```json
  {
    "content": "string (민원 본문)",
    "doc_type": "string (예: official_document, press_release)",
    "options": {
      "temperature": 0.6,
      "max_tokens": 1024
    }
  }
  ```
- **Response**:
  ```json
  {
    "document_draft": "제목: ...",
    "formatted_html": "<p>제목: ...</p>",
    "inference_stats": {
      "time_seconds": 2.1,
      "tokens_per_sec": 120.5
    }
  }
  ```

#### [POST] 민원 답변 초안 생성 (Reasoning 포함)
EXAONE-Deep-7.8B 모델을 사용하여 민원 분석 결과와 RAG 검색 결과를 통합한 답변 초안을 생성합니다.
- **Endpoint**: `/complaints/analyze/generate-civil-response`
- **Request Body**:
  ```json
  {
    "content": "string (민원 본문)",
    "complaint_id": "string (선택 사항)",
    "category": "string (선택 사항)",
    "context_ids": ["integer (RAG 검색된 사례 ID 목록)"],
    "options": {
      "temperature": 0.6,
      "max_tokens": 1024
    }
  }
  ```
- **Response**:
  ```json
  {
    "answer_draft": "안녕하세요. 요청하신 보도블록 보수 건에 대하여...",
    "thought_process": "<thought>\n민원 분석... 답변 구성... 최종 답변 작성\n</thought>",
    "inference_stats": {
      "time_seconds": 2.1,
      "tokens_per_sec": 120.5
    }
  }
  ```

### 2.2 유사 사례 검색 (Similarity Search - RAG)

#### [POST] 유사 민원 및 답변 검색
FAISS 벡터 인덱스를 사용하여 의미적으로 가장 유사한 과거 처리 사례를 검색합니다.
- **Endpoint**: `/search/similar-cases`
- **Request Body**:
  ```json
  {
    "query": "string (현재 민원 내용)",
    "top_k": 3,
    "threshold": 0.7
  }
  ```
- **Response**:
  ```json
  {
    "total_found": 3,
    "results": [
      {
        "id": 505,
        "title": "가로수 보수 요청",
        "content": "...",
        "answer": "...",
        "similarity_score": 0.89
      }
    ]
  }
  ```

### 2.3 피드백 및 데이터 관리 (Feedback)

#### [POST] AI 답변 품질 피드백 수집
시스템 개선을 위해 생성된 답변에 대한 담당자의 평가와 최종 수정본을 수집합니다.
- **Endpoint**: `/feedback/submit`
- **Request Body**:
  ```json
  {
    "draft_id": "integer",
    "rating": "good | normal | bad",
    "final_content": "string (최종적으로 발송된 답변)",
    "officer_comment": "string"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "stored_id": 890
  }
  ```

### 2.4 시스템 관리 (System)

#### [GET] 서버 헬스체크 및 리소스 상태
vLLM 서빙 상태 및 GPU 메모리 점유율 등 시스템 건강 상태를 확인합니다.
- **Endpoint**: `/system/health`
- **Response**:
  ```json
  {
    "status": "online",
    "engine": "vLLM 0.4.0",
    "model": "EXAONE-Deep-7.8B-AWQ",
    "gpu_memory_usage": "7.5GB / 16GB",
    "active_batch_size": 0
  }
  ```

---

## 3. 공통 에러 응답

| HTTP 코드 | 설명 | 메시지 예시 |
|-----------|------|------------|
| 400 | 필수 파라미터 누락 | `{"error": "content is required"}` |
| 422 | 데이터 형식 오류 | `{"error": "content must be at least 20 chars"}` |
| 500 | vLLM 서버 통신 오류 | `{"error": "inference engine is not responding"}` |
| 503 | GPU 리소스 부족 | `{"error": "server busy, please retry"}` |

---
**문서 끝**
