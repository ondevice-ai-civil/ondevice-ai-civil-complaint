# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GovOn은 온디바이스 AI 민원 분석·처리 시스템이다. EXAONE-Deep-7.8B 기반 모델을 QLoRA 파인튜닝 → AWQ 양자화하여 FastAPI + vLLM으로 서빙하며, FAISS 기반 RAG로 유사 민원 검색을 지원한다.

## Git Commit Rules

- **Co-Authored-By 절대 금지**: 커밋 메시지에 `Co-Authored-By: Claude` 또는 AI 관련 co-author를 절대 포함하지 않는다.
- 커밋 메시지는 **한글**로 작성한다.
- Conventional Commits 형식을 따른다: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`

## PR Rules

- PR 본문에 `Claude Code` 또는 AI 생성 표시를 포함하지 않는다.
- PR 대상 브랜치는 `main`이다. `main`은 직접 push 금지, PR을 통해서만 머지.
- 브랜치 네이밍: `feat/이슈번호-설명`, `fix/이슈번호-설명`, `docs/이슈번호-설명`, `chore/이슈번호-설명`

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"          # Dev tools (pytest, black, isort, flake8, mypy)
pip install -e ".[inference]"    # Inference server

# Run tests
pytest tests/ -v --cov=src --cov-report=term-missing
pytest tests/test_inference/                        # Inference module only
pytest tests/test_data_collection_preprocessing/    # Data pipeline only

# Run single test file
pytest tests/test_inference/test_schemas.py -v

# Lint & format
black --line-length 100 src/
isort --profile black src/
flake8 src/

# Start inference server
uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000 --reload

# Data pipeline
python -m src.data_collection_preprocessing.pipeline --mode full
```

## Architecture

```
src/
├── data_collection_preprocessing/   # AI Hub 수집 → PII 마스킹 → EXAONE 형식 변환 → AWQ 캘리브레이션
├── training/                        # QLoRA 파인튜닝 (SFTTrainer, WandB 연동)
├── quantization/                    # AWQ 양자화 (W4A16g128), LoRA 병합
├── inference/                       # FastAPI 서빙 (핵심 모듈)
│   ├── api_server.py               # vLLMEngineManager, 엔드포인트, 보안 미들웨어
│   ├── retriever.py                # FAISS IndexFlatIP + multilingual-e5-large 임베딩
│   ├── index_manager.py            # MultiIndexManager (CASE/LAW/MANUAL/NOTICE)
│   ├── schemas.py                  # Pydantic 요청/응답 모델
│   ├── vllm_stabilizer.py          # EXAONE용 transformers 런타임 패치
│   └── db/                         # SQLAlchemy ORM, Alembic 마이그레이션
└── evaluation/                     # 모델 평가 스크립트
```

### Key Patterns

- **vLLMEngineManager**: AsyncLLMEngine + CivilComplaintRetriever + MultiIndexManager 생명주기 관리
- **보안 레이어**: `verify_api_key()` (X-API-Key), slowapi Rate Limiting, 환경변수 기반 CORS
- **Prompt Injection 방어**: `_escape_special_tokens()`로 EXAONE 특수 토큰 이스케이프
- **EXAONE 채팅 템플릿**: `[|system|]...[|endofturn|][|user|]...[|assistant|]`
- **임베딩**: multilingual-e5-large (dim=1024), `query:` prefix 필수

### Inference Server Environment Variables

```
MODEL_PATH, DATA_PATH, INDEX_PATH, GPU_UTILIZATION, MAX_MODEL_LEN, API_KEY, CORS_ORIGINS
```

## Code Style

- Python 3.10+, black (line-length=100), isort (black profile), type hints 사용
- 로깅은 `loguru.logger` 사용 (`print()` 금지)
- API 에러는 내부 정보 노출 방지 (스택 트레이스를 클라이언트에 반환하지 않음)
