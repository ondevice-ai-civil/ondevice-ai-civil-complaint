# 시작하기

GovOn MVP는 로컬 daemon runtime을 중심으로 동작한다. 현재 저장소에는 shell-first runtime과 관련 테스트/문서가 포함되어 있으며, 최종 `govon` 설치 경험은 별도 packaging 작업으로 이어진다.

## 기본 환경

- Python 3.10+
- Node.js 22
- `uv` 또는 `pip`

## 로컬 개발 환경 준비

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

CI와 같은 환경을 더 가깝게 맞추려면 `uv` 기반 설치를 사용할 수 있다.

```bash
uv sync --extra dev --extra inference --extra database
```

## 주요 로컬 검증 명령

```bash
pytest tests/test_inference -q
python -m uvicorn src.inference.api_server:app --host 127.0.0.1 --port 8000
cd site && mkdocs build --strict
```

## 런타임 확인

- `GET /health`가 200을 반환해야 한다.
- `session_store.driver`는 `sqlite`로 보고되어야 한다.
- `SKIP_MODEL_LOAD=true` 환경에서는 검색/생성 일부가 graceful degradation을 보여도 된다.
