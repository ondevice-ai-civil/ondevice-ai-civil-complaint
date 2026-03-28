# 개발 규칙

이 문서는 GovOn 프로젝트에 기여할 때 따라야 하는 브랜치 전략, 커밋 컨벤션, PR 규칙, 코드 스타일을 안내한다.

---

## 브랜치 전략

GovOn은 **GitHub Flow** 기반의 `main` 단일 브랜치 전략을 사용한다.

```
main (프로덕션)
 ├── feat/42-add-law-index
 ├── fix/55-faiss-metadata-error
 ├── docs/60-api-reference
 └── chore/70-ci-cache
```

| 브랜치 | 용도 | 규칙 |
|--------|------|------|
| `main` | 프로덕션 (안정 버전) | 직접 push 금지, PR 머지만 허용 |
| `feat/*` | 새 기능 개발 | `feat/이슈번호-설명` 형식 |
| `fix/*` | 버그 수정 | `fix/이슈번호-설명` 형식 |
| `docs/*` | 문서 작업 | `docs/이슈번호-설명` 형식 |
| `chore/*` | 설정/인프라 작업 | `chore/이슈번호-설명` 형식 |

### 작업 흐름

```bash
# 1. main에서 최신 코드를 가져온다
git checkout main
git pull origin main

# 2. 이슈 번호를 포함한 브랜치를 생성한다
git checkout -b feat/42-add-law-index

# 3. 작업 후 커밋한다
git add <변경된 파일>
git commit -m "feat: 법령 인덱스 검색 엔드포인트 추가"

# 4. 원격에 push한다
git push origin feat/42-add-law-index

# 5. GitHub에서 main 대상으로 PR을 생성한다
```

---

## 커밋 컨벤션

[Conventional Commits](https://www.conventionalcommits.org/) 형식을 따른다. 커밋 메시지는 **한글**로 작성한다.

### 형식

```
<type>: <설명>

[선택 본문]

[선택 꼬리말]
```

### 사용 가능한 type

| type | 용도 | 예시 |
|------|------|------|
| `feat` | 새 기능 추가 | `feat: QLoRA 학습 스크립트 구현` |
| `fix` | 버그 수정 | `fix: AWQ 양자화 OOM 해결` |
| `docs` | 문서 추가/수정 | `docs: API 명세서 업데이트` |
| `style` | 코드 포매팅 (로직 변경 없음) | `style: black 포맷터 적용` |
| `refactor` | 리팩토링 (기능 변경 없음) | `refactor: 추론 엔진 구조 개선` |
| `test` | 테스트 추가/수정 | `test: 민원 분류 단위 테스트 추가` |
| `chore` | 빌드, CI, 의존성 관리 | `chore: GitHub Actions 워크플로우 추가` |
| `perf` | 성능 개선 | `perf: vLLM 배치 추론 속도 최적화` |

### 좋은 커밋 메시지 예시

```bash
# 구체적이고 명확한 메시지
git commit -m "feat: 법령 인덱스 검색 엔드포인트 추가"
git commit -m "fix: FAISS 인덱스 메타데이터 경로 오류 수정"
git commit -m "refactor: vLLMEngineManager 초기화 로직 분리"

# 본문이 필요한 경우
git commit -m "fix: GPU OOM 발생 시 서버 크래시 방지

GPU_UTILIZATION 기본값을 0.85에서 0.8로 변경하고,
OOM 발생 시 graceful shutdown 로직을 추가한다.

Closes #55"
```

---

## PR 규칙

### PR 작성 규칙

1. **대상 브랜치**: 항상 `main`을 대상으로 PR을 생성한다.
2. **제목**: 커밋 컨벤션과 동일한 형식을 사용한다 (예: `feat: 법령 인덱스 검색 엔드포인트 추가`).
3. **본문**: PR 템플릿에 따라 작성한다.
    - 작업 배경 설명
    - 주요 변경 사항 요약
    - 테스트 결과 기록
4. **이슈 연결**: 관련 이슈를 `Closes #이슈번호`로 연결한다.
5. **리뷰어**: 최소 1명의 리뷰어를 지정한다.

### PR 체크리스트

PR을 생성하기 전에 다음 항목을 확인한다.

- [ ] 커밋 메시지가 컨벤션을 따르는가?
- [ ] 관련 이슈가 연결되어 있는가?
- [ ] 테스트가 통과하는가? (`pytest tests/ -v`)
- [ ] 린트가 통과하는가? (`black --check .` / `isort --check .` / `flake8 .`)
- [ ] 문서가 업데이트되었는가? (해당하는 경우)

### CI 자동 검사

PR을 생성하면 다음 CI 파이프라인이 자동으로 실행된다.

| 단계 | 내용 |
|------|------|
| **Lint** | Black, isort, Flake8 코드 스타일 검사 |
| **Test** | Python 3.10, 3.11 / Ubuntu latest, 22.04 매트릭스 테스트 |
| **Build** | 패키지 빌드 및 아티팩트 업로드 |
| **Security** | Bandit 정적 분석 + pip-audit 의존성 취약점 스캔 |

모든 CI 검사를 통과해야 PR을 머지할 수 있다.

---

## 코드 리뷰 가이드

### 리뷰 태그

리뷰어는 다음 태그를 사용하여 코멘트의 중요도를 표시한다.

| 태그 | 의미 | 대응 |
|------|------|------|
| `[MUST]` | 반드시 수정해야 할 사항 (보안, 버그, 성능) | 수정 후 re-review 요청 |
| `[SHOULD]` | 수정을 강하게 권장하는 사항 | 가능한 한 반영 |
| `[NITS]` | 사소한 개선 사항 (코드 스타일 등) | 선택적 반영 |
| `[QUESTION]` | 이해를 위한 질문 | 답변 후 진행 |

### 리뷰 기준

1. **정확성**: 코드가 의도한 대로 동작하는가?
2. **가독성**: 코드가 이해하기 쉬운가?
3. **테스트**: 적절한 테스트가 포함되어 있는가?
4. **보안**: 민감 정보(API 키, 모델 가중치 경로)가 노출되지 않는가?

---

## 코드 스타일

### 도구 설정

| 항목 | 도구 | 설정 |
|------|------|------|
| 포매터 | `black` | line-length=100, target-version py310~py312 |
| import 정렬 | `isort` | black profile, line_length=100 |
| 린터 | `flake8` | 기본 설정 |
| 타입 검사 | `mypy` | python_version=3.10, warn_return_any=true |

### 포매팅 및 린트 실행

```bash
# 코드 포매팅
black --line-length 100 src/

# import 정렬
isort --profile black src/

# 린트 검사
flake8 src/

# 타입 검사
mypy src/
```

### 코드 작성 규칙

**타입 힌트를 적극 활용한다.**

```python
def classify_complaint(text: str, model_name: str = "exaone") -> dict:
    """민원 텍스트를 분류합니다.

    Args:
        text: 분류할 민원 텍스트
        model_name: 사용할 모델 이름

    Returns:
        분류 결과를 담은 딕셔너리
    """
    ...
```

**로깅은 `loguru.logger`를 사용한다.** `print()` 함수는 사용하지 않는다.

```python
# 올바른 사용
from loguru import logger

logger.info("서버 기동 완료: port={}", port)
logger.error("모델 로딩 실패: {}", str(e))

# 잘못된 사용
print("서버 기동 완료")  # 금지
```

**API 에러는 내부 정보를 노출하지 않는다.** 스택 트레이스를 클라이언트에 반환하지 않는다.

```python
# 올바른 사용
from fastapi import HTTPException

try:
    result = await engine.generate(prompt)
except Exception as e:
    logger.error("생성 실패: {}", str(e))
    raise HTTPException(status_code=500, detail="요청 처리 중 오류가 발생했습니다.")

# 잘못된 사용 - 내부 정보 노출
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))  # 금지
```

### 파일 구조

모듈별로 디렉토리를 분리한다.

```
src/
├── data_collection_preprocessing/   # 데이터 수집 및 전처리
├── training/                        # QLoRA 파인튜닝
├── quantization/                    # AWQ 양자화
├── inference/                       # FastAPI 추론 서버
│   ├── api_server.py               # 엔드포인트, 보안 미들웨어
│   ├── retriever.py                # FAISS 벡터 검색
│   ├── index_manager.py            # 멀티 인덱스 관리
│   ├── schemas.py                  # Pydantic 요청/응답 모델
│   ├── vllm_stabilizer.py          # EXAONE 런타임 패치
│   └── db/                         # SQLAlchemy ORM, Alembic 마이그레이션
└── evaluation/                     # 모델 평가
```

설정 파일은 `configs/` 디렉토리에, 노트북은 `notebooks/` 디렉토리에 관리한다.

---

## 데이터 파이프라인

`src/data_collection_preprocessing/` 모듈은 다음 단계를 순차적으로 처리한다.

1. AI Hub 원본 데이터 수집
2. PII(개인정보) 마스킹
3. EXAONE 채팅 템플릿 형식으로 변환
4. AWQ 캘리브레이션 데이터 생성

### 전체 파이프라인 실행

```bash
python -m src.data_collection_preprocessing.pipeline --mode full
```

파이프라인이 완료되면 `data/processed/` 디렉토리에 JSONL 형식의 학습 데이터가 생성된다.

### EXAONE 채팅 템플릿 형식

변환된 데이터는 다음 형식을 따른다.

```
[|system|]시스템 프롬프트[|endofturn|]
[|user|]민원 내용[|endofturn|]
[|assistant|]답변 내용[|endofturn|]
```

---

## 학습 및 양자화

### QLoRA 파인튜닝

```bash
python -m src.training.train_qlora \
  --train_path data/processed/v2_train.jsonl \
  --val_path data/processed/v2_val.jsonl \
  --output_dir models/checkpoints/exaone-civil-qlora \
  --epochs 3 \
  --batch_size 4 \
  --lr 2e-4
```

주요 학습 파라미터:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--model_id` | `LGAI-EXAONE/EXAONE-Deep-7.8B` | 베이스 모델 ID |
| `--epochs` | `3` | 학습 에폭 수 |
| `--batch_size` | `4` | 배치 크기 |
| `--lr` | `2e-4` | 학습률 |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--max_seq_length` | `2048` | 최대 시퀀스 길이 |

### AWQ 양자화

QLoRA 파인튜닝이 완료되면 LoRA 어댑터를 베이스 모델에 병합한 뒤 AWQ 양자화를 수행한다.

```bash
# 1. LoRA 병합
python -m src.quantization.merge_lora

# 2. AWQ 양자화 (W4A16g128)
python -m src.quantization.quantize_awq
```

### 모델 평가

```bash
# vLLM 기반 평가
python -m src.evaluation.evaluate_m3_vllm

# AWQ 양자화 모델 평가
python -m src.evaluation.evaluate_m3_autoawq
```

---

## 이슈 작성 가이드

### 버그 리포트

- 재현 가능한 단계를 명확히 기술한다.
- 예상 동작과 실제 동작을 구분하여 기록한다.
- 환경 정보를 포함한다 (OS, Python 버전, GPU 등).

### 기능 요청

- 해결하려는 문제를 먼저 설명한다.
- 제안하는 해결 방법을 기술한다.
- 대안이 있다면 함께 기록한다.

이슈 템플릿을 활용하여 작성한다: 기능 요청, 버그 리포트, 문서 작업.

---

## 다음 단계

- [시작하기](getting-started.md) -- 로컬 환경 구축 및 서버 실행
- [트러블슈팅](troubleshooting.md) -- 자주 발생하는 문제 해결
- [보안 정책](security.md) -- 취약점 보고 및 보안 가이드
