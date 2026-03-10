# Data Collection & Preprocessing Tests

이 폴더에는 데이터 수집 및 전처리 모듈에 대한 테스트가 포함되어 있습니다.

## 폴더 구조

```
tests/test_data_collection_preprocessing/
├── __init__.py
├── README.md                      # 이 파일
├── test_pii_masking.py            # PII 마스킹 테스트
├── test_data_preprocessor.py      # 데이터 전처리기 테스트
└── test_calibration_dataset.py    # 캘리브레이션 데이터셋 테스트
```

## 테스트 모듈 설명

### 1. test_pii_masking.py
개인정보 비식별화 모듈에 대한 테스트입니다.

| 테스트 클래스 | 설명 |
|--------------|------|
| `TestPIIMasker` | PIIMasker 클래스의 핵심 기능 테스트 |
| `TestConvenienceFunctions` | `mask_pii()`, `validate_no_pii()` 편의 함수 테스트 |
| `TestKoreanNameMasking` | 한국어 이름 탐지 및 마스킹 테스트 |

**주요 테스트 케이스:**
- 주민등록번호 마스킹 (`901231-1234567`)
- 휴대폰 번호 마스킹 (`010-1234-5678`)
- 유선전화 마스킹 (`02-1234-5678`)
- 이메일 주소 마스킹
- IP 주소 마스킹
- 신용카드 번호 마스킹
- 복합 PII 마스킹

### 2. test_data_preprocessor.py
EXAONE 형식 데이터 변환 및 전처리 테스트입니다.

| 테스트 클래스 | 설명 |
|--------------|------|
| `TestDataPreprocessor` | 데이터 전처리기 핵심 기능 테스트 |
| `TestDataQualityReport` | 데이터 품질 리포트 테스트 |
| `TestDatasetSaving` | 데이터셋 저장 (JSONL/JSON) 테스트 |

**주요 테스트 케이스:**
- 유효한 데이터 처리
- 짧은 콘텐츠 필터링
- 중복 데이터 제거
- EXAONE `<thought>` 태그 형식 출력
- 카테고리 정규화
- PII 마스킹 통합
- 데이터셋 분할 (Train/Val/Test)

### 3. test_calibration_dataset.py
AWQ 양자화용 캘리브레이션 데이터셋 생성 테스트입니다.

| 테스트 클래스 | 설명 |
|--------------|------|
| `TestCalibrationDatasetGenerator` | 캘리브레이션 샘플 생성 테스트 |
| `TestCalibrationStats` | 통계 계산 테스트 |
| `TestCalibrationDatasetSaving` | 캘리브레이션 데이터 저장 테스트 |

**주요 테스트 케이스:**
- 캘리브레이션 데이터셋 생성
- 카테고리 다양성 확보
- 토큰 수 추정
- EXAONE Chat Template 형식 변환
- 중복 제거
- JSON/TXT 형식 저장

---

## 서버 환경 설정

### 1. 시스템 요구사항

```bash
# Python 버전 확인
python --version  # Python 3.9+ 필요

# pip 업그레이드
pip install --upgrade pip
```

### 2. 의존성 설치

```bash
# 프로젝트 루트 디렉토리에서 실행
cd /path/to/GovOn

# 전체 의존성 설치 (개발 의존성 포함)
pip install -r requirements.txt

# 또는 pyproject.toml 사용 시
pip install -e ".[dev]"
```

### 3. 테스트 의존성

```bash
# 테스트에 필요한 최소 패키지
pip install pytest pytest-cov pytest-asyncio
```

---

## 테스트 실행 방법

### 전체 테스트 실행

```bash
# 프로젝트 루트에서 실행
cd /path/to/GovOn

# 전체 테스트 실행
pytest tests/test_data_collection_preprocessing/ -v

# 또는
python -m pytest tests/test_data_collection_preprocessing/ -v
```

### 개별 테스트 파일 실행

```bash
# PII 마스킹 테스트
pytest tests/test_data_collection_preprocessing/test_pii_masking.py -v

# 전처리기 테스트
pytest tests/test_data_collection_preprocessing/test_data_preprocessor.py -v

# 캘리브레이션 데이터셋 테스트
pytest tests/test_data_collection_preprocessing/test_calibration_dataset.py -v
```

### 특정 테스트 클래스/함수 실행

```bash
# 특정 클래스 실행
pytest tests/test_data_collection_preprocessing/test_pii_masking.py::TestPIIMasker -v

# 특정 테스트 함수 실행
pytest tests/test_data_collection_preprocessing/test_pii_masking.py::TestPIIMasker::test_mask_phone_mobile -v
```

### 테스트 커버리지 확인

```bash
# 커버리지 리포트 생성
pytest tests/test_data_collection_preprocessing/ --cov=src/data_collection_preprocessing --cov-report=html

# 터미널에서 커버리지 확인
pytest tests/test_data_collection_preprocessing/ --cov=src/data_collection_preprocessing --cov-report=term-missing
```

---

## CI/CD 환경 설정

### GitHub Actions 예시

```yaml
name: Data Collection Tests

on:
  push:
    paths:
      - 'src/data_collection_preprocessing/**'
      - 'tests/test_data_collection_preprocessing/**'
  pull_request:
    paths:
      - 'src/data_collection_preprocessing/**'
      - 'tests/test_data_collection_preprocessing/**'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest tests/test_data_collection_preprocessing/ -v --cov=src/data_collection_preprocessing --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

### Docker 환경에서 테스트

```dockerfile
# Dockerfile.test
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install pytest pytest-cov

COPY src/ ./src/
COPY tests/ ./tests/

CMD ["pytest", "tests/test_data_collection_preprocessing/", "-v", "--tb=short"]
```

```bash
# Docker 빌드 및 실행
docker build -f Dockerfile.test -t data-collection-tests .
docker run --rm data-collection-tests
```

---

## 트러블슈팅

### 1. ModuleNotFoundError

```
ModuleNotFoundError: No module named 'src'
```

**해결방법:**
```bash
# PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:/path/to/GovOn"

# 또는 pytest.ini 생성
echo "[pytest]
pythonpath = ." > pytest.ini
```

### 2. 임시 디렉토리 권한 오류

```
PermissionError: [Errno 13] Permission denied: '/tmp/...'
```

**해결방법:**
```bash
# 임시 디렉토리 권한 확인
ls -la /tmp

# 대체 경로 사용
export TMPDIR=/var/tmp
pytest tests/test_data_collection_preprocessing/ -v
```

### 3. 메모리 부족 (대용량 데이터 테스트 시)

```bash
# 병렬 실행 비활성화
pytest tests/test_data_collection_preprocessing/ -v --workers 1

# 또는 특정 테스트만 실행
pytest tests/test_data_collection_preprocessing/test_pii_masking.py -v
```

### 4. 인코딩 오류 (한글 데이터)

```bash
# 환경변수 설정
export LANG=ko_KR.UTF-8
export LC_ALL=ko_KR.UTF-8

# 또는
export PYTHONIOENCODING=utf-8
```

---

## 테스트 데이터 관리

테스트에서는 실제 API 호출을 하지 않고 mock 데이터를 사용합니다.

### Mock 데이터 위치
- 각 테스트 파일 내 `@pytest.fixture`로 정의된 샘플 데이터

### 테스트용 API 키 (선택사항)
통합 테스트 실행 시 `.env.test` 파일 사용:

```bash
# .env.test
AIHUB_API_KEY=test_key_for_mock
SEOUL_API_KEY=test_key_for_mock
```

```bash
# 테스트용 환경변수 로드
export $(cat .env.test | xargs)
pytest tests/test_data_collection_preprocessing/ -v
```

---

## 테스트 결과 해석

### 성공 시 출력 예시

```
========================= test session starts ==========================
platform linux -- Python 3.10.12, pytest-7.4.0
collected 35 items

test_pii_masking.py::TestPIIMasker::test_mask_resident_id PASSED   [  3%]
test_pii_masking.py::TestPIIMasker::test_mask_phone_mobile PASSED  [  6%]
...
========================= 35 passed in 2.45s ===========================
```

### 실패 시 디버깅

```bash
# 상세 로그 출력
pytest tests/test_data_collection_preprocessing/ -v --tb=long

# 첫 번째 실패에서 중지
pytest tests/test_data_collection_preprocessing/ -v -x

# 실패한 테스트만 재실행
pytest tests/test_data_collection_preprocessing/ -v --lf
```

---

## 관련 문서

- [데이터 수집/전처리 모듈 README](../../src/data_collection_preprocessing/README.md)
- [PRD 문서](../../docs/prd.md)
- [데이터 수집 계획](../../docs/outputs/M1_Planning/03_Data_Collection/)
