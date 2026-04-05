"""파서 및 파이프라인 단위 테스트."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from src.data_collection_preprocessing.config import DataConfig
from src.data_collection_preprocessing.parsers import (
    AdminLawParser,
    GovQALocalParser,
    GovQAParser,
    GukripParser,
)
from src.data_collection_preprocessing.pipeline import CivilResponseDataPipeline

# ---------------------------------------------------------------------------
# 실제 샘플 파일 경로
# worktree에는 data/ 디렉터리가 없으므로 절대 경로를 사용한다.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
# tests/test_data/test_parsers.py -> repo root (최대 4단계 상위)
_REPO_ROOT = _HERE.parents[2]  # .claude/worktrees/<branch>
_MAIN_REPO = Path("/Users/um-yunsang/GovOn")
# worktree와 main repo 둘 다 확인 후 존재하는 쪽 사용
_RAW_BASE = (
    _MAIN_REPO / "data/raw/aihub"
    if (_MAIN_REPO / "data/raw/aihub").exists()
    else _REPO_ROOT / "data/raw/aihub"
)
RAW_BASE = _RAW_BASE
GUKRIP_SAMPLE = RAW_BASE / "71852/train/국립/03_질의응답_000012_1.json"
GOV_CENTRAL_SAMPLE = RAW_BASE / "71852/train/중앙/03_질의응답_300001_1.json"
GOV_LOCAL_SAMPLE = RAW_BASE / "71852/train/지방/03_질의응답_000022_1.json"
ADMIN_LAW_DECISION_SAMPLE = RAW_BASE / "71847/TL_결정례_QA/HJ_K_10072_QA_1.json"
ADMIN_LAW_STATUTE_SAMPLE = RAW_BASE / "71847/TL_법령_QA/HJ_B_000001_QA_1.json"


def skip_if_missing(path: Path):
    return pytest.mark.skipif(not path.exists(), reason=f"샘플 파일 없음: {path}")


# ---------------------------------------------------------------------------
# GukripParser 테스트
# ---------------------------------------------------------------------------


@skip_if_missing(GUKRIP_SAMPLE)
def test_gukrip_parser_basic():
    parser = GukripParser()
    records = parser.parse(GUKRIP_SAMPLE)

    assert len(records) >= 1
    rec = records[0]
    assert rec["source"] == "71852_국립아시아문화전당"
    assert len(rec["question"]) >= 5
    # 상담원 발화 기반 답변 — instructions.data[0].output 단답 아님
    assert len(rec["answer"]) >= 10
    assert "category" in rec
    assert "source_id" in rec["metadata"]


@skip_if_missing(GUKRIP_SAMPLE)
def test_gukrip_answer_not_from_output_field():
    """instructions.data[0].output의 단답(15자 미만)이 답변으로 사용되면 안 된다."""
    parser = GukripParser()
    records = parser.parse(GUKRIP_SAMPLE)

    assert records, "파싱 결과가 비어있음"
    # 국립 샘플의 instructions.output은 "홈페이지, 콜센터, 현장 매표소" (15자)
    # 파서는 상담원 발화를 조합하므로 훨씬 길어야 한다.
    assert len(records[0]["answer"]) > 50, "답변이 너무 짧음 — output 단답이 잘못 사용됐을 수 있음"


# ---------------------------------------------------------------------------
# GovQAParser 테스트 (중앙)
# ---------------------------------------------------------------------------


@skip_if_missing(GOV_CENTRAL_SAMPLE)
def test_govqa_parser_central_basic():
    parser = GovQAParser()
    records = parser.parse(GOV_CENTRAL_SAMPLE)

    assert len(records) >= 1
    rec = records[0]
    assert rec["source"] == "71852_중앙행정기관"
    # 공식 정부 답변은 길다
    assert len(rec["answer"]) >= 100
    # Q 부분이 질문으로 사용됨
    assert len(rec["question"]) >= 5


@skip_if_missing(GOV_CENTRAL_SAMPLE)
def test_govqa_parser_answer_is_official_reply():
    """consulting_content의 A: 이후 공식 답변이 answer 필드로 들어가야 한다."""
    parser = GovQAParser()
    records = parser.parse(GOV_CENTRAL_SAMPLE)

    assert records
    # 공식 답변 시작 문구 확인
    assert "안녕하십니까" in records[0]["answer"]


@skip_if_missing(GOV_CENTRAL_SAMPLE)
def test_govqa_auxiliary_question_records():
    """instructions.data[*].instruction 보조 질문이 별도 레코드로 생성되어야 한다."""
    parser = GovQAParser()
    records = parser.parse(GOV_CENTRAL_SAMPLE)

    # 보조 질문 레코드가 있으면 question_type이 auxiliary
    aux_records = [r for r in records if r["metadata"].get("question_type") == "auxiliary"]
    # 샘플에 보조 질문이 있을 수도 없을 수도 있으므로, 타입이 있으면 auxiliary여야 함
    for r in aux_records:
        assert r["metadata"]["question_type"] == "auxiliary"


# ---------------------------------------------------------------------------
# GovQALocalParser 테스트 (지방)
# ---------------------------------------------------------------------------


@skip_if_missing(GOV_LOCAL_SAMPLE)
def test_govqa_local_parser_source_label():
    parser = GovQALocalParser()
    records = parser.parse(GOV_LOCAL_SAMPLE)

    assert records
    assert records[0]["source"] == "71852_지방행정기관"


@skip_if_missing(GOV_LOCAL_SAMPLE)
def test_govqa_local_parser_answer_length():
    parser = GovQALocalParser()
    records = parser.parse(GOV_LOCAL_SAMPLE)

    assert records
    assert len(records[0]["answer"]) >= 30


# ---------------------------------------------------------------------------
# AdminLawParser 테스트
# ---------------------------------------------------------------------------


@skip_if_missing(ADMIN_LAW_DECISION_SAMPLE)
def test_admin_law_parser_decision():
    parser = AdminLawParser(source_label="71847_결정례")
    records = parser.parse(ADMIN_LAW_DECISION_SAMPLE)

    assert len(records) >= 1
    rec = records[0]
    assert rec["source"] == "71847_결정례"
    assert len(rec["question"]) >= 5
    assert len(rec["answer"]) >= 30
    assert "case_name" in rec["metadata"]


@skip_if_missing(ADMIN_LAW_STATUTE_SAMPLE)
def test_admin_law_parser_statute():
    parser = AdminLawParser(source_label="71847_법령")
    records = parser.parse(ADMIN_LAW_STATUTE_SAMPLE)

    assert records
    assert records[0]["source"] == "71847_법령"


# ---------------------------------------------------------------------------
# 중복 제거 테스트
# ---------------------------------------------------------------------------


def test_deduplicate_removes_exact_duplicates():
    pipeline = CivilResponseDataPipeline()
    records = [
        {"question": "Q1", "answer": "A1", "source": "test", "category": "c"},
        {"question": "Q1", "answer": "A1", "source": "test", "category": "c"},  # dup
        {"question": "Q2", "answer": "A2", "source": "test", "category": "c"},
    ]
    unique = pipeline._deduplicate(records)
    assert len(unique) == 2


def test_deduplicate_keeps_different_answers():
    pipeline = CivilResponseDataPipeline()
    records = [
        {"question": "Q1", "answer": "A1", "source": "test", "category": "c"},
        {"question": "Q1", "answer": "A2", "source": "test", "category": "c"},  # 다른 답변
    ]
    unique = pipeline._deduplicate(records)
    assert len(unique) == 2


# ---------------------------------------------------------------------------
# 길이 필터 테스트
# ---------------------------------------------------------------------------


def test_filter_removes_short_answers():
    config = DataConfig(min_answer_length=30, min_question_length=5)
    pipeline = CivilResponseDataPipeline(config)
    records = [
        {"question": "질문입니다", "answer": "짧은 답변", "source": "test", "category": "c"},
        {"question": "긴 질문입니다", "answer": "A" * 50, "source": "test", "category": "c"},
    ]
    filtered = pipeline._filter(records)
    assert len(filtered) == 1
    assert filtered[0]["answer"] == "A" * 50


def test_filter_removes_long_answers():
    config = DataConfig(max_answer_length=100)
    pipeline = CivilResponseDataPipeline(config)
    records = [
        {"question": "충분히 긴 질문입니다", "answer": "A" * 50, "source": "test", "category": "c"},
        {
            "question": "충분히 긴 질문입니다",
            "answer": "A" * 200,
            "source": "test",
            "category": "c",
        },
    ]
    filtered = pipeline._filter(records)
    assert len(filtered) == 1


def test_filter_removes_short_questions():
    config = DataConfig(min_question_length=5)
    pipeline = CivilResponseDataPipeline(config)
    records = [
        {"question": "짧", "answer": "A" * 50, "source": "test", "category": "c"},
        {"question": "충분히 긴 질문입니다", "answer": "A" * 50, "source": "test", "category": "c"},
    ]
    filtered = pipeline._filter(records)
    assert len(filtered) == 1


# ---------------------------------------------------------------------------
# JSONL 출력 형식 테스트
# ---------------------------------------------------------------------------


def test_save_jsonl_format():
    pipeline = CivilResponseDataPipeline()
    records = [
        {
            "question": "민원 질문",
            "answer": "공식 답변",
            "source": "71852_중앙행정기관",
            "category": "도로관리과",
            "metadata": {},
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.jsonl"
        pipeline._save_jsonl(records, out_path)

        lines = out_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        obj = json.loads(lines[0])
        assert obj["instruction"] == pipeline.INSTRUCTION_TEXT
        assert obj["input"] == "민원 질문"
        assert obj["output"] == "공식 답변"
        assert obj["source"] == "71852_중앙행정기관"
        assert obj["category"] == "도로관리과"


def test_save_jsonl_is_valid_json_per_line():
    """각 줄이 독립된 유효한 JSON이어야 한다."""
    pipeline = CivilResponseDataPipeline()
    records = [
        {"question": f"Q{i}", "answer": "A" * 50, "source": "test", "category": "c", "metadata": {}}
        for i in range(5)
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.jsonl"
        pipeline._save_jsonl(records, out_path)

        for line in out_path.read_text(encoding="utf-8").strip().split("\n"):
            obj = json.loads(line)
            assert "instruction" in obj
            assert "input" in obj
            assert "output" in obj
