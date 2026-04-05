"""법률해석·근거인용 LoRA 학습 데이터셋 생성 파이프라인.

4개 소스를 통합하여 instruction-tuning 형식으로 변환한다.
Output JSONL:
  {"instruction": "...", "input": "질문", "output": "법적 근거를 포함한 답변",
   "source": "...", "category": "..."}
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(
    os.environ.get("GOVON_ROOT", Path(__file__).resolve().parents[2])
)
RAW_DIR = Path(os.environ.get("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw" / "aihub"))
OUTPUT_DIR = PROJECT_ROOT / "training" / "legal_adapter" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HF_REPO = "umyunsang/govon-legal-response-data"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MIN_OUTPUT_LEN = 30
VAL_RATIO = 0.1
SEED = 42

INSTRUCTION_QA = "다음 법률 질문에 관련 법령 조항을 인용하여 답변하세요."
INSTRUCTION_PRECEDENT = "다음 법적 쟁점에 대해 관련 법령을 인용하여 해석하세요."

# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def _dedup_key(inp: str, out: str) -> str:
    """질문+답변 해시로 중복 판별."""
    return hashlib.md5((inp.strip() + out.strip()).encode()).hexdigest()


def _is_valid(record: dict) -> bool:
    """최소 품질 필터."""
    return (
        bool(record.get("input", "").strip())
        and bool(record.get("output", "").strip())
        and len(record["output"].strip()) >= MIN_OUTPUT_LEN
    )


# ---------------------------------------------------------------------------
# 소스 1: HuggingFace 판례 데이터
# ---------------------------------------------------------------------------

def load_hf_precedents() -> list[dict]:
    """HF 판례 데이터에서 참조조문 + 판시사항이 있는 레코드를 변환."""
    print("[소스1] HuggingFace 판례 데이터 로딩...")
    ds = load_dataset(
        "joonhok-exo-ai/korean_law_open_data_precedents",
        split="train",
    )
    records: list[dict] = []
    total = 0
    for row in tqdm(ds, desc="HF 판례"):
        ref_law = (row.get("참조조문") or "").strip()
        issue = (row.get("판시사항") or "").strip()
        summary = (row.get("판결요지") or "").strip()
        if not ref_law or not issue:
            continue
        total += 1
        # 답변: 판결요지가 있으면 사용, 없으면 판시사항 자체를 답변으로
        answer = summary if summary else issue
        # 답변 앞에 참조조문 인용 추가
        if ref_law not in answer:
            answer = f"{ref_law}에 따르면, {answer}"
        rec = {
            "instruction": INSTRUCTION_PRECEDENT,
            "input": issue,
            "output": answer,
            "source": "hf_precedent",
            "category": "판례",
        }
        if _is_valid(rec):
            records.append(rec)
    print(f"  HF 판례: {len(ds):,}건 -> 참조조문+판시사항 존재 {total:,}건 -> 필터 후 {len(records):,}건")
    return records


# ---------------------------------------------------------------------------
# 소스 2 & 3: AI Hub 71841 민사법 / 71843 지식재산권 (taskinfo 구조)
# ---------------------------------------------------------------------------

def _iter_json_files(base_dir: Path) -> Iterator[Path]:
    """디렉터리 아래 모든 .json 파일을 순회."""
    for dirpath, _, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.endswith(".json"):
                yield Path(dirpath) / fn


def _parse_taskinfo_source(
    base_dir: Path,
    source_label: str,
    category: str,
) -> list[dict]:
    """taskinfo.input / taskinfo.output 구조의 AI Hub 데이터 파싱."""
    records: list[dict] = []
    total = 0
    tl_dirs = sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith("TL_")
    )
    for tl_dir in tl_dirs:
        subdir_label = tl_dir.name
        for fpath in tqdm(
            list(_iter_json_files(tl_dir)),
            desc=f"  {subdir_label}",
            leave=False,
        ):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            total += 1
            ti = data.get("taskinfo", {})
            inp = (ti.get("input") or "").strip()
            out = (ti.get("output") or "").strip()
            if not inp or not out:
                continue
            rec = {
                "instruction": INSTRUCTION_QA,
                "input": inp,
                "output": out,
                "source": source_label,
                "category": category,
            }
            if _is_valid(rec):
                records.append(rec)
    print(f"  {source_label}: 파일 {total:,}건 -> 필터 후 {len(records):,}건")
    return records


def load_71841_civil() -> list[dict]:
    """AI Hub 71841 민사법 QA."""
    print("[소스2] AI Hub 71841 민사법 QA 로딩...")
    return _parse_taskinfo_source(
        RAW_DIR / "71841",
        source_label="71841_민사법",
        category="민사",
    )


def load_71843_ip() -> list[dict]:
    """AI Hub 71843 지식재산권법 QA."""
    print("[소스3] AI Hub 71843 지식재산권법 QA 로딩...")
    return _parse_taskinfo_source(
        RAW_DIR / "71843",
        source_label="71843_지식재산권",
        category="지식재산권",
    )


# ---------------------------------------------------------------------------
# 소스 4: AI Hub 71848 형사법 (label 구조)
# ---------------------------------------------------------------------------

def load_71848_criminal() -> list[dict]:
    """AI Hub 71848 형사법 QA — label.input / label.output 구조."""
    print("[소스4] AI Hub 71848 형사법 QA 로딩...")
    base_dir = RAW_DIR / "71848"
    records: list[dict] = []
    total = 0
    tl_dirs = sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith("TL_")
    )
    for tl_dir in tl_dirs:
        subdir_label = tl_dir.name
        # source에 하위 카테고리 포함
        src = f"71848_형사법_{subdir_label.replace('TL_', '').replace('_QA', '')}"
        for fpath in tqdm(
            list(_iter_json_files(tl_dir)),
            desc=f"  {subdir_label}",
            leave=False,
        ):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            total += 1
            label = data.get("label", {})
            inp = (label.get("input") or "").strip()
            out = (label.get("output") or "").strip()
            if not inp or not out:
                continue
            rec = {
                "instruction": INSTRUCTION_QA,
                "input": inp,
                "output": out,
                "source": src,
                "category": "형사",
            }
            if _is_valid(rec):
                records.append(rec)
    print(f"  71848 형사법: 파일 {total:,}건 -> 필터 후 {len(records):,}건")
    return records


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("=== Legal Response Dataset Pipeline ===")
    print("=" * 60)

    # 1. 수집
    all_records: list[dict] = []
    src_stats: dict[str, int] = {}

    for loader, name in [
        (load_hf_precedents, "HF 판례"),
        (load_71841_civil, "71841 민사법"),
        (load_71843_ip, "71843 지식재산권"),
        (load_71848_criminal, "71848 형사법"),
    ]:
        recs = loader()
        src_stats[name] = len(recs)
        all_records.extend(recs)

    print(f"\n통합 전 총: {len(all_records):,}건")

    # 2. 중복 제거
    seen: set[str] = set()
    deduped: list[dict] = []
    for rec in all_records:
        key = _dedup_key(rec["input"], rec["output"])
        if key not in seen:
            seen.add(key)
            deduped.append(rec)
    dup_removed = len(all_records) - len(deduped)
    print(f"중복 제거: {dup_removed:,}건 제거 -> {len(deduped):,}건")

    # 3. Shuffle & split
    random.seed(SEED)
    random.shuffle(deduped)
    val_size = int(len(deduped) * VAL_RATIO)
    val_data = deduped[:val_size]
    train_data = deduped[val_size:]
    print(f"최종: train {len(train_data):,}건 / val {len(val_data):,}건")

    # 4. JSONL 저장
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"
    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"저장: {path} ({len(data):,}건)")

    # 5. README 생성
    readme_path = OUTPUT_DIR / "README.md"
    readme_path.write_text(
        _build_readme(src_stats, dup_removed, len(train_data), len(val_data)),
        encoding="utf-8",
    )

    # 6. HF Hub 업로드
    print("\nHuggingFace Hub 업로드 시작...")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)
    for fpath in [train_path, val_path, readme_path]:
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fpath.name,
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        print(f"  업로드: {fpath.name}")

    print(f"\nHF Hub 업로드 완료: https://huggingface.co/datasets/{HF_REPO}")

    # 최종 요약
    print("\n" + "=" * 60)
    print("=== Legal Response Dataset Pipeline 완료 ===")
    print("=" * 60)
    for name, cnt in src_stats.items():
        print(f"  {name}: {cnt:,}건")
    print(f"  중복 제거: {dup_removed:,}건")
    print(f"  최종: train {len(train_data):,}건 / val {len(val_data):,}건")
    print(f"  HF Hub: {HF_REPO}")
    print("=" * 60)


def _build_readme(
    src_stats: dict[str, int],
    dup_removed: int,
    train_count: int,
    val_count: int,
) -> str:
    total = train_count + val_count
    lines = [
        "---",
        "license: cc-by-4.0",
        "language:",
        "- ko",
        "task_categories:",
        "- question-answering",
        "- text-generation",
        "tags:",
        "- legal",
        "- korean",
        "- instruction-tuning",
        "- lora",
        "size_categories:",
        f"- {'100K<n<1M' if total > 100_000 else '10K<n<100K'}",
        "---",
        "",
        "# GovOn Legal Response Dataset",
        "",
        "법률해석 및 근거인용 LoRA 어댑터 학습용 instruction-tuning 데이터셋.",
        "",
        "## 데이터 소스",
        "",
    ]
    for name, cnt in src_stats.items():
        lines.append(f"- **{name}**: {cnt:,}건")
    lines += [
        "",
        "## 통계",
        "",
        f"- 중복 제거: {dup_removed:,}건",
        f"- Train: {train_count:,}건",
        f"- Validation: {val_count:,}건",
        f"- 총합: {total:,}건",
        "",
        "## 형식",
        "",
        "각 레코드는 JSONL 형식이며 다음 필드를 포함합니다:",
        "",
        "```json",
        '{',
        '  "instruction": "다음 법률 질문에 관련 법령 조항을 인용하여 답변하세요.",',
        '  "input": "질문 텍스트",',
        '  "output": "법적 근거를 포함한 답변",',
        '  "source": "데이터 소스 식별자",',
        '  "category": "법률 카테고리"',
        '}',
        "```",
        "",
        "## 라이선스",
        "",
        "CC-BY-4.0",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
