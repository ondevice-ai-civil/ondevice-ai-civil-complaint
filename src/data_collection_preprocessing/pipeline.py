"""민원답변 학습 데이터 파이프라인."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from .config import DataConfig
from .parsers import AdminLawParser, GovQALocalParser, GovQAParser, GukripParser

logger = logging.getLogger(__name__)


class CivilResponseDataPipeline:
    """AI Hub 원시 데이터를 instruction-tuning JSONL로 변환하는 파이프라인."""

    INSTRUCTION_TEXT = "다음 민원에 대한 답변을 작성해 주세요."

    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()

    def run(self) -> dict[str, int]:
        """전체 파이프라인 실행. 결과 통계 반환."""
        records: list[dict] = []

        logger.info("71852 데이터 처리 시작")
        records_71852 = self._process_71852()
        logger.info("71852 데이터 %d개 수집", len(records_71852))
        records.extend(records_71852)

        logger.info("71847 데이터 처리 시작")
        records_71847 = self._process_71847()
        logger.info("71847 데이터 %d개 수집", len(records_71847))
        records.extend(records_71847)

        logger.info("중복 제거 전 총 %d개", len(records))
        records = self._deduplicate(records)
        logger.info("중복 제거 후 %d개", len(records))

        records = self._filter(records)
        logger.info("필터링 후 %d개", len(records))

        train, val = self._split(records)
        logger.info("train=%d, val=%d", len(train), len(val))

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_jsonl(train, output_dir / "train.jsonl")
        self._save_jsonl(val, output_dir / "val.jsonl")

        return {
            "total": len(records),
            "train": len(train),
            "val": len(val),
        }

    # ------------------------------------------------------------------
    # 데이터셋별 처리
    # ------------------------------------------------------------------

    def _process_71852(self) -> list[dict]:
        base = Path(self.config.raw_dir) / "71852"
        records: list[dict] = []

        # 국립아시아문화전당
        gukrp = GukripParser()
        for split in ("train", "val"):
            dir_path = base / split / "국립"
            if dir_path.exists():
                records.extend(self._parse_dir(gukrp, dir_path))

        # 중앙행정기관
        gov_central = GovQAParser()
        for split in ("train", "val"):
            dir_path = base / split / "중앙"
            if dir_path.exists():
                records.extend(self._parse_dir(gov_central, dir_path))

        # 지방행정기관
        gov_local = GovQALocalParser()
        for split in ("train", "val"):
            dir_path = base / split / "지방"
            if dir_path.exists():
                records.extend(self._parse_dir(gov_local, dir_path))

        return records

    def _process_71847(self) -> list[dict]:
        base = Path(self.config.raw_dir) / "71847"
        records: list[dict] = []

        # 결정례 QA
        decision_parser = AdminLawParser(source_label="71847_결정례")
        dir_path = base / "TL_결정례_QA"
        if dir_path.exists():
            records.extend(self._parse_dir(decision_parser, dir_path))

        # 법령 QA
        law_parser = AdminLawParser(source_label="71847_법령")
        dir_path = base / "TL_법령_QA"
        if dir_path.exists():
            records.extend(self._parse_dir(law_parser, dir_path))

        return records

    # ------------------------------------------------------------------
    # 유틸리티
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dir(parser: Any, dir_path: Path) -> list[dict]:
        records: list[dict] = []
        json_files = list(dir_path.glob("*.json"))
        logger.debug("  %s: %d 파일", dir_path, len(json_files))
        for filepath in json_files:
            try:
                records.extend(parser.parse(filepath))
            except Exception as exc:  # noqa: BLE001
                logger.warning("파싱 실패 %s: %s", filepath, exc)
        return records

    def _deduplicate(self, records: list[dict]) -> list[dict]:
        """질문+답변 해시 기반 중복 제거."""
        seen: set[str] = set()
        unique: list[dict] = []
        for rec in records:
            key = hashlib.md5(  # nosec B324
                (rec["question"] + rec["answer"]).encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        return unique

    def _filter(self, records: list[dict]) -> list[dict]:
        """길이 필터링."""
        filtered: list[dict] = []
        for rec in records:
            answer_len = len(rec["answer"])
            question_len = len(rec["question"])
            if answer_len < self.config.min_answer_length:
                continue
            if answer_len > self.config.max_answer_length:
                continue
            if question_len < self.config.min_question_length:
                continue
            filtered.append(rec)
        return filtered

    def _split(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        """train/val 분리 (셔플 후 비율 분할)."""
        shuffled = list(records)
        random.seed(42)
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * self.config.train_ratio)
        return shuffled[:split_idx], shuffled[split_idx:]

    def _save_jsonl(self, records: list[dict], filepath: Path) -> None:
        """Instruction-tuning 표준 JSONL 형식으로 저장."""
        filepath = Path(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            for rec in records:
                line = {
                    "instruction": self.INSTRUCTION_TEXT,
                    "input": rec["question"],
                    "output": rec["answer"],
                    "source": rec["source"],
                    "category": rec.get("category", ""),
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        logger.info("저장 완료: %s (%d 레코드)", filepath, len(records))
