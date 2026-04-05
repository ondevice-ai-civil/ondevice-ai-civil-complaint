"""AI Hub 데이터셋 파서 모듈.

각 파서는 단일 JSON 파일을 파싱하여 학습 레코드 목록을 반환한다.
반환 형식:
    {
        "question": str,
        "answer": str,
        "source": str,
        "category": str,
        "metadata": dict,
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _load_json(filepath: Path) -> Any:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


class GukripParser:
    """71852 국립아시아문화전당 파서.

    consulting_content의 '상담원:' 발화를 추출하여 답변으로 사용하고,
    instructions[0].data[0].instruction을 질문으로 사용한다.
    """

    def parse(self, filepath: Path) -> list[dict]:
        data = _load_json(filepath)
        if isinstance(data, list):
            records = []
            for item in data:
                records.extend(self._parse_item(item))
            return records
        return self._parse_item(data)

    def _parse_item(self, item: dict) -> list[dict]:
        content: str = item.get("consulting_content", "")
        source_id: str = item.get("source_id", "")
        consulting_date: str = item.get("consulting_date", "")
        category: str = item.get("consulting_category", "")

        # 상담원 발화 추출
        agent_turns = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("상담원:"):
                turn_text = line[len("상담원:") :].strip()
                if turn_text:
                    agent_turns.append(turn_text)

        if not agent_turns:
            return []

        answer = " ".join(agent_turns)

        # instruction에서 질문 추출
        instructions = item.get("instructions", [])
        if not instructions:
            return []

        data_list = instructions[0].get("data", [])
        if not data_list:
            return []

        question = data_list[0].get("instruction", "").strip()
        if not question:
            return []

        return [
            {
                "question": question,
                "answer": answer,
                "source": "71852_국립아시아문화전당",
                "category": category,
                "metadata": {
                    "source_id": source_id,
                    "consulting_date": consulting_date,
                },
            }
        ]


class GovQAParser:
    """71852 중앙/지방행정기관 파서.

    consulting_content에서 Q/A 형식을 파싱하여 공식 정부 답변을 추출한다.
    보조 질문(instructions.data[*].instruction)은 별도 레코드로 생성한다.
    """

    # A 구분자 패턴: "\nA :" 또는 "\nA:"
    _A_SEP = re.compile(r"\nA\s*:")

    def parse(self, filepath: Path) -> list[dict]:
        data = _load_json(filepath)
        if isinstance(data, list):
            records = []
            for item in data:
                records.extend(self._parse_item(item))
            return records
        return self._parse_item(data)

    def _parse_item(self, item: dict) -> list[dict]:
        content: str = item.get("consulting_content", "")
        source_str: str = item.get("source", "")
        source_id: str = item.get("source_id", "")
        consulting_date: str = item.get("consulting_date", "")
        category: str = item.get("consulting_category", "")

        # A 부분 분리
        parts = self._A_SEP.split(content, maxsplit=1)
        if len(parts) < 2:
            return []

        q_part, a_part = parts[0], parts[1].strip()
        if not a_part:
            return []

        # Q 부분에서 질문 추출
        question = self._extract_question(q_part)
        if not question:
            return []

        records = [
            {
                "question": question,
                "answer": a_part,
                "source": "71852_중앙행정기관",
                "category": category,
                "metadata": {
                    "source_id": source_id,
                    "consulting_date": consulting_date,
                    "org": source_str,
                },
            }
        ]

        # 보조 질문(instructions.data[*].instruction)으로 추가 레코드 생성
        instructions = item.get("instructions", [])
        if instructions:
            for instr_item in instructions[0].get("data", []):
                sub_q = instr_item.get("instruction", "").strip()
                if sub_q and sub_q != question:
                    records.append(
                        {
                            "question": sub_q,
                            "answer": a_part,
                            "source": "71852_중앙행정기관",
                            "category": category,
                            "metadata": {
                                "source_id": source_id,
                                "consulting_date": consulting_date,
                                "org": source_str,
                                "question_type": "auxiliary",
                            },
                        }
                    )

        return records

    @staticmethod
    def _extract_question(q_part: str) -> str:
        """Q 블록에서 질문 텍스트를 추출한다."""
        # "Q :" 또는 "Q:" 이후 텍스트 추출
        q_match = re.search(r"\nQ\s*:(.*?)(?=\n\n|\Z)", q_part, re.DOTALL)
        if q_match:
            return q_match.group(1).strip()

        # fallback: "제목 :" 이후 텍스트
        title_match = re.search(r"제목\s*:\s*(.+)", q_part)
        if title_match:
            return title_match.group(1).strip()

        return q_part.strip()


class GovQALocalParser(GovQAParser):
    """71852 지방행정기관 파서 — GovQAParser와 동일한 로직, source 레이블만 다름."""

    def _parse_item(self, item: dict) -> list[dict]:
        records = super()._parse_item(item)
        for r in records:
            r["source"] = "71852_지방행정기관"
        return records


class AdminLawParser:
    """71847 행정법 파서.

    label.input을 질문, label.output을 답변으로 사용한다.
    결정례(TL_결정례_QA)와 법령(TL_법령_QA) 모두 동일 구조.
    """

    def __init__(self, source_label: str = "71847_결정례"):
        self.source_label = source_label

    def parse(self, filepath: Path) -> list[dict]:
        data = _load_json(filepath)
        if isinstance(data, list):
            records = []
            for item in data:
                records.extend(self._parse_item(item))
            return records
        return self._parse_item(data)

    def _parse_item(self, item: dict) -> list[dict]:
        label = item.get("label", {})
        question = label.get("input", "").strip()
        answer = label.get("output", "").strip()

        if not question or not answer:
            return []

        info = item.get("info", {})
        case_name = info.get("caseName", info.get("title", ""))
        category = info.get("ministry", info.get("caseCode", ""))

        return [
            {
                "question": question,
                "answer": answer,
                "source": self.source_label,
                "category": category,
                "metadata": {
                    "case_name": case_name,
                    "law_class": info.get("lawClass", ""),
                },
            }
        ]
