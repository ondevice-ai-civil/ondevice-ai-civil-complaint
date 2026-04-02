"""
RAG 검색 Tool: 기존 HybridSearchEngine / CivilComplaintRetriever를 표준 tool 인터페이스로 래핑.

GovOn shell agent가 ``rag_search`` 이름으로 호출하여
유사 민원·매뉴얼·법령·공시 문서를 검색한다.

Issue: #395
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Dict, List, Optional

from loguru import logger
from pydantic import Field

from src.inference.hybrid_search import HybridSearchEngine, SearchMode
from src.inference.index_manager import IndexType
from src.inference.retriever import CivilComplaintRetriever
from src.inference.tools.base import BaseTool, ToolInput, ToolOutput


# ---------------------------------------------------------------------------
# 입출력 스키마
# ---------------------------------------------------------------------------


class RAGSearchInput(ToolInput):
    """RAG 검색 tool 입력 스키마.

    Attributes
    ----------
    query : str
        검색 쿼리 텍스트.
    doc_type : IndexType
        검색 대상 문서 타입 (case, law, manual, notice).
    top_k : int
        반환할 최대 결과 수 (기본 5, 최대 50).
    search_mode : SearchMode
        검색 모드 (dense, sparse, hybrid).
    """

    query: str = Field(..., min_length=1, max_length=2000, description="검색 쿼리 텍스트")
    doc_type: IndexType = Field(default=IndexType.CASE, description="검색 대상 문서 타입")
    top_k: int = Field(default=5, gt=0, le=50, description="반환할 최대 결과 수")
    search_mode: SearchMode = Field(default=SearchMode.HYBRID, description="검색 모드")


class RAGSearchResultItem(ToolInput):
    """개별 검색 결과 항목."""

    doc_id: str = ""
    source_type: str = ""
    title: str = ""
    content: str = ""
    score: float = 0.0
    reliability_score: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_index: int = 0
    total_chunks: int = 1


# ---------------------------------------------------------------------------
# RAG 검색 Tool
# ---------------------------------------------------------------------------


class RAGSearchTool(BaseTool):
    """기존 RAG 검색 계층을 표준 tool 인터페이스로 제공한다.

    HybridSearchEngine이 있으면 우선 사용하고,
    없으면 CivilComplaintRetriever로 레거시 폴백한다.

    Parameters
    ----------
    hybrid_engine : Optional[HybridSearchEngine]
        Dense+Sparse 하이브리드 검색 엔진.
    retriever : Optional[CivilComplaintRetriever]
        레거시 FAISS 검색 리트리버 (폴백용).
    pii_masker : Optional[object]
        검색 결과 PII 마스킹용 (mask_all 메서드 필요).
    """

    name: ClassVar[str] = "rag_search"
    description: ClassVar[str] = (
        "내부 인덱스에서 유사 민원 사례, 법령, 매뉴얼, 공시 문서를 검색합니다. "
        "키워드 또는 자연어 질의를 입력하면 관련도 순으로 top-k 결과를 반환합니다."
    )

    def __init__(
        self,
        hybrid_engine: Optional[HybridSearchEngine] = None,
        retriever: Optional[CivilComplaintRetriever] = None,
        pii_masker: Optional[Any] = None,
    ) -> None:
        if hybrid_engine is None and retriever is None:
            raise ValueError("hybrid_engine 또는 retriever 중 하나는 필수입니다.")
        self.hybrid_engine = hybrid_engine
        self.retriever = retriever
        self.pii_masker = pii_masker

    def _get_input_schema(self) -> Dict[str, Any]:
        return RAGSearchInput.model_json_schema()

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        """RAG 검색을 실행하고 결과를 ToolOutput으로 반환한다."""
        if not isinstance(tool_input, RAGSearchInput):
            return ToolOutput(success=False, error="잘못된 입력 타입입니다. RAGSearchInput이 필요합니다.")

        inp: RAGSearchInput = tool_input

        # HybridSearchEngine 우선
        if self.hybrid_engine:
            results, actual_mode = await self._search_hybrid(inp)
        elif self.retriever:
            results, actual_mode = await self._search_legacy(inp)
        else:
            return ToolOutput(success=False, error="검색 엔진이 초기화되지 않았습니다.")

        # PII 마스킹
        if self.pii_masker:
            results = self._mask_results(results)

        return ToolOutput(
            success=True,
            data={
                "query": inp.query,
                "doc_type": inp.doc_type.value,
                "search_mode": actual_mode.value,
                "results": results,
                "total": len(results),
            },
        )

    async def _search_hybrid(
        self, inp: RAGSearchInput
    ) -> tuple[List[Dict[str, Any]], SearchMode]:
        """HybridSearchEngine을 사용하여 검색한다."""
        results_raw, actual_mode = await self.hybrid_engine.search(
            query=inp.query,
            index_type=inp.doc_type,
            top_k=inp.top_k,
            mode=inp.search_mode,
        )
        results = [
            {
                "doc_id": r.get("doc_id", ""),
                "source_type": r.get("doc_type", inp.doc_type.value),
                "title": r.get("title", ""),
                "content": _extract_content_by_type(r, inp.doc_type),
                "score": r.get("score", 0.0),
                "reliability_score": r.get("reliability_score", 1.0),
                "metadata": r.get("extras", {}),
                "chunk_index": r.get("chunk_index", 0),
                "total_chunks": r.get("chunk_total", 1),
            }
            for r in results_raw
        ]
        return results, actual_mode

    async def _search_legacy(
        self, inp: RAGSearchInput
    ) -> tuple[List[Dict[str, Any]], SearchMode]:
        """CivilComplaintRetriever 레거시 폴백 검색."""
        loop = asyncio.get_running_loop()
        raw_results = await loop.run_in_executor(
            None, self.retriever.search, inp.query, inp.top_k
        )
        results = [
            {
                "doc_id": raw.get("id", raw.get("doc_id", "")),
                "source_type": inp.doc_type.value,
                "title": raw.get("category", ""),
                "content": (raw.get("complaint", "") + "\n" + raw.get("answer", "")).strip(),
                "score": raw.get("score", 0.0),
                "reliability_score": 1.0,
                "metadata": {},
                "chunk_index": 0,
                "total_chunks": 1,
            }
            for raw in raw_results
        ]
        return results, SearchMode.DENSE

    def _mask_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """검색 결과 내 PII를 마스킹한다."""
        for r in results:
            if r.get("content"):
                r["content"] = self.pii_masker.mask_all(r["content"])
            meta = r.get("metadata", {})
            for key in ("complaint_text", "answer_text", "complaint", "answer"):
                if key in meta and isinstance(meta[key], str):
                    meta[key] = self.pii_masker.mask_all(meta[key])
        return results


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _extract_content_by_type(result: dict, index_type: IndexType) -> str:
    """인덱스 타입별로 적절한 content 텍스트를 추출한다."""
    extras = result.get("extras", {})
    if index_type == IndexType.CASE:
        text = (extras.get("complaint_text", "") + "\n" + extras.get("answer_text", "")).strip()
    elif index_type == IndexType.LAW:
        text = extras.get("law_text", "") or extras.get("content", "")
    elif index_type == IndexType.MANUAL:
        text = extras.get("manual_text", "") or extras.get("content", "")
    elif index_type == IndexType.NOTICE:
        text = extras.get("notice_text", "") or extras.get("content", "")
    else:
        text = ""
    return text or result.get("title", "")
