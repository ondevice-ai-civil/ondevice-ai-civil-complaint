"""LangGraph capability 공통 추상화."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    """RAG/API 출처 무관하게 동일한 구조로 evidence를 표현.

    source_type: "rag" | "api" | "llm_generated"
    """

    source_type: str  # "rag" | "api" | "llm_generated"
    title: str
    excerpt: str  # 본문 발췌 (최대 500자)
    link_or_path: str = ""  # URL(API) 또는 file_path(RAG)
    page: Optional[int] = None
    score: float = 0.0
    provider_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "title": self.title,
            "excerpt": self.excerpt,
            "link_or_path": self.link_or_path,
            "page": self.page,
            "score": self.score,
            "provider_meta": self.provider_meta,
        }


@dataclass
class EvidenceEnvelope:
    """mixed evidence 결과 컨테이너."""

    items: List[EvidenceItem] = field(default_factory=list)
    summary_text: str = ""
    status: str = "ok"  # "ok" | "empty" | "partial" | "error"
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "summary_text": self.summary_text,
            "status": self.status,
            "errors": self.errors,
        }


@dataclass
class CapabilityMetadata:
    """planner·executor·session trace에서 공통으로 사용하는 capability 메타데이터."""

    name: str  # tool registry key (예: "api_lookup")
    description: str  # LLM planner가 읽는 한국어 설명 (1-2문장)
    approval_summary: str  # approval_wait 프롬프트에 표시되는 요약
    provider: str  # 데이터 제공자 식별자 (예: "data.go.kr")
    timeout_sec: float = 10.0  # 기본 타임아웃


@dataclass
class LookupResult:
    """api_lookup 공통 응답 스키마."""

    success: bool
    query: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    context_text: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    error: Optional[str] = None
    empty_reason: Optional[str] = None  # "quota", "no_match", "provider_error"
    latency_ms: float = 0.0
    evidence: Optional[EvidenceEnvelope] = None  # 정규화된 evidence (새 필드)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "success": self.success,
            "query": self.query,
            "count": len(self.results),
            "results": self.results,
            "context_text": self.context_text,
            "citations": self.citations,
            "provider": self.provider,
            "error": self.error,
            "empty_reason": self.empty_reason,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.evidence is not None:
            d["evidence"] = self.evidence.to_dict()
        return d


class CapabilityBase(ABC):
    """LangGraph tool capability 추상 베이스.

    RegistryExecutorAdapter의 tool_registry에 등록 가능한 async callable 인터페이스.
    """

    @property
    @abstractmethod
    def metadata(self) -> CapabilityMetadata: ...

    @abstractmethod
    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        session: Any,  # SessionContext (순환 import 방지)
    ) -> LookupResult: ...

    async def __call__(
        self,
        query: str,
        context: Dict[str, Any],
        session: Any,
    ) -> Dict[str, Any]:
        """RegistryExecutorAdapter 호환 진입점."""
        import time

        start = time.monotonic()
        result = await self.execute(query, context, session)
        result.latency_ms = (time.monotonic() - start) * 1000
        return result.to_dict()
