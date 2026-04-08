"""adapter_tools 팩토리 함수 단위 테스트."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.inference.adapter_registry import AdapterMeta, AdapterRegistry
from src.inference.graph.tools.adapter_tools import build_adapter_tools


@pytest.fixture(autouse=True)
def _reset_registry():
    """각 테스트 전후로 싱글톤 초기화."""
    AdapterRegistry.reset()
    yield
    AdapterRegistry.reset()


def _mock_registry(adapters: dict[str, AdapterMeta]) -> AdapterRegistry:
    """테스트용 레지스트리를 구성한다."""
    registry = AdapterRegistry.__new__(AdapterRegistry)
    registry._adapters = adapters
    AdapterRegistry._instance = registry
    return registry


class TestBuildAdapterTools:
    """build_adapter_tools 팩토리 테스트."""

    def test_creates_tools_per_adapter(self):
        """어댑터 수만큼 도구가 생성되는지 검증."""
        _mock_registry(
            {
                "alpha": AdapterMeta(
                    name="alpha",
                    path="/path/alpha",
                    description="Alpha adapter",
                    domain="alpha",
                    lora_id=1,
                    tool_description="Alpha tool description",
                ),
                "beta": AdapterMeta(
                    name="beta",
                    path="/path/beta",
                    description="Beta adapter",
                    domain="beta",
                    lora_id=2,
                    tool_description="Beta tool description",
                ),
            }
        )

        mock_fn = AsyncMock(return_value={"text": "result"})
        tools = build_adapter_tools(mock_fn)

        assert len(tools) == 2

    def test_tool_names_follow_convention(self):
        """도구명이 {adapter_name}_adapter 형식인지 검증."""
        _mock_registry(
            {
                "public_admin": AdapterMeta(
                    name="public_admin",
                    path="/path/pa",
                    description="PA",
                    domain="pa",
                    lora_id=1,
                    tool_description="PA tool desc",
                ),
                "legal": AdapterMeta(
                    name="legal",
                    path="/path/legal",
                    description="Legal",
                    domain="legal",
                    lora_id=2,
                    tool_description="Legal tool desc",
                ),
            }
        )

        mock_fn = AsyncMock(return_value={"text": "result"})
        tools = build_adapter_tools(mock_fn)
        names = sorted(t.name for t in tools)

        assert names == ["legal_adapter", "public_admin_adapter"]

    def test_tool_description_from_yaml(self):
        """도구 description이 YAML의 tool_description에서 오는지 검증."""
        _mock_registry(
            {
                "test_adapter": AdapterMeta(
                    name="test_adapter",
                    path="/path/test",
                    description="기본 설명",
                    domain="test",
                    lora_id=1,
                    tool_description="YAML에서 정의한 상세 도구 설명",
                ),
            }
        )

        mock_fn = AsyncMock(return_value={"text": "result"})
        tools = build_adapter_tools(mock_fn)

        assert len(tools) == 1
        assert tools[0].description == "YAML에서 정의한 상세 도구 설명"

    def test_tool_description_fallback(self):
        """tool_description이 없으면 description + keywords로 조합되는지 검증."""
        _mock_registry(
            {
                "fallback": AdapterMeta(
                    name="fallback",
                    path="/path/fb",
                    description="기본 설명",
                    domain="fb",
                    lora_id=1,
                    keywords=("키워드1", "키워드2"),
                    tool_description="",
                ),
            }
        )

        mock_fn = AsyncMock(return_value={"text": "result"})
        tools = build_adapter_tools(mock_fn)

        assert "기본 설명" in tools[0].description
        assert "키워드1" in tools[0].description

    @pytest.mark.asyncio
    async def test_tool_passes_correct_adapter_context(self):
        """도구 실행 시 올바른 adapter context가 전달되는지 검증."""
        _mock_registry(
            {
                "public_admin": AdapterMeta(
                    name="public_admin",
                    path="/path/pa",
                    description="PA",
                    domain="pa",
                    lora_id=1,
                    tool_description="PA tool desc",
                ),
                "legal": AdapterMeta(
                    name="legal",
                    path="/path/legal",
                    description="Legal",
                    domain="legal",
                    lora_id=2,
                    tool_description="Legal tool desc",
                ),
            }
        )

        mock_fn = AsyncMock(return_value={"text": "응답 결과", "citations": []})
        tools = build_adapter_tools(mock_fn)

        # 이름으로 도구 찾기
        tool_map = {t.name: t for t in tools}

        # public_admin_adapter 실행
        result = await tool_map["public_admin_adapter"].ainvoke({"query": "민원 질의"})
        mock_fn.assert_called_with(
            query="민원 질의",
            context={"adapter": "public_admin"},
            session=None,
        )

        # legal_adapter 실행
        result = await tool_map["legal_adapter"].ainvoke({"query": "법률 질의"})
        mock_fn.assert_called_with(
            query="법률 질의",
            context={"adapter": "legal"},
            session=None,
        )

        # 반환값이 JSON 문자열인지 검증
        parsed = json.loads(result)
        assert parsed["text"] == "응답 결과"

    def test_empty_registry_returns_empty_list(self):
        """어댑터가 없으면 빈 리스트를 반환하는지 검증."""
        _mock_registry({})

        mock_fn = AsyncMock(return_value={"text": "result"})
        tools = build_adapter_tools(mock_fn)

        assert tools == []
