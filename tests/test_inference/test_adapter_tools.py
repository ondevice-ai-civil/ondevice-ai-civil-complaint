"""adapter_tools 팩토리 함수 단위 테스트.

graph 패키지의 import chain(faiss, vllm 등)을 우회하기 위해
importlib를 사용하여 adapter_tools 모듈만 직접 로드한다.
sys.modules를 오염시키지 않도록 테스트 종료 시 복원한다.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pathlib
import sys
import types
from unittest.mock import AsyncMock

import pytest

from src.inference.adapter_registry import AdapterMeta, AdapterRegistry


def _import_adapter_tools():
    """graph 패키지 초기화를 우회하여 adapter_tools만 로드한다."""
    mod_name = "src.inference.graph.tools.adapter_tools"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    # 상위 패키지 stub 등록 (아직 없는 경우만)
    stubs = {}
    for parent in ["src.inference.graph", "src.inference.graph.tools"]:
        if parent not in sys.modules:
            stub = types.ModuleType(parent)
            stub.__path__ = []
            sys.modules[parent] = stub
            stubs[parent] = True

    adapter_tools_path = str(
        pathlib.Path(__file__).resolve().parents[2]
        / "src"
        / "inference"
        / "graph"
        / "tools"
        / "adapter_tools.py"
    )

    spec = importlib.util.spec_from_file_location(mod_name, adapter_tools_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    # stub 패키지 복원 — 다른 테스트가 정상 import 할 수 있도록
    for parent in stubs:
        del sys.modules[parent]

    return mod


_adapter_tools_mod = _import_adapter_tools()
build_adapter_tools = _adapter_tools_mod.build_adapter_tools


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
