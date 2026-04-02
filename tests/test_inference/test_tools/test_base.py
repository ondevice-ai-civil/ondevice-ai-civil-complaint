"""
BaseTool, ToolInput, ToolOutput, ToolRegistry 단위 테스트.
"""

import pytest

from src.inference.tools.base import BaseTool, ToolInput, ToolOutput, ToolRegistry


# ---------------------------------------------------------------------------
# 테스트용 구현체
# ---------------------------------------------------------------------------


class DummyInput(ToolInput):
    value: str = "hello"


class SuccessTool(BaseTool):
    name = "success_tool"
    description = "항상 성공하는 테스트 tool"

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        return ToolOutput(success=True, data={"echo": "ok"})


class FailTool(BaseTool):
    name = "fail_tool"
    description = "항상 실패하는 테스트 tool"

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        raise RuntimeError("의도적 실패")


# ---------------------------------------------------------------------------
# BaseTool 테스트
# ---------------------------------------------------------------------------


class TestBaseTool:
    @pytest.mark.asyncio
    async def test_run_success(self):
        tool = SuccessTool()
        output = await tool.run(DummyInput())
        assert output.success is True
        assert output.data == {"echo": "ok"}
        assert output.elapsed_ms is not None
        assert output.elapsed_ms >= 0
        assert output.tool_name == "success_tool"

    @pytest.mark.asyncio
    async def test_run_failure_returns_error_output(self):
        tool = FailTool()
        output = await tool.run(DummyInput())
        assert output.success is False
        assert "의도적 실패" in output.error
        assert output.tool_name == "fail_tool"
        assert output.elapsed_ms is not None

    def test_get_schema(self):
        tool = SuccessTool()
        schema = tool.get_schema()
        assert schema["name"] == "success_tool"
        assert schema["description"] == "항상 성공하는 테스트 tool"


# ---------------------------------------------------------------------------
# ToolOutput 테스트
# ---------------------------------------------------------------------------


class TestToolOutput:
    def test_default_values(self):
        output = ToolOutput()
        assert output.success is True
        assert output.data is None
        assert output.error is None

    def test_error_output(self):
        output = ToolOutput(success=False, error="test error")
        assert output.success is False
        assert output.error == "test error"


# ---------------------------------------------------------------------------
# ToolRegistry 테스트
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = SuccessTool()
        registry.register(tool)
        assert registry.get("success_tool") is tool
        assert "success_tool" in registry
        assert len(registry) == 1

    def test_get_nonexistent_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_execute_registered_tool(self):
        registry = ToolRegistry()
        registry.register(SuccessTool())
        output = await registry.execute("success_tool", DummyInput())
        assert output.success is True

    @pytest.mark.asyncio
    async def test_execute_unregistered_tool(self):
        registry = ToolRegistry()
        output = await registry.execute("nonexistent", DummyInput())
        assert output.success is False
        assert "등록되지 않은 tool" in output.error

    def test_register_empty_name_raises(self):
        registry = ToolRegistry()

        class NoNameTool(BaseTool):
            name = ""
            description = "no name"

            async def execute(self, tool_input):
                return ToolOutput()

        with pytest.raises(ValueError, match="비어 있습니다"):
            registry.register(NoNameTool())

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(SuccessTool())
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "success_tool"

    def test_duplicate_register_overwrites(self):
        registry = ToolRegistry()
        registry.register(SuccessTool())
        registry.register(SuccessTool())
        assert len(registry) == 1
