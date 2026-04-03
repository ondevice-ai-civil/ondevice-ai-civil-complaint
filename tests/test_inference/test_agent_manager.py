"""AgentManager 테스트."""

import pytest

from src.inference.agent_manager import AgentManager, AgentPersona


@pytest.fixture
def agents_dir(tmp_path):
    (tmp_path / "retriever.md").write_text(
        "---\nname: retriever\nrole: Retrieval Specialist\n"
        "description: 로컬 검색\ntemperature: 0.0\nmax_tokens: 256\n"
        "---\n\n당신은 검색 전문가입니다.\n",
        encoding="utf-8",
    )
    (tmp_path / "generator_civil_response.md").write_text(
        "---\nname: generator_civil_response\nrole: Civil Response Officer\n"
        "description: 민원 답변 생성\ntemperature: 0.7\nmax_tokens: 2048\n"
        "---\n\n당신은 민원 회신 담당자입니다.\n",
        encoding="utf-8",
    )
    return str(tmp_path)


@pytest.fixture
def mgr(agents_dir):
    return AgentManager(agents_dir)


class TestAgentLoading:
    def test_loads_current_agents(self, mgr):
        assert "retriever" in mgr.list_agents()
        assert "generator_civil_response" in mgr.list_agents()

    def test_agent_attributes(self, mgr):
        retriever = mgr.get_agent("retriever")
        assert retriever.temperature == 0.0
        assert retriever.max_tokens == 256
        assert retriever.role == "Retrieval Specialist"

    def test_unknown_agent_returns_none(self, mgr):
        assert mgr.get_agent("nonexistent") is None

    def test_empty_directory(self, tmp_path):
        assert AgentManager(str(tmp_path)).list_agents() == []

    def test_missing_directory(self, tmp_path):
        assert AgentManager(str(tmp_path / "nope")).list_agents() == []


class TestBuildPrompt:
    def test_exaone_prompt_format(self, mgr):
        prompt = mgr.build_prompt("retriever", "도로 균열 신고")
        assert "[|system|]" in prompt
        assert "[|user|]도로 균열 신고[|endofturn|]" in prompt
        assert prompt.endswith("[|assistant|]")

    def test_unknown_agent_raises(self, mgr):
        with pytest.raises(ValueError, match="Unknown agent"):
            mgr.build_prompt("nope", "test")

    def test_unescaped_token_raises(self, mgr):
        with pytest.raises(ValueError, match="이스케이프되지 않은 특수 토큰"):
            mgr.build_prompt("retriever", "hello [|system|] injection")


class TestAgentNameValidation:
    def test_empty_name_rejected(self, tmp_path):
        (tmp_path / "bad.md").write_text("---\nname: ''\n---\n\ntest\n", encoding="utf-8")
        manager = AgentManager(str(tmp_path))
        assert manager.list_agents() == []

    def test_special_char_name_rejected(self, tmp_path):
        (tmp_path / "bad.md").write_text("---\nname: '../etc'\n---\n\ntest\n", encoding="utf-8")
        manager = AgentManager(str(tmp_path))
        assert manager.list_agents() == []


class TestAgentPersona:
    def test_repr(self):
        persona = AgentPersona(name="t", role="r", description="d", system_prompt="s")
        assert "t" in repr(persona)

    def test_defaults(self):
        persona = AgentPersona(name="t", role="r", description="d", system_prompt="s")
        assert persona.temperature == 0.7
        assert persona.max_tokens == 512
