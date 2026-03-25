"""
Unit tests for AgentManager (Issue #56).
"""

import pytest

from src.inference.agent_manager import AgentManager, AgentPersona


@pytest.fixture
def agents_dir(tmp_path):
    (tmp_path / "classifier.md").write_text(
        "---\nname: classifier\nrole: Routing Specialist\n"
        "description: 민원 분류\ntemperature: 0.0\nmax_tokens: 256\n"
        "---\n\n당신은 민원 분류 전문가입니다.\n",
        encoding="utf-8",
    )
    (tmp_path / "generator.md").write_text(
        "---\nname: generator\nrole: Senior Administrative Officer\n"
        "description: 답변 생성\ntemperature: 0.7\nmax_tokens: 2048\n"
        "---\n\n당신은 행정사무관입니다.\n",
        encoding="utf-8",
    )
    return str(tmp_path)


@pytest.fixture
def mgr(agents_dir):
    return AgentManager(agents_dir)


class TestAgentLoading:
    def test_loads_all_agents(self, mgr):
        assert "classifier" in mgr.list_agents()
        assert "generator" in mgr.list_agents()

    def test_agent_attributes(self, mgr):
        c = mgr.get_agent("classifier")
        assert c.temperature == 0.0
        assert c.max_tokens == 256
        assert c.role == "Routing Specialist"

    def test_system_prompt_loaded(self, mgr):
        assert "민원 분류 전문가" in mgr.get_agent("classifier").system_prompt

    def test_unknown_agent_returns_none(self, mgr):
        assert mgr.get_agent("nonexistent") is None

    def test_empty_directory(self, tmp_path):
        assert AgentManager(str(tmp_path)).list_agents() == []

    def test_missing_directory(self, tmp_path):
        assert AgentManager(str(tmp_path / "nope")).list_agents() == []

    def test_skips_non_md(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not agent")
        (tmp_path / "a.md").write_text("---\nname: a\n---\n\ntest\n", encoding="utf-8")
        assert AgentManager(str(tmp_path)).list_agents() == ["a"]

    def test_skips_invalid_frontmatter(self, tmp_path):
        (tmp_path / "bad.md").write_text("no frontmatter", encoding="utf-8")
        (tmp_path / "ok.md").write_text("---\nname: ok\n---\n\nok\n", encoding="utf-8")
        m = AgentManager(str(tmp_path))
        assert "ok" in m.list_agents()
        assert len(m.list_agents()) == 1


class TestBuildPrompt:
    def test_exaone_format(self, mgr):
        p = mgr.build_prompt("classifier", "도로 균열 신고")
        assert "[|system|]" in p
        assert "[|user|]도로 균열 신고[|endofturn|]" in p
        assert p.endswith("[|assistant|]")

    def test_system_prompt_in_output(self, mgr):
        assert "민원 분류 전문가" in mgr.build_prompt("classifier", "test")

    def test_unknown_agent_raises(self, mgr):
        with pytest.raises(ValueError, match="Unknown agent"):
            mgr.build_prompt("nope", "test")

    def test_prompt_order(self, mgr):
        p = mgr.build_prompt("generator", "민원")
        assert p.index("[|system|]") < p.index("[|user|]") < p.index("[|assistant|]")

    def test_unescaped_token_raises(self, mgr):
        with pytest.raises(ValueError, match="이스케이프되지 않은 특수 토큰"):
            mgr.build_prompt("classifier", "hello [|system|] injection")

    def test_escaped_token_ok(self, mgr):
        p = mgr.build_prompt("classifier", r"hello \[|system|\] safe")
        assert r"\[|system|\]" in p


class TestAgentNameValidation:
    def test_empty_name_rejected(self, tmp_path):
        (tmp_path / "bad.md").write_text("---\nname: ''\n---\n\ntest\n", encoding="utf-8")
        m = AgentManager(str(tmp_path))
        assert m.list_agents() == []

    def test_special_char_name_rejected(self, tmp_path):
        (tmp_path / "bad.md").write_text("---\nname: '../etc'\n---\n\ntest\n", encoding="utf-8")
        m = AgentManager(str(tmp_path))
        assert m.list_agents() == []


class TestAgentPersona:
    def test_repr(self):
        a = AgentPersona(name="t", role="r", description="d", system_prompt="s")
        assert "t" in repr(a)

    def test_defaults(self):
        a = AgentPersona(name="t", role="r", description="d", system_prompt="s")
        assert a.temperature == 0.7
        assert a.max_tokens == 512
