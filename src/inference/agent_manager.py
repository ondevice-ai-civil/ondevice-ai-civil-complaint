"""
Agent Manager for multi-persona prompt construction.

Loads agent persona definitions from Markdown files with YAML frontmatter,
and builds EXAONE chat template prompts with the appropriate system message.

Issue: #56
"""

import os
import re
from typing import Dict, List, Optional

import yaml
from loguru import logger


class AgentPersona:
    """Parsed agent persona from a Markdown file."""

    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.name = name
        self.role = role
        self.description = description
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __repr__(self) -> str:
        return f"AgentPersona(name={self.name!r}, role={self.role!r})"


class AgentManager:
    """
    Loads and manages agent personas from Markdown files.

    Each agent file uses YAML frontmatter for configuration and Markdown body
    for the system prompt content.

    Usage:
        manager = AgentManager("agents/")
        persona = manager.get_agent("classifier")
        prompt = manager.build_prompt("classifier", "민원 내용을 분류해주세요.")
    """

    _FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)

    def __init__(self, agents_dir: str = "agents"):
        self.agents_dir = agents_dir
        self._agents: Dict[str, AgentPersona] = {}
        self._load_agents()

    _DANGEROUS_TOKENS = ["[|user|]", "[|assistant|]", "[|system|]", "[|endofturn|]"]

    def _load_agents(self) -> None:
        base_dir = os.path.realpath(self.agents_dir)
        if not os.path.isdir(base_dir):
            logger.warning(f"Agents directory not found: {base_dir}")
            return

        for filename in os.listdir(base_dir):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.realpath(os.path.join(base_dir, filename))
            if not filepath.startswith(base_dir + os.sep):
                logger.warning(f"경로 순회 시도 감지, 무시됨: {filename}")
                continue
            try:
                agent = self._parse_agent_file(filepath)
                self._agents[agent.name] = agent
                logger.info(f"Loaded agent: {agent.name} ({agent.role})")
            except Exception as e:
                logger.error(f"Failed to load agent from {filename}: {e}")

    def _parse_agent_file(self, filepath: str) -> AgentPersona:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        match = self._FRONTMATTER_RE.match(content)
        if not match:
            raise ValueError(f"Invalid agent file format (missing YAML frontmatter): {filepath}")

        frontmatter = yaml.safe_load(match.group(1))
        body = match.group(2).strip()

        name = (frontmatter.get("name") or "").strip()
        if not name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"유효하지 않은 에이전트 이름: {name!r} in {filepath}")

        return AgentPersona(
            name=name,
            role=frontmatter.get("role", ""),
            description=frontmatter.get("description", ""),
            system_prompt=body,
            temperature=float(frontmatter.get("temperature", 0.7)),
            max_tokens=int(frontmatter.get("max_tokens", 512)),
        )

    def get_agent(self, name: str) -> Optional[AgentPersona]:
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    def build_prompt(self, agent_name: str, user_message: str) -> str:
        """
        Build an EXAONE chat template prompt with the agent's system message.

        Format:
            [|system|]{system_prompt}[|endofturn|]
            [|user|]{user_message}[|endofturn|]
            [|assistant|]
        """
        agent = self._agents.get(agent_name)
        if agent is None:
            raise ValueError(f"Unknown agent: {agent_name}")

        for token in self._DANGEROUS_TOKENS:
            if token in user_message:
                raise ValueError(f"이스케이프되지 않은 특수 토큰 감지: {token}")

        return (
            f"[|system|]{agent.system_prompt}[|endofturn|]"
            f"\n[|user|]{user_message}[|endofturn|]"
            f"\n[|assistant|]"
        )
