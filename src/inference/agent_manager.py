import os
from typing import Dict, Optional
import yaml

class AgentManager:
    """
    Manages specialized agent personas (system prompts) defined in Markdown files.
    Inspired by msitarzewski/agency-agents architecture.
    """
    def __init__(self, agents_dir: str = "src/inference/agents"):
        self.agents_dir = agents_dir
        self.personas: Dict[str, str] = {}
        self._load_all_agents()

    def _load_all_agents(self):
        """Load all .md agent definitions from the directory."""
        if not os.path.exists(self.agents_dir):
            # Fallback for relative path issues in different environments
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.agents_dir = os.path.join(base_dir, "inference", "agents")

        if not os.path.exists(self.agents_dir):
            print(f"Warning: Agents directory not found at {self.agents_dir}")
            return

        for filename in os.listdir(self.agents_dir):
            if filename.endswith(".md"):
                agent_name = filename[:-3]
                file_path = os.path.join(self.agents_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Optionally strip YAML frontmatter if present
                    if content.startswith("---"):
                        _, _, body = content.split("---", 2)
                        self.personas[agent_name] = body.strip()
                    else:
                        self.personas[agent_name] = content.strip()
        
        print(f"Loaded {len(self.personas)} agent personas: {list(self.personas.keys())}")

    def get_persona(self, agent_name: str) -> Optional[str]:
        """Get the system prompt for a specific agent."""
        return self.personas.get(agent_name)

    def wrap_with_persona(self, agent_name: str, user_content: str) -> str:
        """Wrap user content with the agent's system prompt using EXAONE chat template."""
        persona = self.get_persona(agent_name)
        if not persona:
            return user_content
        
        # Standard EXAONE Chat Template: [|system|]...[|endofturn|][|user|]...[|assistant|]
        return f"[|system|]{persona}[|endofturn|][|user|]{user_content}[|assistant|]"
