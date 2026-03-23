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
        # Robust path resolution: Use the file's location as a base
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        
        if not os.path.isabs(self.agents_dir):
            self.agents_dir = os.path.join(current_file_dir, "agents")

        if not os.path.exists(self.agents_dir):
            # Try project root fallback if needed
            project_root = os.path.dirname(os.path.dirname(current_file_dir))
            self.agents_dir = os.path.join(project_root, "src", "inference", "agents")

        if not os.path.exists(self.agents_dir):
            print(f"Error: Agents directory not found at {self.agents_dir}")
            return

        for filename in os.listdir(self.agents_dir):
            if filename.endswith(".md"):
                agent_name = filename[:-3]
                file_path = os.path.join(self.agents_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        # Better YAML parsing: check if starts and has a closing ---
                        if content.startswith("---"):
                            parts = content.split("---")
                            if len(parts) >= 3:
                                # Body is everything after the second ---
                                self.personas[agent_name] = "---".join(parts[2:]).strip()
                            else:
                                self.personas[agent_name] = content
                        else:
                            self.personas[agent_name] = content
                except Exception as e:
                    print(f"Failed to load agent {agent_name}: {e}")
        
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
