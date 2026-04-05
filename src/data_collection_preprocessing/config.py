"""데이터 파이프라인 설정."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    raw_dir: Path = Path("data/raw/aihub")
    output_dir: Path = Path("data/processed")
    min_answer_length: int = 30
    max_answer_length: int = 4096
    min_question_length: int = 5
    train_ratio: float = 0.9
