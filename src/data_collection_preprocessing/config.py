"""
Configuration Module for Data Collection and Preprocessing

Manages environment variables, API keys, and pipeline configurations.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("data_pipeline.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)


@dataclass
class AIHubConfig:
    """AI Hub API Configuration"""

    api_key: str = field(default_factory=lambda: os.getenv("AIHUB_API_KEY", ""))

    # Priority dataset keys for civil complaint data
    # Dataset 71852: Public Civil Complaint LLM Data (highest priority)
    # Dataset 71844: Private Civil Complaint LLM Data
    # Dataset 98: Call Center Q&A Data
    # Dataset 619: Civil Complaint Automation Language Data
    dataset_keys: List[str] = field(default_factory=lambda: ["71852", "71844", "98", "619"])

    # aihubshell binary path
    shell_path: str = field(default_factory=lambda: os.getenv("AIHUB_SHELL_PATH", "./aihubshell"))

    # Download directory
    download_dir: str = field(
        default_factory=lambda: os.getenv(
            "AIHUB_DOWNLOAD_DIR",
            str(Path(__file__).parent.parent.parent / "data" / "raw" / "aihub"),
        )
    )


@dataclass
class PreprocessingConfig:
    """Data Preprocessing Configuration"""

    # Minimum text length for valid complaint
    min_complaint_length: int = 20

    # Minimum answer length
    min_answer_length: int = 10

    # Maximum text length (for model context window)
    max_text_length: int = 4096

    # Train/Val/Test split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Output directories
    processed_dir: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent.parent / "data" / "processed")
    )

    # EXAONE template configuration
    instruction_template: str = (
        "다음 민원에 대해 단계적으로 분석하고, "
        "표준 서식에 맞춰 공손하고 명확한 답변을 작성하세요."
    )

    # Categories for civil complaints (Korean local government standard)
    categories: List[str] = field(
        default_factory=lambda: [
            "도로/교통",
            "환경/위생",
            "주택/건축",
            "복지/보건",
            "문화/체육",
            "경제/일자리",
            "교육/청소년",
            "안전/재난",
            "행정/민원",
            "기타",
        ]
    )


@dataclass
class CalibrationConfig:
    """AWQ Quantization Calibration Dataset Configuration"""

    # Number of samples for calibration
    num_samples: int = 512

    # Sequence length for calibration
    seq_length: int = 2048

    # Random seed for reproducibility
    random_seed: int = 42

    # Output path
    output_path: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent.parent / "data" / "calibration")
    )


@dataclass
class Config:
    """Main Configuration Class"""

    # Sub-configurations
    aihub: AIHubConfig = field(default_factory=AIHubConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # General settings
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    # Logging level
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def __post_init__(self):
        """Initialize directories and validate configuration"""
        self._create_directories()
        self._validate_config()

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.aihub.download_dir,
            self.preprocessing.processed_dir,
            self.calibration.output_path,
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {dir_path}")

    def _validate_config(self):
        """Validate configuration values"""
        # Check train/val/test split ratios
        total_ratio = (
            self.preprocessing.train_ratio
            + self.preprocessing.val_ratio
            + self.preprocessing.test_ratio
        )
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Train/Val/Test ratios must sum to 1.0, got {total_ratio}")

        # Warn if API keys are not set
        if not self.aihub.api_key:
            logger.warning("AI Hub API key is not set. Set AIHUB_API_KEY environment variable.")

    def get_api_status(self) -> Dict[str, bool]:
        """Check which API keys are configured"""
        return {
            "aihub": bool(self.aihub.api_key),
        }

    @classmethod
    def from_env_file(cls, env_path: str = ".env") -> "Config":
        """Load configuration from a specific .env file"""
        load_dotenv(env_path, override=True)
        return cls()


# Create a default configuration instance
default_config = Config()


def get_config() -> Config:
    """Get the default configuration instance"""
    return default_config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Project root: {config.project_root}")
    print(f"API Status: {config.get_api_status()}")
    print(f"AI Hub datasets: {config.aihub.dataset_keys}")
    print(f"Preprocessing categories: {config.preprocessing.categories}")
