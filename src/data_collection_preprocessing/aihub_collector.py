"""
AI Hub Data Collector Module

Collects and processes civil complaint datasets from AI Hub.
Supports dataset keys:
- 71852: Public Civil Complaint LLM Data (Priority 1)
- 71844: Private Civil Complaint LLM Data (Priority 2)
- 98: Call Center Q&A Data
- 619: Civil Complaint Automation Language Data
"""

import json
import logging
import os
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .config import AIHubConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about an AI Hub dataset"""

    key: str
    name: str
    description: str
    expected_format: str  # json, csv, etc.
    priority: int


# Known dataset information
KNOWN_DATASETS: Dict[str, DatasetInfo] = {
    "71852": DatasetInfo(
        key="71852",
        name="Public Civil Complaint LLM Data",
        description="Public institution civil complaint Q&A with reasoning process",
        expected_format="json",
        priority=1,
    ),
    "71844": DatasetInfo(
        key="71844",
        name="Private Civil Complaint LLM Data",
        description="Private sector civil complaint consultation and summary data",
        expected_format="json",
        priority=2,
    ),
    "98": DatasetInfo(
        key="98",
        name="Call Center Q&A Data",
        description="Traditional call center Q&A pairs with standard answers",
        expected_format="json",
        priority=3,
    ),
    "619": DatasetInfo(
        key="619",
        name="Civil Complaint Automation Language Data",
        description="Legal and administrative terminology for NLP processing",
        expected_format="json",
        priority=4,
    ),
}


class AIHubCollector:
    """
    AI Hub Dataset Collector

    Handles downloading and initial processing of AI Hub datasets
    using the aihubshell command-line tool.
    """

    def __init__(self, config: Optional[AIHubConfig] = None):
        """
        Initialize the collector.

        Args:
            config: AI Hub configuration. If None, uses default config.
        """
        self.config = config or get_config().aihub
        self.download_dir = Path(self.config.download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self._validate_setup()

    def _validate_setup(self) -> bool:
        """Validate that aihubshell and API key are available"""
        if not self.config.api_key:
            logger.warning(
                "AI Hub API key not configured. "
                "Set AIHUB_API_KEY environment variable or update config."
            )
            return False

        shell_path = Path(self.config.shell_path)
        if not shell_path.exists():
            logger.warning(
                f"aihubshell not found at {shell_path}. "
                "Download from: curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do"
            )
            return False

        return True

    def list_datasets(self, search_term: str = "civil complaint") -> List[Dict[str, Any]]:
        """
        List available datasets matching search term.

        Args:
            search_term: Term to search for in dataset names

        Returns:
            List of matching dataset information
        """
        try:
            cmd = [self.config.shell_path, "-mode", "l"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                logger.error(f"Failed to list datasets: {result.stderr}")
                return []

            # Parse output and filter by search term
            datasets = []
            for line in result.stdout.split("\n"):
                if search_term.lower() in line.lower() or "civil complaint" in line.lower():
                    datasets.append({"raw": line})

            return datasets

        except subprocess.TimeoutExpired:
            logger.error("Dataset listing timed out")
            return []
        except FileNotFoundError:
            logger.error(f"aihubshell not found at {self.config.shell_path}")
            return []

    def get_dataset_info(self, dataset_key: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific dataset.

        Args:
            dataset_key: The dataset key (e.g., "71852")

        Returns:
            Dataset information dictionary or None
        """
        # Return known info if available
        if dataset_key in KNOWN_DATASETS:
            info = KNOWN_DATASETS[dataset_key]
            return {
                "key": info.key,
                "name": info.name,
                "description": info.description,
                "format": info.expected_format,
                "priority": info.priority,
            }

        # Otherwise, query AI Hub
        try:
            cmd = [self.config.shell_path, "-mode", "l", "-datasetkey", dataset_key]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                return {"key": dataset_key, "info": result.stdout}

            return None

        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            return None

    def download_dataset(
        self, dataset_key: str, file_key: Optional[str] = None, output_dir: Optional[str] = None
    ) -> Optional[Path]:
        """
        Download a dataset from AI Hub.

        Args:
            dataset_key: The dataset key to download
            file_key: Optional specific file key within the dataset
            output_dir: Optional custom output directory

        Returns:
            Path to downloaded file/directory or None if failed
        """
        if not self.config.api_key:
            logger.error("API key is required for downloading datasets")
            return None

        output_path = Path(output_dir) if output_dir else self.download_dir / dataset_key
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting download for dataset {dataset_key}...")

        try:
            cmd = [
                self.config.shell_path,
                "-mode",
                "d",
                "-datasetkey",
                dataset_key,
                "-aihubapikey",
                self.config.api_key,
            ]

            if file_key:
                cmd.extend(["-filekey", file_key])

            # Run in the output directory
            result = subprocess.run(
                cmd,
                cwd=str(output_path),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for large downloads
            )

            if result.returncode == 0:
                logger.info(f"Successfully downloaded dataset {dataset_key}")
                return output_path
            else:
                logger.error(f"Download failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Download timed out for dataset {dataset_key}")
            return None
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None

    def download_all_priority_datasets(self, max_concurrent: int = 2) -> Dict[str, Optional[Path]]:
        """
        Download all priority datasets concurrently.

        Args:
            max_concurrent: Maximum number of concurrent downloads

        Returns:
            Dictionary mapping dataset keys to download paths
        """
        results = {}

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_key = {
                executor.submit(self.download_dataset, key): key for key in self.config.dataset_keys
            }

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(f"Failed to download {key}: {e}")
                    results[key] = None

        return results

    def extract_archive(self, archive_path: Path) -> Optional[Path]:
        """
        Extract a downloaded archive (zip, tar.gz, etc.)

        Args:
            archive_path: Path to the archive file

        Returns:
            Path to extracted directory or None
        """
        extract_dir = archive_path.parent / archive_path.stem

        try:
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(extract_dir)
                logger.info(f"Extracted {archive_path} to {extract_dir}")
                return extract_dir

            # Add support for other formats as needed
            logger.warning(f"Unsupported archive format: {archive_path.suffix}")
            return None

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return None

    def find_json_files(self, directory: Path) -> List[Path]:
        """
        Find all JSON files in a directory recursively.

        Args:
            directory: Directory to search

        Returns:
            List of JSON file paths
        """
        if not directory.exists():
            return []

        return list(directory.rglob("*.json"))

    def load_json_dataset(
        self, json_path: Path, encoding: str = "utf-8"
    ) -> Optional[Dict[str, Any]]:
        """
        Load a JSON dataset file.

        Args:
            json_path: Path to JSON file
            encoding: File encoding

        Returns:
            Parsed JSON data or None
        """
        try:
            with open(json_path, "r", encoding=encoding) as f:
                data = json.load(f)
            logger.info(f"Loaded {json_path} ({len(str(data))} chars)")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in {json_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load {json_path}: {e}")
            return None

    def iterate_dataset(
        self, directory: Path, batch_size: int = 1000
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Iterate through all JSON files in a dataset directory.
        Yields batches of records for memory-efficient processing.

        Args:
            directory: Dataset directory
            batch_size: Number of records per batch

        Yields:
            Batches of data records
        """
        json_files = self.find_json_files(directory)
        logger.info(f"Found {len(json_files)} JSON files in {directory}")

        batch = []

        for json_path in json_files:
            data = self.load_json_dataset(json_path)
            if not data:
                continue

            # Handle different JSON structures
            records = []
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Check for common data field names
                for key in ["data", "items", "records", "documents"]:
                    if key in data and isinstance(data[key], list):
                        records = data[key]
                        break
                else:
                    # Single record
                    records = [data]

            for record in records:
                batch.append(record)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        # Yield remaining records
        if batch:
            yield batch

    def get_download_instructions(self) -> str:
        """
        Get instructions for manual dataset download.

        Returns:
            Formatted instruction string
        """
        instructions = """
========================================
AI Hub Dataset Download Instructions
========================================

1. Download aihubshell:
   curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do
   chmod +x aihubshell

2. List available datasets:
   ./aihubshell -mode l | grep 'civil complaint'

3. Get your API key:
   - Visit https://aihub.or.kr
   - Login and go to My Page
   - Request API key (sent via email)

4. Download priority datasets:
"""
        for key, info in KNOWN_DATASETS.items():
            instructions += f"""
   # {info.name} (Priority {info.priority})
   ./aihubshell -mode d -datasetkey {key} -aihubapikey 'YOUR_API_KEY'
"""

        instructions += """
5. Set environment variables:
   export AIHUB_API_KEY='your_api_key_here'
   export AIHUB_DOWNLOAD_DIR='/path/to/download/dir'

========================================
"""
        return instructions


def create_mock_dataset(output_path: Path, num_samples: int = 100) -> Path:
    """
    Create a mock dataset for testing purposes.

    Args:
        output_path: Output directory
        num_samples: Number of mock samples to generate

    Returns:
        Path to created mock dataset
    """
    output_path.mkdir(parents=True, exist_ok=True)

    mock_data = {
        "info": {
            "name": "Mock Civil Complaint Dataset",
            "version": "1.0",
            "description": "Test dataset for development",
        },
        "data": [],
    }

    categories = [
        "road/traffic",
        "environment",
        "housing",
        "welfare",
        "culture",
        "economy",
        "education",
        "safety",
    ]

    templates = [
        {
            "question": "Our neighborhood road has potholes that need repair.",
            "answer": "Thank you for your report. We have forwarded this to the road maintenance department.",
            "category": "road/traffic",
        },
        {
            "question": "Illegal parking is blocking the fire lane every evening.",
            "answer": "We will increase enforcement patrols in your area.",
            "category": "road/traffic",
        },
        {
            "question": "The streetlights on our block have been out for a week.",
            "answer": "We have scheduled a repair crew to visit your area within 3 business days.",
            "category": "safety",
        },
    ]

    for i in range(num_samples):
        template = templates[i % len(templates)]
        mock_data["data"].append(
            {
                "id": f"MOCK_{i:05d}",
                "question": f"{template['question']} (Case #{i})",
                "answer": template["answer"],
                "category": template["category"],
                "date": "2024-01-01",
            }
        )

    output_file = output_path / "mock_civil_complaints.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mock_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Created mock dataset with {num_samples} samples at {output_file}")
    return output_file


if __name__ == "__main__":
    # Test the collector
    collector = AIHubCollector()

    # Print download instructions
    print(collector.get_download_instructions())

    # Check configuration
    print(f"\nAPI Key configured: {bool(collector.config.api_key)}")
    print(f"Download directory: {collector.download_dir}")

    # Create mock dataset for testing
    mock_path = create_mock_dataset(collector.download_dir / "mock", num_samples=50)
    print(f"\nMock dataset created: {mock_path}")

    # Test iteration
    print("\nTesting dataset iteration:")
    for batch in collector.iterate_dataset(collector.download_dir / "mock", batch_size=10):
        print(f"  Batch size: {len(batch)}")
