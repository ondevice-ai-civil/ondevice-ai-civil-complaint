"""
Seoul Open Data API Collector Module

Collects civil complaint data from Seoul Open Data Portal (data.seoul.go.kr).
Primary endpoint: S_EUNGDAPSO_CASE_INFO (Seoul Eungdapso civil complaint cases)
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import get_config, SeoulAPIConfig

logger = logging.getLogger(__name__)


@dataclass
class APIEndpoint:
    """Seoul API endpoint information"""
    name: str
    service_name: str
    description: str
    fields: List[str]


# Known Seoul Open Data API endpoints
KNOWN_ENDPOINTS: Dict[str, APIEndpoint] = {
    "eungdapso": APIEndpoint(
        name="eungdapso",
        service_name="S_EUNGDAPSO_CASE_INFO",
        description="Seoul Eungdapso Civil Complaint Cases",
        fields=[
            "CASE_NO",      # Case number
            "MENU_NM",      # Category name
            "QSTN_CONT",    # Question content
            "ANSW_CONT",    # Answer content
            "QSTN_DT",      # Question date
            "ANSW_DT",      # Answer date
        ]
    ),
}


class SeoulAPICollector:
    """
    Seoul Open Data API Collector

    Handles fetching civil complaint data from Seoul Open Data Portal.
    Implements rate limiting, retry logic, and incremental saving.
    """

    def __init__(self, config: Optional[SeoulAPIConfig] = None):
        """
        Initialize the collector.

        Args:
            config: Seoul API configuration. If None, uses default config.
        """
        self.config = config or get_config().seoul_api
        self.download_dir = Path(self.config.download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Setup session with retry logic
        self.session = self._create_session()

        # Track collected data hashes for deduplication
        self._collected_hashes: set = set()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic"""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _build_url(
        self,
        service_name: str,
        start_index: int,
        end_index: int
    ) -> str:
        """
        Build the API request URL.

        Args:
            service_name: Name of the API service
            start_index: Start index for pagination (1-based)
            end_index: End index for pagination

        Returns:
            Complete API URL
        """
        return (
            f"{self.config.base_url}/"
            f"{self.config.api_key}/"
            f"{self.config.request_format}/"
            f"{service_name}/"
            f"{start_index}/"
            f"{end_index}/"
        )

    def _make_request(
        self,
        url: str,
        timeout: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Make an API request with error handling.

        Args:
            url: Request URL
            timeout: Request timeout in seconds

        Returns:
            JSON response or None if failed
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            data = response.json()

            # Check for API-level errors
            if "RESULT" in data:
                result = data["RESULT"]
                if result.get("CODE") != "INFO-000":
                    logger.error(f"API Error: {result.get('MESSAGE', 'Unknown error')}")
                    return None

            return data

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out: {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None

    def get_total_count(self, service_name: str) -> Optional[int]:
        """
        Get total number of records for a service.

        Args:
            service_name: API service name

        Returns:
            Total record count or None
        """
        if not self.config.api_key:
            logger.error("API key is required")
            return None

        url = self._build_url(service_name, 1, 1)
        data = self._make_request(url)

        if data and service_name in data:
            return data[service_name].get("list_total_count", 0)

        return None

    def fetch_batch(
        self,
        service_name: str,
        start_index: int,
        end_index: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch a batch of records.

        Args:
            service_name: API service name
            start_index: Start index (1-based)
            end_index: End index

        Returns:
            List of records
        """
        if not self.config.api_key:
            logger.warning("API key not set, returning empty results")
            return []

        url = self._build_url(service_name, start_index, end_index)
        logger.debug(f"Fetching {start_index}-{end_index} from {service_name}")

        data = self._make_request(url)

        if data and service_name in data:
            rows = data[service_name].get("row", [])
            return rows

        return []

    def _compute_hash(self, record: Dict[str, Any]) -> str:
        """Compute hash for deduplication"""
        # Use question content as primary key for deduplication
        content = record.get("QSTN_CONT", "") + record.get("ANSW_CONT", "")
        return hashlib.md5(content.encode()).hexdigest()

    def _is_duplicate(self, record: Dict[str, Any]) -> bool:
        """Check if record is duplicate"""
        hash_val = self._compute_hash(record)
        if hash_val in self._collected_hashes:
            return True
        self._collected_hashes.add(hash_val)
        return False

    def collect_all(
        self,
        endpoint_name: str = "eungdapso",
        max_records: Optional[int] = None,
        save_interval: int = 1000
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Collect all records from an endpoint.

        Args:
            endpoint_name: Name of the endpoint to collect from
            max_records: Maximum number of records to collect (None for all)
            save_interval: Save checkpoint every N records

        Yields:
            Individual records as dictionaries
        """
        if endpoint_name not in KNOWN_ENDPOINTS:
            logger.error(f"Unknown endpoint: {endpoint_name}")
            return

        endpoint = KNOWN_ENDPOINTS[endpoint_name]
        service_name = endpoint.service_name

        # Get total count
        total = self.get_total_count(service_name)
        if total is None:
            logger.error("Failed to get total count")
            return

        if max_records:
            total = min(total, max_records)

        logger.info(f"Collecting {total} records from {endpoint_name}")

        collected_count = 0
        checkpoint_data = []

        start_index = 1
        while start_index <= total:
            end_index = min(
                start_index + self.config.max_rows_per_request - 1,
                total
            )

            batch = self.fetch_batch(service_name, start_index, end_index)

            for record in batch:
                if not self._is_duplicate(record):
                    yield record
                    collected_count += 1
                    checkpoint_data.append(record)

                    # Save checkpoint
                    if collected_count % save_interval == 0:
                        self._save_checkpoint(checkpoint_data, endpoint_name, collected_count)
                        checkpoint_data = []

            start_index = end_index + 1

            # Rate limiting
            time.sleep(self.config.request_delay)

            logger.info(f"Progress: {collected_count}/{total} records")

        # Save remaining data
        if checkpoint_data:
            self._save_checkpoint(checkpoint_data, endpoint_name, collected_count)

        logger.info(f"Collection complete: {collected_count} records")

    def _save_checkpoint(
        self,
        data: List[Dict[str, Any]],
        endpoint_name: str,
        count: int
    ) -> None:
        """Save checkpoint file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{endpoint_name}_checkpoint_{count}_{timestamp}.json"
        filepath = self.download_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Checkpoint saved: {filepath}")

    def collect_and_save(
        self,
        endpoint_name: str = "eungdapso",
        output_filename: Optional[str] = None,
        max_records: Optional[int] = None
    ) -> Optional[Path]:
        """
        Collect all records and save to a single file.

        Args:
            endpoint_name: Name of the endpoint
            output_filename: Output filename (auto-generated if None)
            max_records: Maximum records to collect

        Returns:
            Path to saved file or None
        """
        all_records = []

        for record in self.collect_all(endpoint_name, max_records):
            all_records.append(record)

        if not all_records:
            logger.warning("No records collected")
            return None

        # Generate output filename
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{endpoint_name}_full_{len(all_records)}_{timestamp}.json"

        output_path = self.download_dir / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "source": "Seoul Open Data API",
                    "endpoint": endpoint_name,
                    "count": len(all_records),
                    "collected_at": datetime.now().isoformat(),
                },
                "data": all_records
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(all_records)} records to {output_path}")
        return output_path

    def load_collected_data(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Load previously collected data.

        Args:
            filepath: Path to JSON file

        Returns:
            List of records
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "data" in data:
                return data["data"]
            elif isinstance(data, list):
                return data
            else:
                logger.warning(f"Unexpected data structure in {filepath}")
                return []

        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return []

    def validate_api_key(self) -> bool:
        """
        Validate that the API key is working.

        Returns:
            True if API key is valid
        """
        if not self.config.api_key:
            return False

        # Try to fetch a single record
        count = self.get_total_count(KNOWN_ENDPOINTS["eungdapso"].service_name)
        return count is not None

    def get_api_setup_instructions(self) -> str:
        """
        Get instructions for API key setup.

        Returns:
            Formatted instruction string
        """
        return """
========================================
Seoul Open Data API Setup Instructions
========================================

1. Visit Seoul Open Data Portal:
   https://data.seoul.go.kr/

2. Create an account and login

3. Search for the dataset you need:
   - "Eungdapso Civil Complaint Cases" (recommended)

4. Go to dataset detail page and click [Request API Key]

5. Fill in usage purpose (e.g., "AI model training")
   and URL (can use "localhost" for development)

6. API key is issued immediately

7. Set environment variable:
   export SEOUL_API_KEY='your_api_key_here'

8. Or add to .env file:
   SEOUL_API_KEY=your_api_key_here

========================================
Available Endpoints:
"""


def create_mock_seoul_data(output_path: Path, num_samples: int = 100) -> Path:
    """
    Create mock Seoul API data for testing.

    Args:
        output_path: Output directory
        num_samples: Number of samples

    Returns:
        Path to created file
    """
    output_path.mkdir(parents=True, exist_ok=True)

    mock_data = {
        "metadata": {
            "source": "Mock Seoul Open Data",
            "endpoint": "eungdapso",
            "count": num_samples,
            "collected_at": datetime.now().isoformat(),
        },
        "data": []
    }

    categories = [
        "road/traffic", "environment/sanitation", "housing/construction",
        "welfare/health", "culture/sports", "economy/jobs",
        "education/youth", "safety/disaster", "administration", "other"
    ]

    templates = [
        {
            "question": "There are potholes on the road in front of my house, making it difficult to walk. Please repair them quickly.",
            "answer": "Hello. We will take action on your request for sidewalk repair. We will repair it as soon as possible."
        },
        {
            "question": "Illegal parking is blocking the fire lane every evening. Please strengthen enforcement.",
            "answer": "Regarding the illegal parking complaint, we have notified the relevant department to strengthen enforcement in the area."
        },
        {
            "question": "The streetlights on our block have been out for a week. It's dangerous at night.",
            "answer": "Thank you for reporting the streetlight outage. A repair crew will visit within 3 business days."
        },
        {
            "question": "Construction noise next door continues late at night. Please take action.",
            "answer": "We have checked the construction site regarding the noise complaint. We have guided them on operating hours."
        }
    ]

    for i in range(num_samples):
        template = templates[i % len(templates)]
        category = categories[i % len(categories)]

        mock_data["data"].append({
            "CASE_NO": f"SEOUL_{i:06d}",
            "MENU_NM": category,
            "QSTN_CONT": f"{template['question']} (Case #{i})",
            "ANSW_CONT": template["answer"],
            "QSTN_DT": "2024-01-15",
            "ANSW_DT": "2024-01-16",
        })

    output_file = output_path / "mock_seoul_complaints.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mock_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Created mock Seoul data with {num_samples} samples at {output_file}")
    return output_file


if __name__ == "__main__":
    # Test the collector
    collector = SeoulAPICollector()

    print(collector.get_api_setup_instructions())
    print(f"\nAPI Key configured: {bool(collector.config.api_key)}")
    print(f"Download directory: {collector.download_dir}")

    # Create mock data for testing
    mock_path = create_mock_seoul_data(
        collector.download_dir / "mock",
        num_samples=50
    )
    print(f"\nMock data created: {mock_path}")

    # Validate API key if configured
    if collector.config.api_key:
        print(f"\nValidating API key...")
        is_valid = collector.validate_api_key()
        print(f"API key valid: {is_valid}")
