"""
Calibration Dataset Generator for AWQ Quantization

Generates calibration datasets for AWQ (Activation-aware Weight Quantization)
of the EXAONE-Deep-7.8B model. The calibration dataset is used to determine
optimal quantization parameters by analyzing activation patterns.

Requirements:
- Representative samples from the target domain (civil complaints)
- Diverse samples covering different categories and lengths
- Proper tokenization matching the target model
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import hashlib

from .config import get_config, CalibrationConfig
from .data_preprocessor import ProcessedRecord

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSample:
    """A single calibration sample"""

    text: str
    token_count: int
    category: str
    source: str


@dataclass
class CalibrationStats:
    """Statistics for the calibration dataset"""

    total_samples: int = 0
    total_tokens: int = 0
    avg_tokens_per_sample: float = 0.0
    category_distribution: Dict[str, int] = field(default_factory=dict)
    source_distribution: Dict[str, int] = field(default_factory=dict)
    min_tokens: int = 0
    max_tokens: int = 0


class CalibrationDatasetGenerator:
    """
    Calibration Dataset Generator for AWQ Quantization

    Creates a representative calibration dataset from processed civil
    complaint data for use in AWQ quantization of EXAONE-Deep-7.8B.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None, tokenizer: Optional[Any] = None):
        """
        Initialize the generator.

        Args:
            config: Calibration configuration
            tokenizer: HuggingFace tokenizer (optional, uses simple estimation if None)
        """
        self.config = config or get_config().calibration
        self.tokenizer = tokenizer

        # Create output directory
        self.output_dir = Path(self.config.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducibility
        random.seed(self.config.random_seed)

        # Track selected samples
        self._selected_hashes: set = set()

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses actual tokenizer if available, otherwise uses a simple
        estimation based on character count.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Tokenizer failed, using estimation: {e}")

        # Simple estimation: ~1 token per 4 characters for Korean
        # This is approximate and should be replaced with actual tokenizer
        return max(1, len(text) // 4)

    def _format_calibration_text(self, record: ProcessedRecord) -> str:
        """
        Format a processed record into calibration text.

        Uses the EXAONE chat template format to ensure calibration
        data matches actual inference patterns.

        Args:
            record: Processed record

        Returns:
            Formatted text for calibration
        """
        # EXAONE-Deep chat template format
        template = f"""[|user|]
{record.instruction}

{record.input}
[|assistant|]
{record.output}
[|endofturn|]"""

        return template

    def _compute_hash(self, text: str) -> str:
        """Compute hash for deduplication"""
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()

    def _is_duplicate(self, text: str) -> bool:
        """Check if sample is duplicate"""
        hash_val = self._compute_hash(text)
        if hash_val in self._selected_hashes:
            return True
        self._selected_hashes.add(hash_val)
        return False

    def _select_diverse_samples(
        self, records: List[ProcessedRecord], num_samples: int
    ) -> List[ProcessedRecord]:
        """
        Select diverse samples for calibration.

        Ensures representation across categories and token lengths.

        Args:
            records: All available records
            num_samples: Number of samples to select

        Returns:
            Selected records
        """
        if len(records) <= num_samples:
            return records

        # Group by category
        by_category: Dict[str, List[ProcessedRecord]] = {}
        for record in records:
            cat = record.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(record)

        # Calculate samples per category
        num_categories = len(by_category)
        base_per_category = num_samples // num_categories
        extra = num_samples % num_categories

        selected = []
        categories = list(by_category.keys())
        random.shuffle(categories)

        for idx, category in enumerate(categories):
            cat_records = by_category[category]
            random.shuffle(cat_records)

            # Allocate samples
            n = base_per_category + (1 if idx < extra else 0)
            n = min(n, len(cat_records))

            # Select records with diverse token counts
            if len(cat_records) > n:
                # Sort by estimated token count and select evenly distributed
                sorted_records = sorted(
                    cat_records,
                    key=lambda r: self._estimate_tokens(self._format_calibration_text(r)),
                )
                step = len(sorted_records) // n
                selected.extend(sorted_records[::step][:n])
            else:
                selected.extend(cat_records[:n])

        # If we still need more samples, add randomly
        remaining = num_samples - len(selected)
        if remaining > 0:
            all_remaining = [r for r in records if r not in selected]
            random.shuffle(all_remaining)
            selected.extend(all_remaining[:remaining])

        return selected[:num_samples]

    def generate_calibration_dataset(
        self,
        records: List[ProcessedRecord],
        num_samples: Optional[int] = None,
        max_seq_length: Optional[int] = None,
    ) -> List[CalibrationSample]:
        """
        Generate calibration dataset from processed records.

        Args:
            records: List of processed records
            num_samples: Number of samples (uses config default if None)
            max_seq_length: Maximum sequence length (uses config default if None)

        Returns:
            List of calibration samples
        """
        num_samples = num_samples or self.config.num_samples
        max_seq_length = max_seq_length or self.config.seq_length

        logger.info(
            f"Generating calibration dataset: {num_samples} samples, "
            f"max {max_seq_length} tokens"
        )

        # Select diverse samples
        selected_records = self._select_diverse_samples(records, num_samples * 2)

        calibration_samples = []

        for record in selected_records:
            if len(calibration_samples) >= num_samples:
                break

            # Format text
            text = self._format_calibration_text(record)

            # Check for duplicates
            if self._is_duplicate(text):
                continue

            # Estimate tokens
            token_count = self._estimate_tokens(text)

            # Skip if too long
            if token_count > max_seq_length:
                # Truncate if necessary
                text = text[: max_seq_length * 4]  # Rough character limit
                token_count = self._estimate_tokens(text)

            sample = CalibrationSample(
                text=text, token_count=token_count, category=record.category, source=record.source
            )
            calibration_samples.append(sample)

        logger.info(f"Generated {len(calibration_samples)} calibration samples")
        return calibration_samples

    def compute_statistics(self, samples: List[CalibrationSample]) -> CalibrationStats:
        """
        Compute statistics for calibration dataset.

        Args:
            samples: List of calibration samples

        Returns:
            CalibrationStats object
        """
        if not samples:
            return CalibrationStats()

        stats = CalibrationStats(
            total_samples=len(samples),
            total_tokens=sum(s.token_count for s in samples),
            min_tokens=min(s.token_count for s in samples),
            max_tokens=max(s.token_count for s in samples),
        )

        stats.avg_tokens_per_sample = stats.total_tokens / stats.total_samples

        # Category distribution
        for sample in samples:
            stats.category_distribution[sample.category] = (
                stats.category_distribution.get(sample.category, 0) + 1
            )

            stats.source_distribution[sample.source] = (
                stats.source_distribution.get(sample.source, 0) + 1
            )

        return stats

    def save_calibration_dataset(
        self,
        samples: List[CalibrationSample],
        filename: str = "calibration_dataset",
        format: str = "json",
    ) -> Dict[str, Path]:
        """
        Save calibration dataset to files.

        Args:
            samples: List of calibration samples
            filename: Base filename
            format: Output format

        Returns:
            Dictionary of saved file paths
        """
        paths = {}

        # Save as JSON (full data)
        json_path = self.output_dir / f"{filename}.json"
        data = {
            "config": {
                "num_samples": self.config.num_samples,
                "seq_length": self.config.seq_length,
                "random_seed": self.config.random_seed,
            },
            "samples": [
                {
                    "text": s.text,
                    "token_count": s.token_count,
                    "category": s.category,
                    "source": s.source,
                }
                for s in samples
            ],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        paths["json"] = json_path
        logger.info(f"Saved calibration dataset to {json_path}")

        # Save as plain text (for direct AWQ consumption)
        txt_path = self.output_dir / f"{filename}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(sample.text + "\n\n")

        paths["txt"] = txt_path
        logger.info(f"Saved calibration text to {txt_path}")

        # Save statistics
        stats = self.compute_statistics(samples)
        stats_path = self.output_dir / f"{filename}_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_samples": stats.total_samples,
                    "total_tokens": stats.total_tokens,
                    "avg_tokens_per_sample": stats.avg_tokens_per_sample,
                    "min_tokens": stats.min_tokens,
                    "max_tokens": stats.max_tokens,
                    "category_distribution": stats.category_distribution,
                    "source_distribution": stats.source_distribution,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        paths["stats"] = stats_path
        logger.info(f"Saved calibration stats to {stats_path}")

        return paths

    def generate_and_save(
        self, records: List[ProcessedRecord], filename: str = "calibration_dataset"
    ) -> Dict[str, Path]:
        """
        Generate and save calibration dataset in one step.

        Args:
            records: List of processed records
            filename: Base filename

        Returns:
            Dictionary of saved file paths
        """
        samples = self.generate_calibration_dataset(records)
        return self.save_calibration_dataset(samples, filename)


def generate_sample_calibration_data(output_dir: Path, num_samples: int = 50) -> Path:
    """
    Generate sample calibration data for testing.

    Args:
        output_dir: Output directory
        num_samples: Number of samples

    Returns:
        Path to created file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sample processed records
    from .data_preprocessor import ProcessedRecord

    categories = [
        "road/traffic",
        "environment/sanitation",
        "housing/construction",
        "welfare/health",
        "safety/disaster",
        "administration",
    ]

    records = []
    for i in range(num_samples * 2):  # Create more than needed for selection
        record = ProcessedRecord(
            id=f"CAL_{i:05d}",
            instruction="Please analyze the following civil complaint and provide a response.",
            input=f"[Category: {categories[i % len(categories)]}]\nComplaint: Sample complaint text for calibration sample {i}.",
            output=f"<thought>\n1. Analysis step\n2. Processing step\n</thought>\nSample response for calibration {i}.",
            category=categories[i % len(categories)],
            original_question_length=50 + i % 100,
            original_answer_length=80 + i % 50,
            source="sample",
        )
        records.append(record)

    # Generate calibration dataset
    generator = CalibrationDatasetGenerator()
    paths = generator.generate_and_save(records, "sample_calibration")

    logger.info(f"Created sample calibration dataset: {paths}")
    return paths.get("json", output_dir / "sample_calibration.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate sample calibration data
    config = get_config()
    output_path = Path(config.calibration.output_path)

    sample_path = generate_sample_calibration_data(output_path, num_samples=50)
    print(f"\nSample calibration dataset created at: {sample_path}")

    # Show statistics
    with open(output_path / "sample_calibration_stats.json", "r") as f:
        stats = json.load(f)

    print("\nCalibration Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Avg tokens/sample: {stats['avg_tokens_per_sample']:.1f}")
    print(f"  Token range: {stats['min_tokens']} - {stats['max_tokens']}")
    print(f"  Categories: {list(stats['category_distribution'].keys())}")
