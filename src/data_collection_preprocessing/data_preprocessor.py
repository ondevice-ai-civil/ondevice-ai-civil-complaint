"""
Data Preprocessor Module

Transforms collected civil complaint data into EXAONE-Deep-7.8B fine-tuning format.
Handles:
- Data cleaning and validation
- PII masking
- EXAONE instruction-response format conversion
- Train/validation/test splitting
- Data quality reporting
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Generator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import Counter
import random

from .config import get_config, PreprocessingConfig
from .pii_masking import PIIMasker, mask_pii

logger = logging.getLogger(__name__)


@dataclass
class ProcessedRecord:
    """Represents a processed training record"""
    id: str
    instruction: str
    input: str
    output: str
    category: str
    original_question_length: int
    original_answer_length: int
    source: str  # aihub, seoul_api, etc.


@dataclass
class DataQualityReport:
    """Data quality report after preprocessing"""
    total_raw_records: int = 0
    total_processed_records: int = 0
    filtered_too_short: int = 0
    filtered_duplicates: int = 0
    filtered_invalid: int = 0
    category_distribution: Dict[str, int] = field(default_factory=dict)
    avg_question_length: float = 0.0
    avg_answer_length: float = 0.0
    pii_masked_count: int = 0
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return f"""
Data Quality Report
==================
Total Raw Records: {self.total_raw_records}
Total Processed Records: {self.total_processed_records}
Filtered (too short): {self.filtered_too_short}
Filtered (duplicates): {self.filtered_duplicates}
Filtered (invalid): {self.filtered_invalid}
Average Question Length: {self.avg_question_length:.1f}
Average Answer Length: {self.avg_answer_length:.1f}
PII Masked Count: {self.pii_masked_count}
Processing Time: {self.processing_time_seconds:.2f}s

Category Distribution:
{self._format_categories()}
"""

    def _format_categories(self) -> str:
        lines = []
        for cat, count in sorted(
            self.category_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            lines.append(f"  - {cat}: {count}")
        return "\n".join(lines)


class DataPreprocessor:
    """
    Data Preprocessor for EXAONE Fine-tuning

    Transforms raw civil complaint data into instruction-tuning format
    with proper cleaning, validation, and PII masking.
    """

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        pii_masker: Optional[PIIMasker] = None
    ):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration
            pii_masker: PII masker instance
        """
        self.config = config or get_config().preprocessing
        self.pii_masker = pii_masker or PIIMasker.create_strict_masker()

        # Create output directory
        self.output_dir = Path(self.config.processed_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track processed hashes for deduplication
        self._processed_hashes: set = set()

        # Quality metrics
        self.report = DataQualityReport()

    def _compute_hash(self, text: str) -> str:
        """Compute hash for deduplication"""
        return hashlib.md5(text.encode()).hexdigest()

    def _is_duplicate(self, question: str, answer: str) -> bool:
        """Check if record is duplicate"""
        combined = question + answer
        hash_val = self._compute_hash(combined)
        if hash_val in self._processed_hashes:
            return True
        self._processed_hashes.add(hash_val)
        return False

    def _validate_record(
        self,
        question: str,
        answer: str
    ) -> Tuple[bool, str]:
        """
        Validate a single record.

        Args:
            question: Question/complaint text
            answer: Answer text

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check minimum lengths
        if len(question.strip()) < self.config.min_complaint_length:
            return False, "question_too_short"

        if len(answer.strip()) < self.config.min_answer_length:
            return False, "answer_too_short"

        # Check maximum length
        if len(question) + len(answer) > self.config.max_text_length:
            return False, "too_long"

        # Check for empty or whitespace-only content
        if not question.strip() or not answer.strip():
            return False, "empty_content"

        return True, "valid"

    def _normalize_category(self, category: str) -> str:
        """Normalize category to standard categories"""
        if not category:
            return "other"

        category_lower = category.lower().strip()

        # Map common variations to standard categories
        category_mapping = {
            "road": "road/traffic",
            "traffic": "road/traffic",
            "transportation": "road/traffic",
            "environment": "environment/sanitation",
            "sanitation": "environment/sanitation",
            "housing": "housing/construction",
            "construction": "housing/construction",
            "welfare": "welfare/health",
            "health": "welfare/health",
            "culture": "culture/sports",
            "sports": "culture/sports",
            "economy": "economy/jobs",
            "jobs": "economy/jobs",
            "employment": "economy/jobs",
            "education": "education/youth",
            "youth": "education/youth",
            "safety": "safety/disaster",
            "disaster": "safety/disaster",
            "administration": "administration",
            "civil complaint": "administration",
        }

        for key, value in category_mapping.items():
            if key in category_lower:
                return value

        return "other"

    def _generate_thought_process(
        self,
        category: str,
        question: str
    ) -> str:
        """
        Generate a thinking process for EXAONE's <thought> tag.

        Args:
            category: Complaint category
            question: Complaint question

        Returns:
            Thought process text
        """
        # Extract key phrases for analysis
        key_phrases = self._extract_key_phrases(question)

        thought = f"""<thought>
1. Complaint Type Analysis: Identified as {category} related request.
2. Key Information Extraction: Main issues - {', '.join(key_phrases[:3]) if key_phrases else 'General inquiry'}.
3. Regulation Review: Checking relevant local government ordinances and handling procedures.
4. Response Composition: Preparing appropriate handling procedures and expected timeline.
</thought>"""

        return thought

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text (simplified implementation)"""
        # Simple keyword extraction based on common complaint terms
        keywords = [
            "repair", "maintenance", "parking", "noise", "construction",
            "lighting", "waste", "water", "safety", "welfare",
            "complaint", "request", "report", "inquiry"
        ]

        found = []
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                found.append(keyword)

        return found

    def _format_exaone_record(
        self,
        record_id: str,
        question: str,
        answer: str,
        category: str,
        source: str = "unknown"
    ) -> Optional[ProcessedRecord]:
        """
        Format a record into EXAONE instruction-tuning format.

        Args:
            record_id: Unique record ID
            question: Question/complaint text
            answer: Answer text
            category: Complaint category
            source: Data source identifier

        Returns:
            ProcessedRecord or None if invalid
        """
        # Validate record
        is_valid, reason = self._validate_record(question, answer)
        if not is_valid:
            if reason == "question_too_short" or reason == "answer_too_short":
                self.report.filtered_too_short += 1
            else:
                self.report.filtered_invalid += 1
            return None

        # Check for duplicates
        if self._is_duplicate(question, answer):
            self.report.filtered_duplicates += 1
            return None

        # Mask PII
        masked_question = self.pii_masker.mask_all(question)
        masked_answer = self.pii_masker.mask_all(answer)

        if masked_question != question or masked_answer != answer:
            self.report.pii_masked_count += 1

        # Normalize category
        normalized_category = self._normalize_category(category)

        # Generate thought process
        thought_process = self._generate_thought_process(normalized_category, masked_question)

        # Create formatted record
        record = ProcessedRecord(
            id=record_id,
            instruction=self.config.instruction_template,
            input=f"[Category: {normalized_category}]\nComplaint Content: {masked_question}",
            output=f"{thought_process}\n{masked_answer}",
            category=normalized_category,
            original_question_length=len(question),
            original_answer_length=len(answer),
            source=source
        )

        return record

    def process_raw_data(
        self,
        raw_data: List[Dict[str, Any]],
        source: str = "unknown",
        question_field: str = "question",
        answer_field: str = "answer",
        category_field: str = "category",
        id_field: str = "id"
    ) -> List[ProcessedRecord]:
        """
        Process a list of raw data records.

        Args:
            raw_data: List of raw record dictionaries
            source: Data source identifier
            question_field: Field name for question
            answer_field: Field name for answer
            category_field: Field name for category
            id_field: Field name for ID

        Returns:
            List of processed records
        """
        start_time = datetime.now()
        
        # Special handling for 98 (Dasan Call Center) pairing
        if source == "aihub" and any("도메인" in r and r["도메인"] == "다산콜센터" for r in raw_data[:10]):
            logger.info("Detected Dasan Call Center format (98). Matching Q&A pairs...")
            dialog_map = {}
            for r in raw_data:
                did = r.get("대화셋일련번호")
                if not did: continue
                if did not in dialog_map: dialog_map[did] = {"Q": "", "A": "", "cat": r.get("카테고리", "기타")}
                if r.get("QA") == "Q":
                    dialog_map[did]["Q"] = r.get("고객질문(요청)", "")
                elif r.get("QA") == "A":
                    dialog_map[did]["A"] = r.get("상담사답변", "")
            
            new_raw = []
            for did, content in dialog_map.items():
                if content["Q"] and content["A"]:
                    new_raw.append({
                        "question": content["Q"],
                        "answer": content["A"],
                        "category": content["cat"],
                        "id": did,
                        "_source": "aihub"
                    })
            raw_data = new_raw

        self.report.total_raw_records += len(raw_data)
        processed_records = []

        for idx, raw_record in enumerate(raw_data):
            # Special handling for 71852/71844 Consulting Content
            if "consulting_content" in raw_record:
                content = raw_record["consulting_content"]
                if "Q :" in content and "A :" in content:
                    parts = content.split("A :")
                    q_part = parts[0].replace("제목 :", "").replace("Q :", "").strip()
                    a_part = parts[1].strip()
                    raw_record["question"] = q_part
                    raw_record["answer"] = a_part
                elif "instructions" in raw_record and raw_record["instructions"]:
                    instr = raw_record["instructions"][0]
                    if "data" in instr and instr["data"]:
                        raw_record["question"] = instr["data"][0].get("input", "")
                        raw_record["answer"] = instr["data"][0].get("output", "")

            # Extract fields (handle various naming conventions)
            question = (
                raw_record.get(question_field) or
                raw_record.get("QSTN_CONT") or
                raw_record.get("Q_refined") or
                raw_record.get("question_content") or
                raw_record.get("body") or
                raw_record.get("question") or
                ""
            )

            answer = (
                raw_record.get(answer_field) or
                raw_record.get("ANSW_CONT") or
                raw_record.get("answer_content") or
                raw_record.get("response") or
                raw_record.get("answer") or
                ""
            )

            category = (
                raw_record.get(category_field) or
                raw_record.get("MENU_NM") or
                raw_record.get("category_name") or
                raw_record.get("카테고리") or
                "other"
            )

            record_id = (
                raw_record.get(id_field) or
                raw_record.get("CASE_NO") or
                f"{source}_{idx:06d}"
            )

            # Process record
            processed = self._format_exaone_record(
                record_id=str(record_id),
                question=str(question),
                answer=str(answer),
                category=str(category),
                source=source
            )

            if processed:
                processed_records.append(processed)

        # Update report
        self.report.total_processed_records += len(processed_records)
        self.report.processing_time_seconds += (
            datetime.now() - start_time
        ).total_seconds()

        # Calculate statistics
        if processed_records:
            self.report.avg_question_length = sum(
                r.original_question_length for r in processed_records
            ) / len(processed_records)

            self.report.avg_answer_length = sum(
                r.original_answer_length for r in processed_records
            ) / len(processed_records)

            # Update category distribution
            for record in processed_records:
                self.report.category_distribution[record.category] = (
                    self.report.category_distribution.get(record.category, 0) + 1
                )

        logger.info(
            f"Processed {len(processed_records)}/{len(raw_data)} records from {source}"
        )

        return processed_records

    def split_dataset(
        self,
        records: List[ProcessedRecord],
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Tuple[List[ProcessedRecord], List[ProcessedRecord], List[ProcessedRecord]]:
        """
        Split dataset into train/validation/test sets.

        Args:
            records: List of processed records
            shuffle: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train, validation, test) record lists
        """
        if shuffle:
            random.seed(random_seed)
            records = records.copy()
            random.shuffle(records)

        total = len(records)
        train_end = int(total * self.config.train_ratio)
        val_end = train_end + int(total * self.config.val_ratio)

        train_set = records[:train_end]
        val_set = records[train_end:val_end]
        test_set = records[val_end:]

        logger.info(
            f"Dataset split: train={len(train_set)}, "
            f"val={len(val_set)}, test={len(test_set)}"
        )

        return train_set, val_set, test_set

    def save_dataset(
        self,
        records: List[ProcessedRecord],
        filename: str,
        format: str = "jsonl"
    ) -> Path:
        """
        Save processed records to file.

        Args:
            records: List of processed records
            filename: Output filename (without extension)
            format: Output format ('jsonl' or 'json')

        Returns:
            Path to saved file
        """
        if format == "jsonl":
            output_path = self.output_dir / f"{filename}.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for record in records:
                    record_dict = {
                        "id": record.id,
                        "instruction": record.instruction,
                        "input": record.input,
                        "output": record.output,
                    }
                    f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

        else:  # json
            output_path = self.output_dir / f"{filename}.json"
            records_list = [
                {
                    "id": r.id,
                    "instruction": r.instruction,
                    "input": r.input,
                    "output": r.output,
                }
                for r in records
            ]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records_list, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(records)} records to {output_path}")
        return output_path

    def save_all_splits(
        self,
        train: List[ProcessedRecord],
        val: List[ProcessedRecord],
        test: List[ProcessedRecord],
        prefix: str = "civil_complaint",
        format: str = "jsonl"
    ) -> Dict[str, Path]:
        """
        Save all dataset splits.

        Args:
            train: Training set
            val: Validation set
            test: Test set
            prefix: Filename prefix
            format: Output format

        Returns:
            Dictionary mapping split names to file paths
        """
        paths = {
            "train": self.save_dataset(train, f"{prefix}_train", format),
            "validation": self.save_dataset(val, f"{prefix}_val", format),
            "test": self.save_dataset(test, f"{prefix}_test", format),
        }

        # Save report
        report_path = self.output_dir / f"{prefix}_quality_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Quality report saved to {report_path}")
        paths["report"] = report_path

        return paths

    def get_report(self) -> DataQualityReport:
        """Get the data quality report"""
        return self.report

    def reset(self) -> None:
        """Reset preprocessor state"""
        self._processed_hashes.clear()
        self.report = DataQualityReport()
        self.pii_masker.reset_statistics()


def create_sample_processed_data(
    output_dir: Path,
    num_samples: int = 100
) -> Path:
    """
    Create sample processed data for testing.

    Args:
        output_dir: Output directory
        num_samples: Number of samples

    Returns:
        Path to created file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    categories = [
        "road/traffic", "environment/sanitation", "housing/construction",
        "welfare/health", "safety/disaster", "administration"
    ]

    templates = [
        {
            "question": "There are potholes in front of our apartment building causing difficulty for residents. Please repair them.",
            "answer": "Hello. Regarding your road repair request, our department has confirmed the location. Repairs are scheduled to begin within 7 days."
        },
        {
            "question": "Illegal dumping of waste in the alley is causing sanitation issues. Please take enforcement action.",
            "answer": "Thank you for your report. We have notified the environmental department and will increase patrols in your area."
        },
        {
            "question": "Late-night construction noise is disturbing sleep. Please check construction hours.",
            "answer": "We have confirmed the construction site's hours. They have been warned about nighttime work restrictions."
        }
    ]

    preprocessor = DataPreprocessor()

    for i in range(num_samples):
        template = templates[i % len(templates)]
        category = categories[i % len(categories)]

        raw_record = {
            "id": f"SAMPLE_{i:05d}",
            "question": f"{template['question']} (Case #{i})",
            "answer": template["answer"],
            "category": category
        }

        processed = preprocessor.process_raw_data(
            [raw_record],
            source="sample"
        )

        if processed:
            samples.extend(processed)

    # Save as JSONL
    output_path = output_dir / "sample_processed.jsonl"
    preprocessor.save_dataset(samples, "sample_processed", format="jsonl")

    logger.info(f"Created {len(samples)} sample records at {output_path}")
    return output_path


if __name__ == "__main__":
    # Test preprocessing
    logging.basicConfig(level=logging.INFO)

    preprocessor = DataPreprocessor()

    # Create sample data
    test_data = [
        {
            "id": "TEST_001",
            "question": "Contact me at 010-1234-5678. The road in front of our neighborhood has large potholes making it difficult to drive. Please repair them quickly.",
            "answer": "Hello. Thank you for your road repair request. Our road maintenance team has confirmed the location and repairs are scheduled to begin within 7 days.",
            "category": "road/traffic"
        },
        {
            "id": "TEST_002",
            "question": "Illegal parking continues every evening blocking the fire lane. This is a safety issue.",
            "answer": "Thank you for your report. We have notified the traffic enforcement team to increase patrols in your area.",
            "category": "road/traffic"
        },
        {
            "id": "TEST_003",
            "question": "Short",  # Will be filtered
            "answer": "OK",
            "category": "other"
        }
    ]

    # Process data
    processed = preprocessor.process_raw_data(test_data, source="test")

    print(f"\nProcessed {len(processed)} records")
    print(preprocessor.report)

    # Show sample output
    if processed:
        print("\nSample processed record:")
        sample = processed[0]
        print(f"ID: {sample.id}")
        print(f"Instruction: {sample.instruction}")
        print(f"Input: {sample.input[:200]}...")
        print(f"Output: {sample.output[:300]}...")
