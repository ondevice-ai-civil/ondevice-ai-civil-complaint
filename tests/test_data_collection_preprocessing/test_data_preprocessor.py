"""
Tests for Data Preprocessor Module
"""

import pytest
import json
from pathlib import Path
import tempfile

from src.data_collection_preprocessing.data_preprocessor import (
    DataPreprocessor,
    ProcessedRecord,
    DataQualityReport,
)
from src.data_collection_preprocessing.config import PreprocessingConfig


class TestDataPreprocessor:
    """Tests for DataPreprocessor class"""

    @pytest.fixture
    def preprocessor(self):
        """Create a DataPreprocessor instance with temp output directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PreprocessingConfig()
            config.processed_dir = tmpdir
            yield DataPreprocessor(config=config)

    @pytest.fixture
    def sample_data(self):
        """Create sample raw data for testing"""
        return [
            {
                "id": "TEST_001",
                "question": "There are large potholes on the road in front of our neighborhood, making it difficult to drive. Please repair them quickly.",
                "answer": "Hello. Thank you for your road repair request. We have forwarded this to the road maintenance department.",
                "category": "road/traffic",
            },
            {
                "id": "TEST_002",
                "question": "Illegal parking continues every evening, blocking the fire lane. Please strengthen enforcement.",
                "answer": "We have notified the traffic enforcement team to increase patrols in your area.",
                "category": "road/traffic",
            },
            {
                "id": "TEST_003",
                "question": "Construction noise next door continues late at night. Please take action.",
                "answer": "We have checked the construction site. They have been warned about operating hours.",
                "category": "environment/sanitation",
            },
        ]

    def test_process_valid_data(self, preprocessor, sample_data):
        """Test processing valid data records"""
        processed = preprocessor.process_raw_data(sample_data, source="test")

        assert len(processed) == 3
        assert all(isinstance(r, ProcessedRecord) for r in processed)

    def test_filter_short_content(self, preprocessor):
        """Test that short content is filtered out"""
        short_data = [{"id": "SHORT_001", "question": "Short", "answer": "OK", "category": "test"}]

        processed = preprocessor.process_raw_data(short_data, source="test")
        assert len(processed) == 0
        assert preprocessor.report.filtered_too_short >= 1

    def test_filter_duplicates(self, preprocessor, sample_data):
        """Test duplicate filtering"""
        # Add duplicate record
        duplicate_data = sample_data + [sample_data[0].copy()]

        processed = preprocessor.process_raw_data(duplicate_data, source="test")

        # Should have one less record due to duplicate
        assert len(processed) == 3
        assert preprocessor.report.filtered_duplicates >= 1

    def test_exaone_format_output(self, preprocessor, sample_data):
        """Test that output is in EXAONE format"""
        processed = preprocessor.process_raw_data(sample_data, source="test")

        for record in processed:
            assert record.instruction is not None
            assert record.input is not None
            assert record.output is not None
            assert "<thought>" in record.output

    def test_category_normalization(self, preprocessor):
        """Test category normalization"""
        test_data = [
            {
                "id": "CAT_001",
                "question": "Road issue complaint with sufficient detail for processing.",
                "answer": "Thank you for the detailed report.",
                "category": "road",
            },
            {
                "id": "CAT_002",
                "question": "Traffic violation report with sufficient detail for processing.",
                "answer": "We will investigate this matter.",
                "category": "traffic",
            },
            {
                "id": "CAT_003",
                "question": "Environment issue complaint with sufficient detail for processing.",
                "answer": "Environmental team notified.",
                "category": "environment",
            },
        ]

        processed = preprocessor.process_raw_data(test_data, source="test")

        categories = [r.category for r in processed]
        # All road/traffic related categories should normalize to road/traffic
        assert "road/traffic" in categories or all(c == "other" for c in categories)

    def test_pii_masking_integration(self, preprocessor):
        """Test that PII is masked during preprocessing"""
        pii_data = [
            {
                "id": "PII_001",
                "question": "Please contact me at 010-1234-5678 regarding the pothole repair request.",
                "answer": "We will contact you at the provided number.",
                "category": "road/traffic",
            }
        ]

        processed = preprocessor.process_raw_data(pii_data, source="test")

        assert len(processed) == 1
        # Phone number should be masked
        assert "010-1234-5678" not in processed[0].input
        assert preprocessor.report.pii_masked_count >= 1

    def test_split_dataset(self, preprocessor, sample_data):
        """Test dataset splitting"""
        # Create more records for meaningful split
        extended_data = sample_data * 10  # 30 records
        for i, record in enumerate(extended_data):
            record = record.copy()
            record["id"] = f"SPLIT_{i:03d}"
            extended_data[i] = record

        processed = preprocessor.process_raw_data(extended_data, source="test")
        train, val, test = preprocessor.split_dataset(processed, shuffle=True)

        total = len(train) + len(val) + len(test)
        assert total == len(processed)

        # Check approximate ratios (80/10/10)
        assert len(train) >= len(val)
        assert len(train) >= len(test)


class TestDataQualityReport:
    """Tests for DataQualityReport"""

    def test_report_initialization(self):
        """Test report initialization"""
        report = DataQualityReport()
        assert report.total_raw_records == 0
        assert report.total_processed_records == 0

    def test_report_to_dict(self):
        """Test report serialization"""
        report = DataQualityReport(
            total_raw_records=100,
            total_processed_records=90,
            filtered_too_short=5,
            filtered_duplicates=5,
        )

        data = report.to_dict()
        assert data["total_raw_records"] == 100
        assert data["total_processed_records"] == 90

    def test_report_string_representation(self):
        """Test report string formatting"""
        report = DataQualityReport(
            total_raw_records=100,
            total_processed_records=90,
            category_distribution={"road/traffic": 50, "environment": 40},
        )

        report_str = str(report)
        assert "100" in report_str
        assert "90" in report_str


class TestDatasetSaving:
    """Tests for dataset saving functionality"""

    @pytest.fixture
    def preprocessor_with_data(self):
        """Create preprocessor with processed data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PreprocessingConfig()
            config.processed_dir = tmpdir
            preprocessor = DataPreprocessor(config=config)

            sample_data = [
                {
                    "id": f"SAVE_{i:03d}",
                    "question": f"Sample complaint text number {i} with sufficient detail for processing and testing purposes.",
                    "answer": f"Standard response for complaint {i} with adequate length.",
                    "category": "road/traffic",
                }
                for i in range(10)
            ]

            processed = preprocessor.process_raw_data(sample_data, source="test")
            yield preprocessor, processed, tmpdir

    def test_save_jsonl_format(self, preprocessor_with_data):
        """Test saving in JSONL format"""
        preprocessor, records, tmpdir = preprocessor_with_data

        path = preprocessor.save_dataset(records, "test_dataset", format="jsonl")

        assert path.exists()
        assert path.suffix == ".jsonl"

        # Verify content
        with open(path, "r") as f:
            lines = f.readlines()
            assert len(lines) == len(records)

            first_record = json.loads(lines[0])
            assert "instruction" in first_record
            assert "input" in first_record
            assert "output" in first_record

    def test_save_json_format(self, preprocessor_with_data):
        """Test saving in JSON format"""
        preprocessor, records, tmpdir = preprocessor_with_data

        path = preprocessor.save_dataset(records, "test_dataset", format="json")

        assert path.exists()
        assert path.suffix == ".json"

        with open(path, "r") as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == len(records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
