"""
Tests for Calibration Dataset Generator Module
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data_collection_preprocessing.calibration_dataset import (
    CalibrationDatasetGenerator,
    CalibrationSample,
    CalibrationStats,
)
from src.data_collection_preprocessing.config import CalibrationConfig
from src.data_collection_preprocessing.data_preprocessor import ProcessedRecord


class TestCalibrationDatasetGenerator:
    """Tests for CalibrationDatasetGenerator class"""

    @pytest.fixture
    def generator(self):
        """Create a CalibrationDatasetGenerator instance with temp output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig()
            config.output_path = tmpdir
            yield CalibrationDatasetGenerator(config=config)

    @pytest.fixture
    def sample_records(self):
        """Create sample processed records"""
        categories = ["road/traffic", "environment/sanitation", "welfare/health"]
        records = []

        for i in range(100):
            record = ProcessedRecord(
                id=f"CAL_{i:05d}",
                instruction="Analyze the following civil complaint and provide a response.",
                input=f"[Category: {categories[i % len(categories)]}]\nComplaint: Sample complaint text {i} with details.",
                output=f"<thought>\n1. Analysis\n2. Processing\n</thought>\nResponse for complaint {i}.",
                category=categories[i % len(categories)],
                original_question_length=50 + i % 50,
                original_answer_length=80 + i % 30,
                source="test",
            )
            records.append(record)

        return records

    def test_generate_calibration_dataset(self, generator, sample_records):
        """Test generating calibration dataset"""
        samples = generator.generate_calibration_dataset(sample_records, num_samples=50)

        assert len(samples) == 50
        assert all(isinstance(s, CalibrationSample) for s in samples)

    def test_sample_diversity(self, generator, sample_records):
        """Test that samples are diverse across categories"""
        samples = generator.generate_calibration_dataset(sample_records, num_samples=30)

        categories = set(s.category for s in samples)
        # Should have representation from multiple categories
        assert len(categories) >= 2

    def test_token_estimation(self, generator):
        """Test token count estimation"""
        text = "This is a sample text for token estimation."
        token_count = generator._estimate_tokens(text)

        assert token_count > 0
        # Rough estimate check (about 1 token per 4 chars for this simple text)
        assert token_count < len(text)

    def test_format_calibration_text(self, generator, sample_records):
        """Test calibration text formatting"""
        record = sample_records[0]
        text = generator._format_calibration_text(record)

        assert "[|user|]" in text
        assert "[|assistant|]" in text
        assert "[|endofturn|]" in text
        assert record.instruction in text

    def test_deduplication(self, generator, sample_records):
        """Test that duplicate samples are handled"""
        # Add duplicate records
        duplicated = sample_records + sample_records[:10]

        samples = generator.generate_calibration_dataset(duplicated, num_samples=50)

        # Should not have more unique samples than original
        assert len(samples) <= 50


class TestCalibrationStats:
    """Tests for CalibrationStats"""

    def test_compute_statistics(self):
        """Test statistics computation"""
        samples = [
            CalibrationSample(text="Sample 1", token_count=100, category="cat1", source="test"),
            CalibrationSample(text="Sample 2", token_count=200, category="cat2", source="test"),
            CalibrationSample(text="Sample 3", token_count=150, category="cat1", source="test"),
        ]

        generator = CalibrationDatasetGenerator()
        stats = generator.compute_statistics(samples)

        assert stats.total_samples == 3
        assert stats.total_tokens == 450
        assert stats.avg_tokens_per_sample == 150
        assert stats.min_tokens == 100
        assert stats.max_tokens == 200
        assert stats.category_distribution["cat1"] == 2
        assert stats.category_distribution["cat2"] == 1


class TestCalibrationDatasetSaving:
    """Tests for saving calibration dataset"""

    @pytest.fixture
    def generator_with_samples(self, sample_records=None):
        """Create generator with generated samples"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig()
            config.output_path = tmpdir
            generator = CalibrationDatasetGenerator(config=config)

            # Create sample records
            records = []
            for i in range(50):
                record = ProcessedRecord(
                    id=f"CAL_{i:05d}",
                    instruction="Analyze the complaint.",
                    input=f"[Category: test]\nComplaint: Sample {i}",
                    output=f"<thought>Analysis</thought>\nResponse {i}",
                    category="test",
                    original_question_length=50,
                    original_answer_length=50,
                    source="test",
                )
                records.append(record)

            samples = generator.generate_calibration_dataset(records, num_samples=20)
            yield generator, samples, tmpdir

    def test_save_json_format(self, generator_with_samples):
        """Test saving in JSON format"""
        generator, samples, tmpdir = generator_with_samples

        paths = generator.save_calibration_dataset(samples, "test_calib")

        assert "json" in paths
        assert paths["json"].exists()

        with open(paths["json"], "r") as f:
            data = json.load(f)
            assert "config" in data
            assert "samples" in data
            assert len(data["samples"]) == len(samples)

    def test_save_txt_format(self, generator_with_samples):
        """Test saving in TXT format"""
        generator, samples, tmpdir = generator_with_samples

        paths = generator.save_calibration_dataset(samples, "test_calib")

        assert "txt" in paths
        assert paths["txt"].exists()

        with open(paths["txt"], "r") as f:
            content = f.read()
            # Should contain sample texts
            assert len(content) > 0

    def test_save_stats(self, generator_with_samples):
        """Test saving statistics"""
        generator, samples, tmpdir = generator_with_samples

        paths = generator.save_calibration_dataset(samples, "test_calib")

        assert "stats" in paths
        assert paths["stats"].exists()

        with open(paths["stats"], "r") as f:
            stats = json.load(f)
            assert "total_samples" in stats
            assert "total_tokens" in stats
            assert "category_distribution" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
