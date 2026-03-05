# Data Collection and Preprocessing Module

This module provides a complete data pipeline for collecting, cleaning, and preparing civil complaint data for EXAONE-Deep-7.8B fine-tuning and AWQ quantization.

## Overview

The pipeline consists of the following components:

1. **Data Collection**: Collect civil complaint data from AI Hub and Seoul Open Data API
2. **PII Masking**: Detect and mask personal identifiable information
3. **Data Preprocessing**: Clean, validate, and transform data to EXAONE format
4. **Calibration Dataset**: Generate calibration data for AWQ quantization

## Directory Structure

```
src/data_collection_preprocessing/
    __init__.py              # Module initialization
    config.py                # Configuration management
    aihub_collector.py       # AI Hub data collector
    seoul_api_collector.py   # Seoul Open Data API collector
    pii_masking.py           # PII detection and masking
    data_preprocessor.py     # Data preprocessing and EXAONE format conversion
    calibration_dataset.py   # AWQ calibration dataset generator
    pipeline.py              # Main pipeline orchestrator
    requirements.txt         # Module-specific dependencies
    .env.example             # Environment variables template
    README.md                # This file
```

## Quick Start

### 1. Environment Setup

```bash
# Copy and configure environment variables
cp src/data_collection_preprocessing/.env.example .env

# Edit .env with your API keys
# - AIHUB_API_KEY: Get from https://aihub.or.kr
# - SEOUL_API_KEY: Get from https://data.seoul.go.kr
```

### 2. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install with pip (editable mode)
pip install -e .
```

### 3. Run the Pipeline

```bash
# Run full pipeline with mock data (for testing)
python -m src.data_collection_preprocessing.pipeline --mode full --mock

# Run full pipeline with real data
python -m src.data_collection_preprocessing.pipeline --mode full

# Collect data only
python -m src.data_collection_preprocessing.pipeline --mode collect

# Preprocess existing data
python -m src.data_collection_preprocessing.pipeline --mode preprocess --input data/raw/raw_combined.json
```

## Data Sources

### AI Hub Datasets (Priority Order)

| Dataset Key | Name | Description |
|-------------|------|-------------|
| 71852 | Public Civil Complaint LLM Data | Primary dataset with reasoning process |
| 71844 | Private Civil Complaint LLM Data | Extended domain coverage |
| 98 | Call Center Q&A Data | Traditional Q&A pairs |
| 619 | Civil Complaint Automation Data | Legal/administrative terminology |

### Seoul Open Data API

- **Endpoint**: S_EUNGDAPSO_CASE_INFO (Seoul Eungdapso Civil Complaint Cases)
- **Fields**: Case number, category, question, answer, dates

## Output Format

### Training Data (JSONL)

```json
{
  "id": "AIHUB_000001",
  "instruction": "Please analyze the following civil complaint and provide a response.",
  "input": "[Category: road/traffic]\nComplaint: There are potholes on our street...",
  "output": "<thought>\n1. Complaint Type Analysis: road/traffic request.\n2. Key Information: Pothole location identified.\n3. Regulation Review: Local maintenance guidelines.\n4. Response: Scheduled repair.\n</thought>\nThank you for your report. We have scheduled road repairs..."
}
```

### Calibration Dataset

- **JSON format**: Full metadata with token counts
- **TXT format**: Plain text for direct AWQ consumption
- **Statistics**: Category distribution, token distribution

## Configuration

Key configuration options in `config.py`:

```python
# Preprocessing settings
min_complaint_length = 20      # Minimum characters for valid complaint
min_answer_length = 10         # Minimum characters for valid answer
max_text_length = 4096         # Maximum total text length
train_ratio = 0.8              # Training set ratio
val_ratio = 0.1                # Validation set ratio
test_ratio = 0.1               # Test set ratio

# Calibration settings
num_samples = 512              # Number of calibration samples
seq_length = 2048              # Maximum sequence length
```

## API Key Setup

### AI Hub

1. Visit https://aihub.or.kr and create an account
2. Go to My Page and request API key
3. Download aihubshell tool:
   ```bash
   curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do
   chmod +x aihubshell
   ```

### Seoul Open Data

1. Visit https://data.seoul.go.kr and create an account
2. Search for "Eungdapso" dataset
3. Click "Request API Key" on the dataset page
4. API key is issued immediately

## PII Masking

The module automatically detects and masks:

- Korean resident registration numbers
- Phone numbers (mobile and landline)
- Email addresses
- Physical addresses
- Bank account numbers
- Credit card numbers
- Vehicle license plates
- IP addresses
- Korean names (heuristic detection)

## Troubleshooting

### Common Issues

1. **API key not working**
   - Verify the key in your .env file
   - Check if the key has been activated
   - Some APIs require approval time

2. **Download failures**
   - Check network connectivity
   - Large datasets may timeout; retry with smaller batches
   - Verify aihubshell binary permissions

3. **Encoding errors**
   - Ensure UTF-8 encoding for all files
   - Some API responses may need encoding handling

### Debug Mode

```bash
# Enable verbose logging
python -m src.data_collection_preprocessing.pipeline --mode full --verbose
```

## Testing

```bash
# Run tests
pytest tests/test_data_collection_preprocessing/

# Run with coverage
pytest --cov=src/data_collection_preprocessing tests/
```

## Output Files

After running the pipeline, the following files are generated:

```
data/
    raw/
        aihub/          # Raw AI Hub data
        seoul_api/      # Raw Seoul API data
    processed/
        civil_complaint_train.jsonl    # Training data
        civil_complaint_val.jsonl      # Validation data
        civil_complaint_test.jsonl     # Test data
        civil_complaint_quality_report.json  # Quality metrics
    calibration/
        civil_complaint_calibration.json  # Calibration data with metadata
        civil_complaint_calibration.txt   # Plain text for AWQ
        civil_complaint_calibration_stats.json  # Statistics
```

## Related Documents

- [PRD Document](/docs/prd.md)
- [Dataset Specifications](/docs/outputs/M1_Planning/03_Data_Collection/dataset_and_environment_specs.md)
- [Crawling Targets](/docs/outputs/M1_Planning/03_Data_Collection/crawling_targets.md)
