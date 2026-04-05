#!/usr/bin/env bash
set -euo pipefail

# GovOn Runtime을 HuggingFace Spaces에 배포하는 스크립트
# Usage: ./scripts/deploy-hfspace.sh

SPACE_REPO="${SPACE_REPO:-umyunsang/govon-runtime}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN 환경변수가 필요합니다}"

echo "=== GovOn HF Spaces 배포 ==="
echo "Space: $SPACE_REPO"

# 1. Space 생성 (이미 있으면 skip)
python3 -c "
from huggingface_hub import create_repo
create_repo('$SPACE_REPO', repo_type='space', space_sdk='docker', exist_ok=True, token='$HF_TOKEN', private=True)
print('Space repo ready')
"

# 2. 필요 파일 업로드
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')

# Dockerfile
api.upload_file(path_or_fileobj='Dockerfile.hfspace', path_in_repo='Dockerfile',
    repo_id='$SPACE_REPO', repo_type='space')

# requirements.txt
api.upload_file(path_or_fileobj='requirements.txt', path_in_repo='requirements.txt',
    repo_id='$SPACE_REPO', repo_type='space')

# src/ 디렉터리
api.upload_folder(folder_path='src', path_in_repo='src',
    repo_id='$SPACE_REPO', repo_type='space',
    ignore_patterns=['__pycache__', '*.pyc', '.pytest_cache'])

# agents/ 디렉터리 (존재하면)
import os
if os.path.isdir('agents'):
    api.upload_folder(folder_path='agents', path_in_repo='agents',
        repo_id='$SPACE_REPO', repo_type='space')

print('Files uploaded')
"

# 3. Secrets 설정
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.add_space_secret('$SPACE_REPO', 'HF_TOKEN', '$HF_TOKEN')
print('Secrets configured')
"

echo ""
echo "배포 완료"
echo "https://huggingface.co/spaces/$SPACE_REPO"
echo ""
echo "하드웨어를 설정하려면:"
echo "  python3 -c \"from huggingface_hub import HfApi; HfApi(token='$HF_TOKEN').request_space_hardware('$SPACE_REPO', 'a100-large')\""
