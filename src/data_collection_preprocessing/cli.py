"""CLI entry point: python -m src.data_collection_preprocessing"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DataConfig
from .pipeline import CivilResponseDataPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="민원답변 어댑터 학습 데이터 파이프라인")
    parser.add_argument(
        "--raw-dir",
        default="data/raw/aihub",
        help="AI Hub 원시 데이터 루트 디렉터리 (기본: data/raw/aihub)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="출력 디렉터리 (기본: data/processed)",
    )
    parser.add_argument(
        "--min-answer-length",
        type=int,
        default=30,
        help="최소 답변 길이 (기본: 30자)",
    )
    parser.add_argument(
        "--max-answer-length",
        type=int,
        default=4096,
        help="최대 답변 길이 (기본: 4096자)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="train 비율 (기본: 0.9)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    config = DataConfig(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        min_answer_length=args.min_answer_length,
        max_answer_length=args.max_answer_length,
        train_ratio=args.train_ratio,
    )

    pipeline = CivilResponseDataPipeline(config)
    stats = pipeline.run()

    print("\n파이프라인 완료")
    print(f"  총 레코드: {stats['total']:,}")
    print(f"  train:    {stats['train']:,}")
    print(f"  val:      {stats['val']:,}")
    print(f"  출력 경로: {args.output_dir}/")


if __name__ == "__main__":
    main()
