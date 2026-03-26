"""
FAISS 인덱스 빌드 스크립트.

EmbeddingPipeline으로 v2_train.jsonl 데이터를 임베딩하고,
MultiIndexManager를 사용하여 CASE 타입 FAISS 인덱스를 빌드한다.

사용 예시::

    python scripts/build_faiss_index.py \\
        --data-path data/processed/v2_train.jsonl \\
        --index-dir models/faiss_index \\
        --batch-size 64

    # DB 적재 포함
    python scripts/build_faiss_index.py --with-db
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

from loguru import logger

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_collection_preprocessing.embedding import EmbeddingPipeline
from src.inference.index_manager import IndexType, MultiIndexManager


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="FAISS 인덱스 빌드 스크립트 (CASE 타입)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/v2_train.jsonl",
        help="JSONL 데이터 파일 경로 (기본: data/processed/v2_train.jsonl)",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="models/faiss_index",
        help="인덱스 저장 디렉토리 (기본: models/faiss_index)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="임베딩 배치 크기 (기본: 64)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="SentenceTransformer 모델명 (기본: intfloat/multilingual-e5-large)",
    )
    parser.add_argument(
        "--with-db",
        action="store_true",
        default=False,
        help="DB 적재 수행 여부 (document_source, index_version 테이블)",
    )
    return parser.parse_args()


def save_to_db(metadata_list, index_dir: str, total_documents: int, build_duration: float) -> None:
    """빌드 결과를 DB에 적재한다 (document_source + index_version)."""
    from src.inference.db.converters import dataclass_to_orm
    from src.inference.db.crud import (
        create_document_source,
        create_index_version,
        deactivate_versions,
    )
    from src.inference.db.database import SessionLocal

    # Phase 1: document_source 적재
    db = SessionLocal()
    try:
        logger.info(f"document_source 적재 시작: {len(metadata_list)}건")
        for i, meta in enumerate(metadata_list):
            content = meta.extras.get("complaint_text", "") if meta.extras else ""
            kwargs = dataclass_to_orm(meta, content)
            try:
                create_document_source(db, **kwargs)
            except Exception as e:
                db.rollback()
                logger.warning(f"document_source 적재 실패 (건너뜀): {meta.doc_id}: {e}")
                continue
            if (i + 1) % 1000 == 0:
                db.commit()
                logger.info(f"  {i + 1}/{len(metadata_list)}건 커밋 완료")
        db.commit()
        logger.info(f"document_source 적재 완료")
    except Exception as e:
        db.rollback()
        logger.error(f"document_source 적재 중 오류: {e}")
    finally:
        db.close()

    # Phase 2: index_version 등록 (원자적)
    db = SessionLocal()
    try:
        deactivate_versions(db, "case")
        index_file_path = os.path.join(index_dir, "case", "index.faiss")
        meta_file_path = os.path.join(index_dir, "case", "metadata.json")
        version_str = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
        create_index_version(
            db,
            index_type="case",
            version=version_str,
            total_documents=total_documents,
            index_file_path=index_file_path,
            meta_file_path=meta_file_path,
            build_duration_seconds=build_duration,
            is_active=True,
            notes=f"build_faiss_index.py로 빌드 (문서 {total_documents}건)",
        )
        db.commit()
        logger.info(f"index_version 등록 완료: {version_str}")
    except Exception as e:
        db.rollback()
        logger.error(f"index_version 등록 실패: {e}")
        raise
    finally:
        db.close()


def main() -> None:
    """메인 엔트리포인트."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("FAISS 인덱스 빌드 시작")
    logger.info(f"  데이터 경로: {args.data_path}")
    logger.info(f"  인덱스 디렉토리: {args.index_dir}")
    logger.info(f"  배치 크기: {args.batch_size}")
    logger.info(f"  모델: {args.model_name}")
    logger.info(f"  DB 적재: {args.with_db}")
    logger.info("=" * 60)

    start_time = time.time()

    # 1) 데이터 파일 존재 확인
    if not os.path.exists(args.data_path):
        logger.error(f"데이터 파일이 존재하지 않습니다: {args.data_path}")
        sys.exit(1)

    # 2) 임베딩 파이프라인 실행
    pipeline = EmbeddingPipeline(model_name=args.model_name)
    embeddings, metadata_list = pipeline.process_jsonl(args.data_path, batch_size=args.batch_size)

    # 3) MultiIndexManager로 인덱스 빌드
    logger.info("FAISS 인덱스 빌드 시작")
    manager = MultiIndexManager(base_dir=args.index_dir, embedding_dim=pipeline.embedding_dim)
    manager.add_documents(IndexType.CASE, embeddings, metadata_list)
    manager.save_index(IndexType.CASE)

    build_duration = time.time() - start_time

    # 4) 결과 요약
    stats = manager.get_index_stats()
    case_stats = stats["indexes"]["case"]
    logger.info("=" * 60)
    logger.info("빌드 완료 요약")
    logger.info(f"  총 문서 수: {case_stats['doc_count']}")
    logger.info(f"  메타데이터 수: {case_stats['metadata_count']}")
    logger.info(f"  인덱스 타입: {case_stats['index_class']}")
    logger.info(f"  임베딩 차원: {stats['embedding_dim']}")
    logger.info(f"  소요 시간: {build_duration:.1f}초")
    logger.info(f"  저장 경로: {args.index_dir}/case/")
    logger.info("=" * 60)

    # 5) DB 적재 (선택)
    if args.with_db:
        logger.info("DB 적재 시작")
        save_to_db(
            metadata_list,
            index_dir=args.index_dir,
            total_documents=case_stats["doc_count"],
            build_duration=build_duration,
        )
        logger.info("DB 적재 완료")

    logger.info("모든 작업 완료")


if __name__ == "__main__":
    main()
