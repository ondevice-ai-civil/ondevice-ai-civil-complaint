#!/usr/bin/env python3
"""
FAISS 벡터 검색 시스템 평가 스크립트 (self-retrieval 테스트).

카테고리별 균등 샘플링으로 평가 데이터셋을 구성하고,
Recall@5, Recall@10, MRR, 검색 레이턴시(p50/p95/p99)를 측정한다.

주의: --eval-jsonl 옵션을 지정하지 않으면 인덱스 빌드에 사용된 동일 데이터로
평가를 수행하는 self-retrieval 테스트가 된다. 이 경우 Recall이 인위적으로
높게 측정될 수 있으므로, 실제 검색 품질 평가 시에는 --eval-jsonl로 별도의
평가용 데이터를 지정해야 한다.

Usage:
    # self-retrieval 테스트 (인덱스 빌드 데이터와 동일)
    python scripts/evaluate_search.py \\
        --jsonl data/processed/v2_train.jsonl \\
        --index-dir models/faiss_index \\
        --samples-per-category 20 \\
        --top-k 10

    # 별도 평가 데이터 사용
    python scripts/evaluate_search.py \\
        --jsonl data/processed/v2_train.jsonl \\
        --eval-jsonl data/processed/v2_eval.jsonl \\
        --index-dir models/faiss_index
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_collection_preprocessing.embedding import EmbeddingPipeline
from src.inference.index_manager import IndexType, MultiIndexManager


# ---------------------------------------------------------------------------
# 평가 데이터셋 구성
# ---------------------------------------------------------------------------


def build_eval_dataset(
    records: List[Dict],
    samples_per_category: int,
    seed: int = 42,
) -> List[Dict]:
    """카테고리별 균등 샘플링으로 평가 데이터셋을 구성한다.

    Parameters
    ----------
    records : List[Dict]
        EmbeddingPipeline.load_jsonl()로 읽은 레코드 리스트.
    samples_per_category : int
        카테고리당 최대 샘플 수.
    seed : int
        난수 시드.

    Returns
    -------
    List[Dict]
        균등 샘플링된 평가 레코드 리스트.
    """
    rng = np.random.default_rng(seed)

    by_category: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        cat = record.get("category", "unknown") or "unknown"
        by_category[cat].append(record)

    eval_samples: List[Dict] = []
    for cat, cat_records in sorted(by_category.items()):
        n = min(samples_per_category, len(cat_records))
        indices = rng.choice(len(cat_records), size=n, replace=False)
        for idx in indices:
            eval_samples.append(cat_records[idx])

    logger.info(
        f"평가 데이터셋 구성 완료: {len(by_category)}개 카테고리, "
        f"총 {len(eval_samples)}개 샘플 (카테고리당 최대 {samples_per_category}개)"
    )
    for cat, cat_records in sorted(by_category.items()):
        n = min(samples_per_category, len(cat_records))
        logger.info(f"  - {cat}: {n}/{len(cat_records)}건 샘플링")

    return eval_samples


# ---------------------------------------------------------------------------
# 검색 평가
# ---------------------------------------------------------------------------


def evaluate_search(
    eval_samples: List[Dict],
    manager: MultiIndexManager,
    pipeline: EmbeddingPipeline,
    index_type: IndexType,
    top_k: int = 10,
) -> Dict:
    """검색 성능을 평가한다.

    Parameters
    ----------
    eval_samples : List[Dict]
        평가 샘플 리스트 (각 딕셔너리에 'id', 'complaint_text' 키 필요).
    manager : MultiIndexManager
        로드된 FAISS 인덱스 매니저.
    pipeline : EmbeddingPipeline
        쿼리 임베딩용 파이프라인.
    index_type : IndexType
        검색 대상 인덱스 타입.
    top_k : int
        최대 검색 결과 수.

    Returns
    -------
    Dict
        평가 결과 딕셔너리 (recall@5, recall@10, mrr, latency 등).
    """
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks: List[float] = []
    latencies: List[float] = []

    total = len(eval_samples)
    logger.info(f"검색 평가 시작: {total}개 샘플, top_k={top_k}")

    for i, sample in enumerate(eval_samples):
        query_text = sample["complaint_text"]
        expected_doc_id = sample["id"]

        # 쿼리 임베딩 (E5 모델: "query: " prefix 적용)
        t_start = time.perf_counter()
        query_vec = pipeline.embed_query(query_text)

        # FAISS 검색
        results = manager.search(index_type, query_vec, top_k=top_k)
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000
        latencies.append(latency_ms)

        # 결과에서 doc_id 추출
        result_doc_ids = [r["doc_id"] for r in results]

        # Recall@5 체크
        if expected_doc_id in result_doc_ids[:5]:
            hits_at_5 += 1

        # Recall@10 체크
        if expected_doc_id in result_doc_ids[:10]:
            hits_at_10 += 1

        # MRR 계산
        rank = None
        for j, doc_id in enumerate(result_doc_ids):
            if doc_id == expected_doc_id:
                rank = j + 1
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            logger.info(f"  진행: {i + 1}/{total}")

    # 지표 계산
    recall_at_5 = hits_at_5 / total if total > 0 else 0.0
    recall_at_10 = hits_at_10 / total if total > 0 else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    latency_arr = np.array(latencies)
    p50 = float(np.percentile(latency_arr, 50)) if latencies else 0.0
    p95 = float(np.percentile(latency_arr, 95)) if latencies else 0.0
    p99 = float(np.percentile(latency_arr, 99)) if latencies else 0.0

    return {
        "total_samples": total,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "mrr": mrr,
        "hits_at_5": hits_at_5,
        "hits_at_10": hits_at_10,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "latency_mean_ms": float(np.mean(latency_arr)) if latencies else 0.0,
    }


# ---------------------------------------------------------------------------
# 결과 출력
# ---------------------------------------------------------------------------


def print_results(results: Dict, top_k: int) -> None:
    """평가 결과를 보기 좋게 출력한다."""
    divider = "=" * 60
    print(f"\n{divider}")
    print("  FAISS 벡터 검색 시스템 평가 결과")
    print(divider)

    print(f"\n  평가 샘플 수:  {results['total_samples']}")
    print(f"  검색 Top-K:    {top_k}")

    print(f"\n{'─' * 60}")
    print("  [정량 지표]")
    print(f"{'─' * 60}")

    recall5 = results["recall_at_5"] * 100
    recall10 = results["recall_at_10"] * 100
    mrr = results["mrr"]

    status_5 = "PASS" if recall5 >= 80.0 else "FAIL"
    print(f"  Recall@5:   {recall5:6.2f}%  ({results['hits_at_5']}/{results['total_samples']})  [{status_5}]")
    print(f"  Recall@10:  {recall10:6.2f}%  ({results['hits_at_10']}/{results['total_samples']})")
    print(f"  MRR:        {mrr:6.4f}")

    print(f"\n{'─' * 60}")
    print("  [검색 레이턴시]")
    print(f"{'─' * 60}")
    print(f"  p50:   {results['latency_p50_ms']:8.2f} ms")
    print(f"  p95:   {results['latency_p95_ms']:8.2f} ms")
    print(f"  p99:   {results['latency_p99_ms']:8.2f} ms")
    print(f"  mean:  {results['latency_mean_ms']:8.2f} ms")

    print(f"\n{divider}")
    if recall5 >= 80.0:
        print("  >>> Recall@5 >= 80% 목표 달성!")
    else:
        print(f"  >>> Recall@5 목표 미달 (현재: {recall5:.2f}%, 목표: 80%)")
    print(f"{divider}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAISS 벡터 검색 시스템 Recall@K / MRR / 레이턴시 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="data/processed/v2_train.jsonl",
        help="평가에 사용할 JSONL 파일 경로 (기본: data/processed/v2_train.jsonl)",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="models/faiss_index",
        help="FAISS 인덱스 디렉토리 (기본: models/faiss_index)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="임베딩 모델 이름 (기본: intfloat/multilingual-e5-large)",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=20,
        help="카테고리당 평가 샘플 수 (기본: 20)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="검색 Top-K (기본: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="난수 시드 (기본: 42)",
    )
    parser.add_argument(
        "--eval-jsonl",
        type=str,
        default=None,
        help="별도 평가용 JSONL 파일 경로 (미지정 시 --jsonl과 동일 파일 사용, self-retrieval 테스트)",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="인덱스가 없으면 JSONL로부터 인덱스를 새로 구축",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. 임베딩 파이프라인 초기화
    logger.info("임베딩 파이프라인 초기화 중...")
    pipeline = EmbeddingPipeline(model_name=args.model_name)

    # 2. 평가용 JSONL 로드
    eval_jsonl = args.eval_jsonl or args.jsonl
    if args.eval_jsonl:
        logger.info(f"별도 평가 데이터 사용: {eval_jsonl}")
    else:
        logger.info(f"self-retrieval 테스트 (인덱스 빌드 데이터와 동일): {eval_jsonl}")
    records = pipeline.load_jsonl(eval_jsonl)
    if not records:
        logger.error("유효한 레코드가 없습니다. 종료합니다.")
        sys.exit(1)

    # 3. 인덱스 매니저 초기화 및 인덱스 로드/구축
    logger.info(f"인덱스 매니저 초기화: {args.index_dir}")
    manager = MultiIndexManager(base_dir=args.index_dir, embedding_dim=pipeline.embedding_dim)

    index_type = IndexType.CASE
    stats = manager.get_index_stats()
    case_stats = stats["indexes"].get(index_type.value, {})
    doc_count = case_stats.get("doc_count", 0)

    if doc_count == 0 and args.build_index:
        logger.info("인덱스가 비어있어 새로 구축합니다...")
        embeddings, metadata_list = pipeline.process_jsonl(args.jsonl)
        manager.add_documents(index_type, embeddings, metadata_list)
        manager.save_index(index_type)
        logger.info(f"인덱스 구축 완료: {manager.indexes[index_type].ntotal}건")
    elif doc_count == 0:
        logger.error(
            "인덱스가 비어있습니다. --build-index 옵션으로 인덱스를 먼저 구축하세요."
        )
        sys.exit(1)
    else:
        logger.info(f"기존 인덱스 로드 완료: {doc_count}건")

    # 4. 평가 데이터셋 구성
    eval_samples = build_eval_dataset(
        records,
        samples_per_category=args.samples_per_category,
        seed=args.seed,
    )

    # 5. 검색 평가 수행
    results = evaluate_search(
        eval_samples=eval_samples,
        manager=manager,
        pipeline=pipeline,
        index_type=index_type,
        top_k=args.top_k,
    )

    # 6. 결과 출력
    print_results(results, top_k=args.top_k)


if __name__ == "__main__":
    main()
