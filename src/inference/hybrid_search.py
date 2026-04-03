"""
HybridSearchEngine: Dense(FAISS) + Sparse(BM25) 하이브리드 검색 엔진.

ADR-004 기반 Reciprocal Rank Fusion(RRF) 가중치를 사용하여
데이터 타입별 최적 검색 결과를 제공한다.

Issue: #154
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.inference.bm25_indexer import BM25Indexer
from src.inference.index_manager import IndexType, MultiIndexManager


class SearchMode(str, Enum):
    """검색 모드."""

    DENSE = "dense"  # FAISS 벡터 검색만 사용
    SPARSE = "sparse"  # BM25 키워드 검색만 사용
    HYBRID = "hybrid"  # Dense + Sparse RRF 융합


@dataclass
class RRFWeightConfig:
    """Reciprocal Rank Fusion 가중치 설정.

    데이터 타입별로 Dense/Sparse 검색의 상대적 중요도를 조절한다.
    """

    dense_weight: float = 1.0
    sparse_weight: float = 0.7


# ADR-004 기반 데이터 타입별 기본 RRF 가중치
DEFAULT_RRF_WEIGHTS: Dict[IndexType, RRFWeightConfig] = {
    IndexType.CASE: RRFWeightConfig(dense_weight=1.0, sparse_weight=0.7),
    IndexType.LAW: RRFWeightConfig(dense_weight=0.9, sparse_weight=1.2),
    IndexType.MANUAL: RRFWeightConfig(dense_weight=0.8, sparse_weight=0.8),
    IndexType.NOTICE: RRFWeightConfig(dense_weight=0.6, sparse_weight=0.6),
}


class HybridSearchEngine:
    """Dense + Sparse 하이브리드 검색 엔진.

    FAISS 기반 벡터 검색(Dense)과 BM25 키워드 검색(Sparse)을 결합하여
    Reciprocal Rank Fusion으로 최종 랭킹을 생성한다.

    Parameters
    ----------
    index_manager : MultiIndexManager
        FAISS 인덱스 관리자.
    bm25_indexers : Dict[IndexType, BM25Indexer]
        데이터 타입별 BM25 인덱서.
    embed_model
        SentenceTransformer 임베딩 모델.
    rrf_k : int
        RRF smoothing 파라미터 (기본값: 60).
    rrf_weights : Optional[Dict[IndexType, RRFWeightConfig]]
        데이터 타입별 RRF 가중치. None이면 DEFAULT_RRF_WEIGHTS 사용.
    """

    def __init__(
        self,
        index_manager: MultiIndexManager,
        bm25_indexers: Dict[IndexType, BM25Indexer],
        embed_model: Any,
        rrf_k: int = 60,
        rrf_weights: Optional[Dict[IndexType, RRFWeightConfig]] = None,
    ) -> None:
        if index_manager is None:
            raise RuntimeError("index_manager는 None일 수 없습니다.")

        self.index_manager = index_manager
        self.bm25_indexers = bm25_indexers or {}
        self.embed_model = embed_model
        if rrf_k < 1:
            raise ValueError(f"rrf_k는 1 이상이어야 합니다. (입력값: {rrf_k})")
        self.rrf_k = rrf_k
        self.rrf_weights = rrf_weights or DEFAULT_RRF_WEIGHTS

    async def search(
        self,
        query: str,
        index_type: IndexType,
        top_k: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> Tuple[List[Dict[str, Any]], SearchMode]:
        """하이브리드 검색을 수행한다.

        Parameters
        ----------
        query : str
            검색 쿼리 문자열.
        index_type : IndexType
            검색 대상 인덱스 타입.
        top_k : int
            반환할 최대 결과 수.
        mode : SearchMode
            검색 모드 (dense / sparse / hybrid).

        Returns
        -------
        Tuple[List[Dict[str, Any]], SearchMode]
            score(0~1 정규화)를 포함한 검색 결과 리스트와 실제 사용된 검색 모드.
            BM25 미준비 등으로 폴백이 발생하면 actual_mode가 요청 mode와 다를 수 있다.
        """
        if not query or not query.strip():
            return [], mode
        if top_k <= 0:
            return [], mode

        try:
            if mode == SearchMode.DENSE:
                if self.embed_model is None:
                    raise RuntimeError("Dense 검색에는 embed_model이 필요합니다.")
                query_vector = self._embed_query(query)
                results = await self._dense_search(query_vector, index_type, top_k)
                return results[:top_k], SearchMode.DENSE

            if mode == SearchMode.SPARSE:
                results = await self._sparse_search(query, index_type, top_k)
                return results[:top_k], SearchMode.SPARSE

            # HYBRID 모드
            if self.embed_model is None:
                raise RuntimeError("Hybrid 검색에는 embed_model이 필요합니다.")

            bm25_indexer = self.bm25_indexers.get(index_type)
            if bm25_indexer is None or not bm25_indexer.is_ready():
                logger.warning(
                    f"BM25 인덱스 미사용 가능 (type={index_type.value}). "
                    "Dense 검색으로 폴백합니다."
                )
                query_vector = self._embed_query(query)
                results = await self._dense_search(query_vector, index_type, top_k)
                return results[:top_k], SearchMode.DENSE

            query_vector = self._embed_query(query)
            dense_result, sparse_result = await asyncio.gather(
                self._dense_search(query_vector, index_type, top_k),
                self._sparse_search(query, index_type, top_k),
                return_exceptions=True,
            )

            dense_failed = isinstance(dense_result, BaseException)
            sparse_failed = isinstance(sparse_result, BaseException)

            if dense_failed and sparse_failed:
                logger.error(
                    f"Hybrid 검색 양측 모두 실패 (type={index_type.value}): "
                    f"dense={dense_result}, sparse={sparse_result}"
                )
                raise RuntimeError(
                    f"Dense 및 Sparse 검색 모두 실패: "
                    f"dense={dense_result}, sparse={sparse_result}"
                )

            if dense_failed:
                logger.warning(
                    f"Dense 검색 실패, Sparse 결과로 대체합니다 "
                    f"(type={index_type.value}): {dense_result}"
                )
                dense_results: List[Dict[str, Any]] = []
                sparse_results: List[Dict[str, Any]] = sparse_result
                actual_mode = SearchMode.SPARSE
            elif sparse_failed:
                logger.warning(
                    f"Sparse 검색 실패, Dense 결과로 대체합니다 "
                    f"(type={index_type.value}): {sparse_result}"
                )
                dense_results = dense_result
                sparse_results = []
                actual_mode = SearchMode.DENSE
            else:
                dense_results = dense_result
                sparse_results = sparse_result
                actual_mode = SearchMode.HYBRID

            fused = self._reciprocal_rank_fusion(dense_results, sparse_results, index_type)
            return fused[:top_k], actual_mode

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"검색 중 오류 발생 (type={index_type.value}, mode={mode.value}): {e}")
            return [], mode

    async def _dense_search(
        self,
        query_vector: np.ndarray,
        index_type: IndexType,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """FAISS 벡터 검색을 비동기로 수행한다.

        동기 FAISS 호출을 executor에서 실행하여 이벤트 루프를 차단하지 않는다.
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, self.index_manager.search, index_type, query_vector, top_k
        )
        return results

    async def _sparse_search(
        self,
        query: str,
        index_type: IndexType,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """BM25 키워드 검색을 수행하고 메타데이터를 매핑한다.

        BM25Indexer.search()가 반환하는 corpus_index를
        index_manager.metadata로 매핑하여 통일된 결과 형식을 제공한다.
        """
        indexer = self.bm25_indexers.get(index_type)
        if indexer is None or not indexer.is_ready():
            return []

        try:
            loop = asyncio.get_running_loop()
            bm25_results = await loop.run_in_executor(None, indexer.search, query, top_k)
        except Exception as e:
            logger.error(f"BM25 검색 오류 (type={index_type.value}): {e}")
            return []

        meta_list = self.index_manager.metadata.get(index_type, [])
        results: List[Dict[str, Any]] = []

        for corpus_index, score in bm25_results:
            if corpus_index < len(meta_list):
                item = meta_list[corpus_index].to_dict()
                item["score"] = float(score)
                results.append(item)
            else:
                logger.warning(
                    f"BM25 corpus_index 범위 초과: idx={corpus_index}, "
                    f"meta_len={len(meta_list)}"
                )

        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        index_type: IndexType,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion으로 Dense/Sparse 결과를 병합한다.

        RRF 공식: score(d) = sum(w_i / (k + rank_i(d)))
        - doc_id를 기준으로 결과를 병합한다.
        - 최종 score를 max RRF score로 나누어 0~1 정규화한다.
        """
        weight_config = self.rrf_weights.get(index_type, RRFWeightConfig())

        # doc_id -> {metadata dict, rrf_score}
        doc_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict[str, Any]] = {}

        # Dense 결과 RRF 점수 계산 (1-based rank)
        for rank, result in enumerate(dense_results, start=1):
            doc_id = result.get("doc_id", "")
            if not doc_id:
                logger.warning("doc_id가 누락된 검색 결과를 RRF 병합에서 제외합니다")
                continue
            rrf_score = weight_config.dense_weight / (self.rrf_k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = result.copy()

        # Sparse 결과 RRF 점수 계산 (1-based rank)
        for rank, result in enumerate(sparse_results, start=1):
            doc_id = result.get("doc_id", "")
            if not doc_id:
                logger.warning("doc_id가 누락된 검색 결과를 RRF 병합에서 제외합니다")
                continue
            rrf_score = weight_config.sparse_weight / (self.rrf_k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = result.copy()

        if not doc_scores:
            return []

        # RRF 점수 기준 내림차순 정렬
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)

        # 최대 점수로 0~1 정규화
        max_score = doc_scores[sorted_doc_ids[0]]
        if max_score <= 0:
            return []

        results: List[Dict[str, Any]] = []
        for doc_id in sorted_doc_ids:
            item = doc_data[doc_id]
            item["score"] = doc_scores[doc_id] / max_score
            results.append(item)

        return results

    def _embed_query(self, query: str) -> np.ndarray:
        """쿼리를 임베딩 벡터로 변환한다.

        multilingual-e5-large 모델 규칙에 따라 'query:' 접두사를 추가하고
        L2 정규화된 벡터를 반환한다.
        """
        embedding = self.embed_model.encode([f"query: {query}"], normalize_embeddings=True)
        return embedding[0]

    def is_ready(self, index_type: IndexType) -> bool:
        """지정된 인덱스 타입이 검색 가능한 상태인지 확인한다.

        index_manager에 해당 인덱스가 로드되어 있고
        문서가 1개 이상 존재해야 True를 반환한다.
        """
        if index_type not in self.index_manager.indexes:
            return False
        index = self.index_manager.indexes[index_type]
        return index.ntotal > 0
