"""
MultiIndexManager: ADR-004 기반 다중 FAISS 인덱스 관리 모듈.

데이터 타입별(유사사례, 법령, 매뉴얼, 공시정보) 독립 FAISS 인덱스를 운영하며,
문서 수 기반 IndexFlatIP -> IndexIVFFlat 자동 전환을 지원한다.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from loguru import logger


class IndexType(str, Enum):
    """RAG 검색 대상 데이터 타입."""

    CASE = "case"       # 유사 민원 사례
    LAW = "law"         # 법령/규정
    MANUAL = "manual"   # 업무 매뉴얼
    NOTICE = "notice"   # 기관 공시 정보


@dataclass
class DocumentMetadata:
    """ADR-004 Section B.1 메타데이터 스키마.

    모든 데이터 타입에 공통으로 적용되는 메타데이터 구조.
    타입별 추가 필드(CASE: complaint_text/answer_text, LAW: law_number 등)는
    extras dict로 처리한다.
    """

    doc_id: str
    doc_type: str               # IndexType.value
    source: str                 # 출처 (예: "AI Hub", "법제처", "기관 내부")
    title: str
    category: str               # 민원 카테고리 (도로/교통, 환경/위생 등)
    reliability_score: float    # 신뢰도 (0.0 ~ 1.0)
    created_at: str             # ISO 8601 문자열
    updated_at: str             # ISO 8601 문자열
    valid_from: Optional[str] = None   # 유효 시작일 (법령 시행일)
    valid_until: Optional[str] = None  # 유효 종료일 (폐지/개정 시)
    chunk_index: int = 0        # 청크 인덱스 (긴 문서 분할 시)
    chunk_total: int = 1        # 전체 청크 수
    extras: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """직렬화용 딕셔너리 변환."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """딕셔너리에서 DocumentMetadata 인스턴스를 생성한다."""
        if not hasattr(cls, "_known_fields"):
            cls._known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in cls._known_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# IVFFlat 자동 전환 상수
# ---------------------------------------------------------------------------
_IVF_THRESHOLD = 100_000   # 문서 수 >= 10만건이면 IVFFlat 전환
_IVF_NLIST = 256           # IVFFlat 클러스터 수
_IVF_NPROBE = 16           # 검색 시 탐색할 클러스터 수


class MultiIndexManager:
    """데이터 타입별 독립 FAISS 인덱스를 관리한다.

    - 인덱스 기본 경로: ``models/faiss_index/``
    - 기본 임베딩 차원: 1024 (multilingual-e5-large)
    - 정규화된 벡터 사용 (Inner Product = cosine similarity)
    - 문서 수 >= 10만건 시 IndexFlatIP -> IndexIVFFlat 자동 전환

    디렉토리 구조::

        models/faiss_index/
        +-- case/
        |   +-- index.faiss
        |   +-- metadata.json
        +-- law/
        +-- manual/
        +-- notice/
        +-- index_registry.json
    """

    def __init__(self, base_dir: str, embedding_dim: int = 1024) -> None:
        self.base_dir: str = base_dir
        self.embedding_dim: int = embedding_dim
        self.indexes: Dict[IndexType, faiss.Index] = {}
        self.metadata: Dict[IndexType, List[DocumentMetadata]] = {}

        os.makedirs(self.base_dir, exist_ok=True)

        # 레지스트리 로드 또는 초기화
        self._registry_path: str = os.path.join(self.base_dir, "index_registry.json")
        self._registry: Dict[str, Any] = self._load_registry()

        # 레지스트리에 기록된 인덱스가 있으면 자동 로드
        for index_type in IndexType:
            if index_type.value in self._registry.get("indexes", {}):
                try:
                    self.load_index(index_type)
                except Exception as e:
                    logger.warning(
                        f"인덱스 자동 로드 실패 ({index_type.value}): {e}"
                    )

    # ------------------------------------------------------------------
    # 레지스트리 관리
    # ------------------------------------------------------------------

    def _load_registry(self) -> Dict[str, Any]:
        """index_registry.json 을 로드한다. 파일이 없으면 빈 구조를 반환."""
        if os.path.exists(self._registry_path):
            try:
                with open(self._registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"레지스트리 로드 실패, 초기화합니다: {e}")
        return {"indexes": {}, "updated_at": None}

    def _save_registry(self) -> None:
        """index_registry.json 을 갱신한다."""
        self._registry["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self._registry_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 인덱스 생성 / 로드 / 저장
    # ------------------------------------------------------------------

    def _create_index(self, index_type: IndexType) -> faiss.Index:
        """새 IndexFlatIP 인덱스를 생성한다."""
        index = faiss.IndexFlatIP(self.embedding_dim)
        logger.info(
            f"새 IndexFlatIP 생성 (type={index_type.value}, dim={self.embedding_dim})"
        )
        return index

    def load_index(self, index_type: IndexType) -> faiss.Index:
        """FAISS 인덱스 + 메타데이터 JSON을 로드한다.

        인덱스 파일이 존재하지 않으면 새 빈 인덱스를 생성한다.
        """
        index_dir = os.path.join(self.base_dir, index_type.value)
        index_path = os.path.join(index_dir, "index.faiss")
        meta_path = os.path.join(index_dir, "metadata.json")

        if os.path.exists(index_path):
            self.indexes[index_type] = faiss.read_index(index_path)
            logger.info(
                f"FAISS 인덱스 로드 완료: {index_path} "
                f"(벡터 수: {self.indexes[index_type].ntotal})"
            )
        else:
            logger.info(
                f"인덱스 파일 없음, 새 인덱스 생성: {index_type.value}"
            )
            self.indexes[index_type] = self._create_index(index_type)

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
            self.metadata[index_type] = [
                DocumentMetadata.from_dict(item) for item in raw_list
            ]
            logger.info(
                f"메타데이터 로드 완료: {meta_path} "
                f"(문서 수: {len(self.metadata[index_type])})"
            )
        else:
            self.metadata[index_type] = []

        return self.indexes[index_type]

    def save_index(self, index_type: IndexType) -> None:
        """FAISS 인덱스 + 메타데이터 JSON을 저장하고 레지스트리를 갱신한다."""
        if index_type not in self.indexes:
            raise ValueError(f"인덱스가 로드되지 않았습니다: {index_type.value}")

        index_dir = os.path.join(self.base_dir, index_type.value)
        os.makedirs(index_dir, exist_ok=True)

        index_path = os.path.join(index_dir, "index.faiss")
        meta_path = os.path.join(index_dir, "metadata.json")

        faiss.write_index(self.indexes[index_type], index_path)
        logger.info(f"FAISS 인덱스 저장: {index_path}")

        meta_list = [m.to_dict() for m in self.metadata.get(index_type, [])]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, ensure_ascii=False, indent=2)
        logger.info(f"메타데이터 저장: {meta_path}")

        # 레지스트리 갱신
        now_iso = datetime.now(timezone.utc).isoformat()
        self._registry.setdefault("indexes", {})[index_type.value] = {
            "doc_count": self.indexes[index_type].ntotal,
            "index_class": type(self.indexes[index_type]).__name__,
            "embedding_dim": self.embedding_dim,
            "last_updated": now_iso,
        }
        self._save_registry()

    # ------------------------------------------------------------------
    # 검색
    # ------------------------------------------------------------------

    def search(
        self,
        index_type: IndexType,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Top-K 벡터 검색을 수행하고 메타데이터가 포함된 결과를 반환한다.

        Parameters
        ----------
        index_type : IndexType
            검색 대상 인덱스 타입.
        query_vector : np.ndarray
            정규화된 쿼리 벡터 (shape: ``(dim,)`` 또는 ``(1, dim)``).
        top_k : int
            반환할 최대 결과 수.

        Returns
        -------
        List[Dict[str, Any]]
            ``score`` 와 메타데이터 필드를 포함하는 딕셔너리 리스트.
        """
        if index_type not in self.indexes:
            # 아직 로드되지 않은 인덱스이면 로드 시도
            self.load_index(index_type)

        index = self.indexes[index_type]
        if index.ntotal == 0:
            logger.warning(f"인덱스가 비어 있습니다: {index_type.value}")
            return []

        # 쿼리 벡터 shape 보정
        qv = np.asarray(query_vector, dtype=np.float32)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)

        # IVFFlat 인덱스인 경우 nprobe 설정
        if hasattr(index, "nprobe"):
            index.nprobe = _IVF_NPROBE

        actual_k = min(top_k, index.ntotal)
        distances, indices = index.search(qv, actual_k)

        meta_list = self.metadata.get(index_type, [])
        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if idx < len(meta_list):
                item = meta_list[idx].to_dict()
                item["score"] = float(dist)
                results.append(item)
            else:
                logger.warning(
                    f"메타데이터 인덱스 범위 초과: idx={idx}, "
                    f"meta_len={len(meta_list)}"
                )

        return results

    # ------------------------------------------------------------------
    # 문서 추가
    # ------------------------------------------------------------------

    def add_documents(
        self,
        index_type: IndexType,
        vectors: np.ndarray,
        metadata: List[DocumentMetadata],
    ) -> None:
        """벡터와 메타데이터를 인덱스에 추가한다.

        추가 후 문서 수가 임계값(10만건)을 넘으면 IVFFlat 전환을 시도한다.

        Parameters
        ----------
        vectors : np.ndarray
            정규화된 벡터 배열 (shape: ``(n, dim)``).
        metadata : List[DocumentMetadata]
            각 벡터에 대응하는 메타데이터 리스트.
        """
        if vectors.shape[0] != len(metadata):
            raise ValueError(
                f"벡터 수({vectors.shape[0]})와 "
                f"메타데이터 수({len(metadata)})가 일치하지 않습니다."
            )
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"벡터 차원({vectors.shape[1]})이 "
                f"설정 차원({self.embedding_dim})과 일치하지 않습니다."
            )

        # 인덱스가 아직 없으면 생성
        if index_type not in self.indexes:
            self.indexes[index_type] = self._create_index(index_type)
            self.metadata[index_type] = []

        vecs = np.asarray(vectors, dtype=np.float32)

        # IVFFlat 인덱스에 추가할 때는 train 여부 확인
        index = self.indexes[index_type]
        if hasattr(index, "is_trained") and not index.is_trained:
            logger.warning(
                f"IVFFlat 인덱스가 학습되지 않았습니다. "
                f"Flat 인덱스로 폴백합니다: {index_type.value}"
            )
            self.indexes[index_type] = self._create_index(index_type)
            index = self.indexes[index_type]

        index.add(vecs)
        self.metadata[index_type].extend(metadata)

        logger.info(
            f"문서 추가 완료: type={index_type.value}, "
            f"추가={len(metadata)}, 총={index.ntotal}"
        )

        # IVFFlat 자동 전환 체크
        self._maybe_upgrade_to_ivf(index_type)

    # ------------------------------------------------------------------
    # IVFFlat 자동 전환
    # ------------------------------------------------------------------

    def _maybe_upgrade_to_ivf(self, index_type: IndexType) -> None:
        """문서 수가 임계값 이상이면 IndexFlatIP -> IndexIVFFlat 으로 전환한다.

        - 전환 기준: ntotal >= 100,000
        - nlist = 256, nprobe = 16
        - 이미 IVFFlat 이면 스킵
        - 기존 벡터를 모두 보존한다.
        """
        index = self.indexes.get(index_type)
        if index is None:
            return

        # 이미 IVF 계열이면 스킵
        if isinstance(index, faiss.IndexIVFFlat):
            return

        if index.ntotal < _IVF_THRESHOLD:
            return

        logger.info(
            f"IVFFlat 전환 시작: type={index_type.value}, "
            f"문서 수={index.ntotal}, nlist={_IVF_NLIST}"
        )

        # 기존 벡터 추출
        n = index.ntotal
        vectors = faiss.rev_swig_ptr(
            index.get_xb(), n * self.embedding_dim
        ).reshape(n, self.embedding_dim).copy()

        # IVFFlat 인덱스 생성
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        ivf_index = faiss.IndexIVFFlat(
            quantizer, self.embedding_dim, _IVF_NLIST, faiss.METRIC_INNER_PRODUCT
        )
        ivf_index.nprobe = _IVF_NPROBE

        # 학습 및 벡터 재추가
        logger.info(f"IVFFlat 학습 중 (벡터 수: {n})...")
        ivf_index.train(vectors)
        ivf_index.add(vectors)

        self.indexes[index_type] = ivf_index
        logger.info(
            f"IVFFlat 전환 완료: type={index_type.value}, "
            f"ntotal={ivf_index.ntotal}"
        )

    # ------------------------------------------------------------------
    # 통계
    # ------------------------------------------------------------------

    def get_index_stats(self) -> Dict[str, Any]:
        """각 인덱스의 문서 수, 마지막 갱신 등 통계를 반환한다."""
        stats: Dict[str, Any] = {
            "base_dir": self.base_dir,
            "embedding_dim": self.embedding_dim,
            "indexes": {},
        }

        registry_indexes = self._registry.get("indexes", {})

        for index_type in IndexType:
            entry: Dict[str, Any] = {}
            if index_type in self.indexes:
                idx = self.indexes[index_type]
                entry["loaded"] = True
                entry["doc_count"] = idx.ntotal
                entry["index_class"] = type(idx).__name__
                entry["metadata_count"] = len(
                    self.metadata.get(index_type, [])
                )
            else:
                entry["loaded"] = False
                entry["doc_count"] = 0

            # 레지스트리 정보 병합
            if index_type.value in registry_indexes:
                reg = registry_indexes[index_type.value]
                entry["last_updated"] = reg.get("last_updated")
            else:
                entry["last_updated"] = None

            stats["indexes"][index_type.value] = entry

        return stats
