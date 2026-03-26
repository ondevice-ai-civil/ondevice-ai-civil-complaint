"""
임베딩 파이프라인: intfloat/multilingual-e5-large 기반 JSONL 데이터 임베딩 변환.

JSONL 데이터를 읽어 민원 텍스트를 파싱하고, E5 모델로 임베딩 벡터를 생성한다.
passage: prefix를 적용하여 cosine similarity 검색에 최적화된 정규화 벡터를 반환한다.
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.inference.index_manager import DocumentMetadata


class EmbeddingPipeline:
    """intfloat/multilingual-e5-large 기반 임베딩 파이프라인.

    Parameters
    ----------
    model_name : str
        SentenceTransformer 모델 이름. 기본: intfloat/multilingual-e5-large (1024차원).
    device : str, optional
        추론에 사용할 디바이스. None이면 자동 감지.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        logger.info(f"SentenceTransformer 모델 로딩 시작: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"모델 로딩 완료: dim={self.embedding_dim}, " f"device={self.model.device}")

    # ------------------------------------------------------------------
    # 텍스트 파싱
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_complaint_text(text: str) -> Optional[str]:
        """[|user|] ... [|endofturn|] 구간에서 '민원 내용:' 이후 텍스트를 추출한다."""
        user_match = re.search(r"\[\|user\|\](.*?)\[\|endofturn\|\]", text, re.DOTALL)
        if not user_match:
            return None

        user_block = user_match.group(1)
        complaint_match = re.search(r"민원\s*내용\s*:\s*(.+)", user_block, re.DOTALL)
        if complaint_match:
            return complaint_match.group(1).strip()

        return None

    @staticmethod
    def _parse_answer_text(text: str) -> Optional[str]:
        """[|assistant|] ... [|endofturn|] 구간에서 답변 텍스트를 추출한다."""
        assistant_match = re.search(r"\[\|assistant\|\](.*?)\[\|endofturn\|\]", text, re.DOTALL)
        if not assistant_match:
            return None
        return assistant_match.group(1).strip()

    @staticmethod
    def _parse_category_from_text(text: str) -> Optional[str]:
        """[카테고리: ...] 패턴에서 카테고리를 추출한다."""
        cat_match = re.search(r"\[카테고리:\s*(.+?)\]", text)
        if cat_match:
            return cat_match.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # JSONL 로드
    # ------------------------------------------------------------------

    def load_jsonl(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """JSONL 파일을 로드하고 민원/답변 텍스트를 파싱한다.

        Parameters
        ----------
        jsonl_path : str
            JSONL 파일 경로.

        Returns
        -------
        List[Dict[str, Any]]
            파싱된 레코드 리스트. 각 딕셔너리에 'complaint_text', 'answer_text',
            'category', 'id' 키가 포함된다.
        """
        records: List[Dict[str, Any]] = []
        skipped = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 실패 (line {line_num}): {e}")
                    skipped += 1
                    continue

                text = data.get("text", "")
                complaint_text = self._parse_complaint_text(text)
                answer_text = self._parse_answer_text(text)

                if not complaint_text:
                    logger.warning(
                        f"민원 텍스트 파싱 실패 (line {line_num}, "
                        f"id={data.get('id', 'unknown')})"
                    )
                    skipped += 1
                    continue

                # 카테고리: JSONL의 category 필드 우선, 없으면 텍스트에서 파싱
                category = data.get("category") or self._parse_category_from_text(text) or ""

                records.append(
                    {
                        "id": data.get("id", f"unknown_{line_num}"),
                        "complaint_text": complaint_text,
                        "answer_text": answer_text or "",
                        "category": category,
                    }
                )

        logger.info(
            f"JSONL 로드 완료: 총 {len(records) + skipped}건 중 "
            f"{len(records)}건 성공, {skipped}건 스킵"
        )
        return records

    # ------------------------------------------------------------------
    # 임베딩 생성
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """텍스트 리스트를 임베딩 벡터로 변환한다.

        E5 모델 요구사항에 따라 'passage: ' prefix를 적용한다.

        Parameters
        ----------
        texts : List[str]
            임베딩할 텍스트 리스트.
        batch_size : int
            배치 크기.

        Returns
        -------
        np.ndarray
            정규화된 임베딩 벡터 배열 (shape: ``(n, dim)``).
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # E5 모델은 passage: prefix를 요구한다
        prefixed_texts = [f"passage: {t}" for t in texts]

        logger.info(f"임베딩 생성 시작: {len(texts)}건, batch_size={batch_size}")

        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        logger.info(f"임베딩 생성 완료: shape={embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """쿼리 텍스트를 임베딩 벡터로 변환한다. E5 모델의 'query: ' prefix 적용."""
        embedding = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype=np.float32)

    # ------------------------------------------------------------------
    # 전체 파이프라인
    # ------------------------------------------------------------------

    def process_jsonl(
        self,
        jsonl_path: str,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, List[DocumentMetadata]]:
        """JSONL 파일을 읽어 임베딩 벡터와 DocumentMetadata 리스트로 변환한다.

        Parameters
        ----------
        jsonl_path : str
            JSONL 파일 경로.
        batch_size : int
            임베딩 배치 크기.

        Returns
        -------
        Tuple[np.ndarray, List[DocumentMetadata]]
            (임베딩 벡터 배열, DocumentMetadata 리스트).
        """
        records = self.load_jsonl(jsonl_path)
        if not records:
            raise ValueError(f"유효한 레코드가 없습니다: {jsonl_path}")

        # 민원 텍스트로 임베딩 생성
        texts = [r["complaint_text"] for r in records]
        embeddings = self.embed_documents(texts, batch_size=batch_size)

        # DocumentMetadata 생성
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata_list: List[DocumentMetadata] = []

        for record in records:
            meta = DocumentMetadata(
                doc_id=record["id"],
                doc_type="case",
                source="AI Hub",
                title=f"[{record['category']}] 민원 사례",
                category=record["category"],
                reliability_score=0.6,
                created_at=now_iso,
                updated_at=now_iso,
                chunk_index=0,
                chunk_total=1,
                extras={
                    "complaint_text": record["complaint_text"],
                    "answer_text": record["answer_text"],
                },
            )
            metadata_list.append(meta)

        logger.info(
            f"파이프라인 완료: {len(metadata_list)}건 처리, " f"벡터 shape={embeddings.shape}"
        )
        return embeddings, metadata_list
