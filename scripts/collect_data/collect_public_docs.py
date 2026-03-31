"""
행정안전부 정부 공문서 AI 학습데이터 최종 수집 스크립트.

이슈 #157: 고품질 행정 공문서(보도자료, 보고서 등) 수집 및 인덱싱.
Base URL: apis.data.go.kr/1741000/publicDoc
상세기능: /getDocAll (5종 전체 조회)
"""

import argparse
import json
import os
import re
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from loguru import logger

BASE_URL = "https://apis.data.go.kr/1741000/publicDoc/getDocAll"
CHUNK_SIZE = 1024 * 1024  # 1MB
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


def _download_zip_to_tempfile(url: str, session: requests.Session) -> Optional[Path]:
    """ZIP URL을 청크 스트리밍으로 임시 파일에 저장. 실패 시 None 반환."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                tmp.write(chunk)
            tmp.close()
            return Path(tmp.name)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF ** attempt
                logger.warning(f"ZIP 다운로드 실패 ({url}), {wait:.0f}s 후 재시도: {e}")
                time.sleep(wait)
            else:
                logger.error(f"ZIP 다운로드 최종 실패 ({url}): {e}")
    return None


def _extract_items_from_zip(zip_path: Path) -> List[Dict]:
    """ZIP 파일에서 JSON 파일을 읽어 레코드 목록 반환. 완료 후 임시 파일 삭제."""
    items = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                with zf.open(name) as f:
                    try:
                        content = json.load(f)
                        if isinstance(content, list):
                            items.extend(content)
                        else:
                            items.append(content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 파싱 실패 ({name}): {e}")
    except zipfile.BadZipFile as e:
        logger.error(f"유효하지 않은 ZIP ({zip_path}): {e}")
    finally:
        zip_path.unlink(missing_ok=True)
    return items


def _fetch_zip_worker(args: Tuple[int, str, requests.Session]) -> List[Dict]:
    """스레드 워커: ZIP 다운로드 → 추출 → 임시파일 삭제."""
    idx, url, session = args
    logger.debug(f"[{idx}] ZIP 다운로드 시작: {url}")
    tmp_path = _download_zip_to_tempfile(url, session)
    if tmp_path is None:
        return []
    items = _extract_items_from_zip(tmp_path)
    logger.debug(f"[{idx}] 완료: {len(items)}개 레코드")
    return items


class PublicDocCollector:
    def __init__(self, api_key: str, max_workers: int = 8):
        self.api_key = api_key
        self.max_workers = max_workers
        self._session = requests.Session()
        self._session.headers.update({"Accept-Encoding": "gzip, deflate"})

    def _fetch_metadata(self, limit: int) -> List[Dict]:
        """API에서 메타데이터 목록(resultList) 수집."""
        params = {
            "serviceKey": self.api_key,
            "pageNo": 1,
            "numOfRows": limit,
            "type": "json",
        }
        try:
            resp = self._session.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            body = resp.json().get("response", {}).get("body", {})
            result_list = body.get("resultList", [])
            logger.info(f"메타데이터 {len(result_list)}건 수신")
            return result_list
        except Exception as e:
            logger.error(f"메타데이터 수집 실패: {e}")
            return []

    def fetch_all_docs(self, limit: int = 50) -> List[Dict]:
        """ZIP URL 목록을 병렬로 다운로드·추출하여 레코드 반환."""
        result_list = self._fetch_metadata(limit)
        if not result_list:
            return []

        zip_urls = [
            (i, res["url"])
            for i, res in enumerate(result_list)
            if res.get("url") and res.get("extType") == "zip"
        ]

        if not zip_urls:
            logger.warning("ZIP extType 항목이 없습니다.")
            return []

        logger.info(f"ZIP {len(zip_urls)}건 병렬 다운로드 시작 (workers={self.max_workers})")

        all_items: List[Dict] = []
        worker_args = [(idx, url, self._session) for idx, url in zip_urls]

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_fetch_zip_worker, arg): arg[0] for arg in worker_args}
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    items = future.result()
                    all_items.extend(items)
                except Exception as e:
                    logger.error(f"워커 예외: {e}")
                if done % 10 == 0 or done == len(zip_urls):
                    logger.info(f"진행: {done}/{len(zip_urls)} ZIP 완료, 누적 {len(all_items)}건")

        logger.info(f"전체 수집 완료: {len(all_items)}개 레코드")
        return all_items

    def process_item(self, item: Dict) -> List[Dict]:
        """실제 행안부 JSON 구조(meta, data.text)에 맞춰 데이터를 변환한다."""
        results = []

        meta = item.get("meta", {})
        data = item.get("data", {})

        title = meta.get("title") or "정부 공문서"
        content = data.get("text") or ""

        if not content:
            return []

        # 이미지 태그 제거, 표(table) 구조 보존
        clean_content = re.sub(r"<img[^>]*>", "", content)
        allowed_tags = ["table", "thead", "tbody", "tr", "th", "td"]
        tag_pattern = re.compile(
            r"<(/?)" r"(?!(" + "|".join(allowed_tags) + r")\b)[^>]+>", re.IGNORECASE
        )
        clean_content = tag_pattern.sub("", clean_content)
        clean_content = re.sub(r"[ \t]+", " ", clean_content).strip()

        summary = item.get("summary") or data.get("summary") or meta.get("summary")

        if summary:
            results.append({
                "instruction": "다음 정부 공문서 내용을 바탕으로 핵심 내용을 요약하고 공공기관 보고서 형식(개조식)으로 재작성하세요.",
                "input": f"제목: {title}\n본문: {clean_content}",
                "output": summary,
            })
        else:
            results.append({
                "instruction": "다음은 정부에서 발행한 공문서 본문입니다. 이 문서의 내용을 한 문장으로 요약하고 적절한 제목을 제안하세요.",
                "input": f"본문: {clean_content[:2000]}",
                "output": f"제목: {title}\n요약: 이 문서는 {title}에 관한 내용을 담고 있습니다.",
            })

        qna = item.get("qna_data") or data.get("qna_data") or item.get("qna")
        if isinstance(qna, list):
            for qa in qna:
                q = qa.get("question") or qa.get("q")
                a = qa.get("answer") or qa.get("a")
                if q and a:
                    results.append({
                        "instruction": f"제공된 공문서 '{title}'의 내용을 바탕으로 다음 질문에 답하세요.",
                        "input": f"질문: {q}",
                        "output": a,
                    })

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/raw/manuals/public_docs.jsonl")
    parser.add_argument("--workers", type=int, default=8, help="병렬 다운로드 스레드 수")
    args = parser.parse_args()

    api_key = os.getenv("DATA_GO_KR_API_KEY")
    if not api_key:
        logger.error("DATA_GO_KR_API_KEY 환경변수가 없습니다.")
        return

    collector = PublicDocCollector(api_key, max_workers=args.workers)
    raw_docs = collector.fetch_all_docs(limit=args.limit)

    if not raw_docs:
        logger.warning("수집된 데이터가 없습니다. API 승인 및 키 상태를 확인하세요.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for d in raw_docs:
            for item in collector.process_item(d):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_records += 1

    logger.info(
        f"수집 및 변환 완료: {len(raw_docs)}개 문서 → {total_records}개 레코드 → {output_path}"
    )


if __name__ == "__main__":
    main()
