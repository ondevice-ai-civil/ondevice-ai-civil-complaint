import requests
import json
import os
from loguru import logger

def collect_saha_notices():
    """행안부 API를 통한 부산 사하구 고시공고 실데이터 수집"""
    key = "e8cd6e25666c8391c17658e557e65e526051027e52a060548e0d5879a9fe5fc4"
    logger.info("사하구청 고시공고 데이터 수집 시작...")
    
    # 행안부 지방자치단체 고시공고 정보 API 엔드포인트
    url = "http://apis.data.go.kr/1741000/GosiGonggoService/getGosiGonggoList"
    params = {
        "serviceKey": key,
        "pageNo": 1,
        "numOfRows": 50,
        "type": "json",
        "inst_nm": "부산광역시 사하구"
    }
    
    try:
        res = requests.get(url, params=params, timeout=15)
        data = res.json()
        items = data.get("GosiGonggo", [{}, {"row": []}])[1].get("row", [])
        
        if items:
            output_path = "data/raw/notices/saha_notices.jsonl"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for it in items:
                    item = {
                        "doc_id": f"NOTICE_SAHA_{it.get('gosi_id')}",
                        "doc_type": "notice",
                        "source": "사하구청 (행안부 API)",
                        "title": it.get("gosi_title"),
                        "content": f"제목: {it.get('gosi_title')}\n공고기관: {it.get('inst_nm')}\n날짜: {it.get('gosi_date')}\n내용: {it.get('gosi_cont', '')}",
                        "category": "행정공고"
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"✅ 사하구 고시공고 수집 완료: {len(items)}건")
            return True
    except Exception as e:
        logger.error(f"사하구 고시공고 수집 실패: {e}")
    return False

if __name__ == "__main__":
    os.makedirs("data/raw/notices", exist_ok=True)
    collect_saha_notices()
