import requests
import urllib.parse
from loguru import logger

def test_public_doc_variants():
    # 제공된 키
    raw_key = "e8cd6e25666c8391c17658e557e65e526051027e52a060548e0d5879a9fe5fc4"
    # 수동 인코딩된 키 (필요한 경우)
    encoded_key = urllib.parse.quote(raw_key)
    
    base_url = "https://apis.data.go.kr/1741000/publicDoc/getDocAll"
    
    variants = [
        ("Decoding Key (Raw)", raw_key),
        ("Manual Encoded Key", encoded_key)
    ]
    
    for label, key in variants:
        logger.info(f"--- 테스트 시작: {label} ---")
        params = {
            "serviceKey": key,
            "pageNo": 1,
            "numOfRows": 1,
            "type": "json"
        }
        try:
            # params를 사용하면 requests가 키를 다시 인코딩하므로, 
            # 만약 이미 인코딩된 키를 보낼 때는 URL에 직접 붙여야 함
            url = f"{base_url}?serviceKey={key}&pageNo=1&numOfRows=1&type=json"
            res = requests.get(url, timeout=15)
            
            logger.info(f"HTTP Status: {res.status_code}")
            if res.status_code == 200:
                logger.info(f"응답 본문 일부: {res.text[:300]}")
                if "resultCode" in res.text or "response" in res.text:
                    logger.info(f"✅ 성공 가능성 발견! ({label})")
            else:
                logger.warning(f"❌ 실패 ({label}): {res.text[:100]}")
        except Exception as e:
            logger.error(f"에러 ({label}): {e}")

if __name__ == "__main__":
    test_public_doc_variants()
