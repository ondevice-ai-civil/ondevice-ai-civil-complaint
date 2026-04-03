import requests
import os
import sys

def test_law_api():
    oc = os.getenv('LAW_GO_KR_OC')
    url = "http://www.law.go.kr/DRF/lawSearch.do"
    params = {"target": "law", "query": "민원", "type": "XML", "OC": oc}
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200 and "<law" in res.text:
            print(f"[LAW API] ✅ 유효함 (상태코드: 200)")
            return True
        else:
            print(f"[LAW API] ❌ 오류 (상태코드: {res.status_code})")
            print(f"응답내용 일부: {res.text[:200]}")
            return False
    except Exception as e:
        print(f"[LAW API] ❌ 연결 실패: {e}")
        return False

def test_alio_api():
    key = os.getenv('DATA_GO_KR_API_KEY')
    # Decoding 키를 사용하기 때문에 requests가 한 번 더 인코딩하도록 함
    url = "https://apis.data.go.kr/1051000/public_inst/list"
    params = {"serviceKey": key, "pageNo": 1, "numOfRows": 1, "resultType": "json"}
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            try:
                data = res.json()
                code = data.get("response", {}).get("header", {}).get("resultCode")
                if code == "00":
                    print(f"[ALIO API] ✅ 유효함 (상태코드: 200, 결과코드: 00)")
                    return True
                else:
                    msg = data.get("response", {}).get("header", {}).get("resultMsg", "알 수 없는 오류")
                    print(f"[ALIO API] ❌ 인증 오류 (결과코드: {code}, 메시지: {msg})")
                    return False
            except Exception:
                if "<ServiceKey Error" in res.text:
                    print("[ALIO API] ❌ 인증키 오류 (ServiceKey Error)")
                else:
                    print(f"[ALIO API] ❌ 비정상 응답: {res.text[:200]}")
                return False
        else:
            print(f"[ALIO API] ❌ HTTP 오류 (상태코드: {res.status_code})")
            return False
    except Exception as e:
        print(f"[ALIO API] ❌ 연결 실패: {e}")
        return False

if __name__ == "__main__":
    print("-" * 50)
    print("🚀 API 키 유효성 검사 시작")
    law_ok = test_law_api()
    alio_ok = test_alio_api()
    print("-" * 50)
    if law_ok and alio_ok:
        print("✨ 모든 API 키가 정상적으로 작동합니다!")
    else:
        print("⚠️ 일부 API 키에 확인이 필요합니다.")
        sys.exit(1)
