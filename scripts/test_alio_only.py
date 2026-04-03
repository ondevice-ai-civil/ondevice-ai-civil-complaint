import requests
import os

def test_alio_api():
    key = os.getenv('DATA_GO_KR_API_KEY')
    # Decoding 키를 직접 사용하여 requests가 인코딩하도록 위임
    url = "https://apis.data.go.kr/1051000/public_inst/list"
    params = {
        "serviceKey": key,
        "pageNo": 1,
        "numOfRows": 1,
        "resultType": "json"
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        print(f"HTTP Status: {res.status_code}")
        if res.status_code == 200:
            if "<ServiceKey Error" in res.text:
                print("❌ 인증키 오류 (ServiceKey Error)")
                return False
            
            try:
                data = res.json()
                header = data.get("response", {}).get("header", {})
                code = header.get("resultCode")
                msg = header.get("resultMsg")
                if code == "00":
                    print(f"✅ ALIO API 유효함! (결과코드: {code})")
                    return True
                else:
                    print(f"❌ 인증 오류 발생 (코드: {code}, 메시지: {msg})")
                    return False
            except Exception as e:
                print(f"⚠️ JSON 파싱 실패 또는 비정상 응답: {res.text[:200]}")
                return False
        else:
            print(f"❌ HTTP 요청 실패 (Status: {res.status_code})")
            return False
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return False

if __name__ == "__main__":
    test_alio_api()
