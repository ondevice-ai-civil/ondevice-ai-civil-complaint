import requests
import os

def test_law():
    oc = os.getenv('LAW_GO_KR_OC')
    url = f"http://www.law.go.kr/DRF/lawSearch.do?target=law&query=민원&type=XML&OC={oc}"
    try:
        res = requests.get(url, timeout=10)
        print(f"[LAW] Status: {res.status_code}")
        if "사용자 정보 검증에 실패" in res.text:
            print("[LAW] ❌ IP 미승인 상태 (등록한 IP가 반영되지 않았거나 다름)")
        elif "<law" in res.text:
            print("[LAW] ✅ 인증 성공! 데이터 수집 가능")
        else:
            print(f"[LAW] ⚠️ 응답 확인 필요 (내용 일부): {res.text[:200]}")
    except Exception as e:
        print(f"[LAW] ❌ 에러: {e}")

def test_alio():
    key = os.getenv('DATA_GO_KR_API_KEY')
    # Decoding 키 사용
    url = "https://apis.data.go.kr/1051000/public_inst/list"
    params = {"serviceKey": key, "pageNo": 1, "numOfRows": 1, "resultType": "json"}
    try:
        res = requests.get(url, params=params, timeout=10)
        print(f"[ALIO] Status: {res.status_code}")
        if res.status_code == 200:
            if "SERVICE_KEY_IS_NOT_REGISTERED" in res.text:
                print("[ALIO] ❌ 키 미활성 상태 (동기화 대기 중)")
            elif "INVALID_REQUEST_PARAMETER_ERROR" in res.text:
                print("[ALIO] ❌ 파라미터 오류")
            else:
                try:
                    data = res.json()
                    # 결과 코드 확인
                    res_code = data.get("response", {}).get("header", {}).get("resultCode")
                    if res_code == "00":
                        print("[ALIO] ✅ 인증 성공! 데이터 수집 가능")
                    else:
                        print(f"[ALIO] ❌ 결과 오류 (코드: {res_code})")
                except:
                    print(f"[ALIO] ⚠️ 비정상 응답 (내용 일부): {res.text[:200]}")
        else:
            print(f"[ALIO] ❌ HTTP 오류: {res.status_code}")
    except Exception as e:
        print(f"[ALIO] ❌ 연결 에러: {e}")

if __name__ == "__main__":
    print("-" * 50)
    print("🚀 API 최종 유효성 검사 시작")
    test_law()
    test_alio()
    print("-" * 50)
