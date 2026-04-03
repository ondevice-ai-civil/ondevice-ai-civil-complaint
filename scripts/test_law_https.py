import requests
import os

def test_law_https():
    oc = os.getenv('LAW_GO_KR_OC')
    # HTTPS 주소로 시도
    url = f"https://www.law.go.kr/DRF/lawSearch.do?target=law&query=민원&type=XML&OC={oc}"
    try:
        res = requests.get(url, timeout=15)
        print(f"URL: {url}")
        print(f"Status: {res.status_code}")
        if "사용자 정보 검증에 실패" in res.text:
            print("❌ HTTPS로도 IP 인증 실패")
        elif "<law" in res.text:
            print("✅ HTTPS 호출 성공!")
        else:
            print(f"⚠️ 응답 확인 필요: {res.text[:200]}")
    except Exception as e:
        print(f"❌ 에러: {e}")

if __name__ == "__main__":
    test_law_https()
