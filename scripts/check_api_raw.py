import requests
import json

url = "https://apis.data.go.kr/1741000/publicDoc/getDocAll"
params = {
    "serviceKey": "e8cd6e25666c8391c17658e557e65e526051027e52a060548e0d5879a9fe5fc4",
    "pageNo": 1,
    "numOfRows": 1,
    "type": "json"
}

try:
    print(f"호출 URL: {url}")
    res = requests.get(url, params=params, timeout=20)
    print(f"상태 코드: {res.status_code}")
    
    if res.status_code == 200:
        # JSON 응답이면 예쁘게 출력, 아니면 텍스트 출력
        try:
            data = res.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except:
            print("응답이 JSON 형식이 아닙니다:")
            print(res.text[:1000])
    else:
        print(f"에러 응답: {res.text}")
except Exception as e:
    print(f"호출 에러: {e}")
