import requests
import zipfile
import io
import json

url = "https://apis.data.go.kr/1741000/publicDoc/getDocAll"
params = {
    "serviceKey": "e8cd6e25666c8391c17658e557e65e526051027e52a060548e0d5879a9fe5fc4",
    "pageNo": 1,
    "numOfRows": 1,
    "type": "json"
}

try:
    print("API 호출 중...")
    res = requests.get(url, params=params, timeout=30)
    data = res.json()
    download_url = data['response']['body']['resultList'][0]['url']
    
    print(f"데이터 다운로드 중: {download_url}")
    zip_res = requests.get(download_url, timeout=60)
    
    with zipfile.ZipFile(io.BytesIO(zip_res.content)) as z:
        json_files = [f for f in z.namelist() if f.endswith('.json')]
        if json_files:
            print(f"JSON 파일 발견: {json_files[0]}")
            with z.open(json_files[0]) as f:
                sample_data = json.load(f)
                # 리스트면 첫 번째 항목, 아니면 데이터 자체 출력
                sample_item = sample_data[0] if isinstance(sample_data, list) else sample_data
                print("\n--- 데이터 샘플 구조 ---")
                print(json.dumps(sample_item, indent=2, ensure_ascii=False)[:2000])
        else:
            print("ZIP 내부에 JSON 파일이 없습니다.")
            print("전체 파일 목록:", z.namelist()[:10])
except Exception as e:
    print(f"에러 발생: {e}")
