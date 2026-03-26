import json
import requests
from masking_pii import mask_pii


class CivilComplaintCrawler:
    """
    지자체 전자민원 수집 및 EXAONE-Deep-7.8B 챗 템플릿 변환 프로토타입
    """

    def __init__(self, api_key="sample_key"):
        self.api_key = api_key
        # 서울 열린데이터 광장 예시 엔드포인트 (S_EUNGDAPSO_CASE_INFO)
        self.base_url = f"http://openAPI.seoul.go.kr:8088/{api_key}/json/S_EUNGDAPSO_CASE_INFO"

    def fetch_samples(self, start=1, end=5):
        """
        서울시 응답소 민원사례 API 호출 시뮬레이션
        (실제 작동을 위해서는 API 키 발급이 필요하며, 여기서는 예시 샘플 데이터를 활용)
        """
        # 실제 호출 예시 (API 키 유효 시 주석 해제)
        # url = f"{self.base_url}/{start}/{end}/"
        # response = requests.get(url)
        # data = response.json()

        # Prototype 샘플 데이터 (API 응답 형식 가정)
        mock_response = [
            {
                "QSTN_CONT": "010-1234-5678로 연락 주세요. 집 앞 보도블록이 파손되어 보행에 불편함이 큽니다. 빠른 조치 부탁드립니다.",
                "ANSW_CONT": "안녕하세요. 요청하신 보도블록 보수 건에 대하여 조치 예정입니다. 빠른 시일 내에 보수하겠습니다.",
                "MENU_NM": "도로/교통",
            },
            {
                "QSTN_CONT": "저녁마다 불법 주차로 인해 소방도로가 막혀 있습니다. 단속 강화 바랍니다.",
                "ANSW_CONT": "불법 주정차 민원 관련하여 해당 구역 단속을 강화하도록 해당 부서에 전달하였습니다.",
                "MENU_NM": "도로/교통",
            },
        ]
        return mock_response

    def transform_to_exaone_format(self, raw_data):
        """
        수집된 데이터를 EXAONE-Deep-7.8B의 챗 템플릿 형식으로 변환 (Instruction-Input-Output)
        *Thought 태그는 학습 시 모델이 생성하도록 구성 (Response 앞단에 삽입 가능)
        """
        processed_data = []

        instruction = "다음 민원에 대해 단계적으로 분석하고, 표준 서식에 맞춰 공손하고 명확한 답변을 작성하세요."

        for item in raw_data:
            # 1. 비식별화
            masked_question = mask_pii(item.get("QSTN_CONT", ""))
            masked_answer = mask_pii(item.get("ANSW_CONT", ""))
            category = item.get("MENU_NM", "기타")

            # 2. EXAONE Chat Template 구조화
            # <thought> 태그 내의 추론 과정은 향후 모델이 생성할 수 있도록 가이드라인 설계
            formatted_entry = {
                "instruction": instruction,
                "input": f"[카테고리: {category}]\n민원 내용: {masked_question}",
                "output": f"<thought>\n1. 민원 분석: {category} 관련 시설물 보수 요청 확인.\n2. 답변 방향: 현장 확인 및 조치 계획 안내.\n3. 최종 답변 작성.\n</thought>\n{masked_answer}",
            }
            processed_data.append(formatted_entry)

        return processed_data


if __name__ == "__main__":
    crawler = CivilComplaintCrawler()

    # 1. 샘플 데이터 수집
    samples = crawler.fetch_samples()
    print(f"✅ {len(samples)}건의 샘플 데이터를 수집했습니다.")

    # 2. EXAONE 포맷 변환 및 비식별화
    dataset = crawler.transform_to_exaone_format(samples)

    # 3. 결과 저장 (JSONL 형식 권장)
    output_file = "dataset_prototype.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"🚀 EXAONE 템플릿 변환 완료: '{output_file}' 파일로 저장되었습니다.")
