import os
import json
import subprocess
from dotenv import load_dotenv

# .env 파일에서 AIHUB_API_KEY 로드
load_dotenv()
API_KEY = os.getenv("AIHUB_API_KEY")


class AIHubCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.dataset_key = "71852"
        self.file_keys = ["553394", "553392"]  # 질의응답, 분류

    def download_file(self, file_key):
        """aihubshell을 사용하여 특정 파일을 다운로드 (Mock)"""
        print(f"📥 다운로드 시작 (file_key: {file_key})...")
        # 실제 다운로드 명령어 (커맨드라인에서 수동 실행 권장, 여기서는 로직만 작성)
        # command = f"./aihubshell -mode d -datasetkey {self.dataset_key} -filekey {file_key} -aihubapikey '{self.api_key}'"
        # subprocess.run(command, shell=True)
        print("💡 실제 대용량 데이터는 './aihubshell -mode d' 명령을 직접 사용하여 다운로드하세요.")

    def preprocess_json(self, json_path):
        """수집된 AI Hub JSON 데이터를 EXAONE-Deep 챗 템플릿으로 변환"""
        # AI Hub 공공 민원 데이터셋 구조 가정 (샘플 예시)
        # { "info": ..., "data": [{ "title": "...", "question": "...", "answer": "...", "category": "..." }] }

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            processed_dataset = []
            instruction = "공공 민원 데이터를 분석하여 단계별 추론 과정과 함께 공손하고 명확한 표준 답변을 작성하세요."

            # JSON 구조에 따라 수정 필요 (예시: data 필드 내 리스트 형태)
            items = raw_data.get("data", [])
            for item in items:
                # 1. 원본 데이터 추출
                question = item.get("question", "")
                answer = item.get("answer", "")
                category = item.get("category", "공공행정")

                # 2. EXAONE Chat Template (Instruction-Input-Output)
                formatted_entry = {
                    "instruction": instruction,
                    "input": f"[카테고리: {category}]\n민원 내용: {question}",
                    "output": f"<thought>\n1. 민원 유형 분석: {category} 관련 요청으로 파악됨.\n2. 핵심 정보 추출: 민원인의 주요 불편 사항 확인.\n3. 법령/규정 검토: 관련 지자체 조례 및 처리 지침 확인.\n4. 최종 답변 구성: 처리 절차 및 예상 소요 시간 안내.\n</thought>\n{answer}",
                }
                processed_dataset.append(formatted_entry)

            # 결과 저장
            output_name = f"exaone_ready_{os.path.basename(json_path)}"
            with open(output_name, "w", encoding="utf-8") as f:
                json.dump(processed_dataset, f, ensure_ascii=False, indent=2)

            print(f"✅ EXAONE 템플릿 변환 완료: {output_name}")

        except Exception as e:
            print(f"❌ 전처리 중 오류 발생: {e}")


if __name__ == "__main__":
    if not API_KEY:
        print("🚨 API KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    else:
        collector = AIHubCollector(API_KEY)
        print("🚀 AI Hub 데이터 수집 및 전처리 모듈 준비 완료.")
        # 사용 예시: collector.preprocess_json('TL_지방행정기관_질의응답.json')
