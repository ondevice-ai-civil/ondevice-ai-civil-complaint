import re


def mask_pii(text):
    """
    개인정보(PII)를 탐지하고 마스킹하는 간단한 정규표현식 모듈 (Prototype)
    """
    if not text:
        return ""

    # 1. 주민등록번호 (Mock)
    text = re.sub(r"\d{6}-\d{7}", "[주민번호 마스킹]", text)

    # 2. 전화번호 (Mock)
    text = re.sub(r"010-\d{3,4}-\d{4}", "010-****-****", text)

    # 3. 이메일 (Mock)
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[이메일 마스킹]", text)

    # 4. 이름 (2~4글자 가정, 간단한 예시)
    # 실제 운영 시에는 KoNLPy 또는 전용 NER 모델 활용 권장

    return text
