"""
Issue #70 - 학습 데이터 전면 재구성 v2 (로컬 실행)

v2.1 개선사항 (편향 분석 반영):
- 71847 카테고리 재매핑: 법률 title/caseTypeName/agenda 기반 세분화
- 71847 샘플링 제한: 전체의 25~35%로 축소 (민원 데이터 주축)
- 프롬프트 통일: 모든 소스에 동일 instruction 적용
- 답변 길이 필터: 71847 최소 100자, 전체 최소 50자
- 테스트셋 균형: 카테고리별 균등 + 71852 소스 우선

사용법:
    python src/data_collection_preprocessing/reconstruct_data_v2.py
"""

import os
import json
import glob
import re
import hashlib
import random
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Tuple, Optional

import numpy as np
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ─── 설정 ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data/raw/aihub")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_71852_LABEL = os.path.join(RAW_DIR, "71852/label")
DATASET_71852_SOURCE = os.path.join(RAW_DIR, "71852/source")
DATASET_619_LABEL = os.path.join(RAW_DIR, "619/label")
DATASET_98_LABEL = os.path.join(RAW_DIR, "98/label")
DATASET_71847_JSON = os.path.join(RAW_DIR, "71847/json")

MIN_PER_CATEGORY = 30

# 71847 비율 제한: 전체의 25~35%
MAX_71847_RATIO = 0.30

# 답변 길이 필터
MIN_ANSWER_LEN_71847 = 100   # 71847 최소 답변 길이
MIN_ANSWER_LEN_GLOBAL = 50   # 전체 최소 답변 길이

STANDARD_CATEGORIES = ["교통", "환경", "복지", "건축", "행정", "세금", "안전", "기타"]

SYSTEM_MESSAGE = "당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다."
INSTRUCTION = "다음 민원에 대해 공손하고 명확한 답변을 작성하세요."

# ─── EXAONE Chat Template ─────────────────────────────────────────────
def format_chat_template(system: str, user: str, assistant: str) -> str:
    """EXAONE chat template 포맷으로 변환. 마지막에 [|endofturn|] 포함."""
    return (
        f"[|system|]{system}[|endofturn|]\n"
        f"[|user|]{user}[|endofturn|]\n"
        f"[|assistant|]{assistant}[|endofturn|]"
    )


# ─── 카테고리 매핑 (71852/98용) ────────────────────────────────────────
CATEGORY_MAP = {
    "교통": "교통", "교통행정": "교통", "교통과": "교통",
    "대중교통": "교통", "도로교통": "교통", "교통정책": "교통",
    "교통정책과": "교통", "도로과": "교통",
    "환경": "환경", "환경과": "환경", "환경미화": "환경",
    "환경위생": "환경", "환경정책": "환경", "상하수도": "환경",
    "수도": "환경", "하수도": "환경", "청소행정": "환경",
    "공원녹지": "환경", "산림": "환경", "녹지": "환경",
    "복지": "복지", "복지과": "복지", "복지정책": "복지",
    "사회복지": "복지", "보건": "복지", "보건소": "복지",
    "보건의료": "복지", "노인복지": "복지", "아동복지": "복지",
    "장애인복지": "복지", "여성가족": "복지", "주민생활지원": "복지",
    "건축": "건축", "건축과": "건축", "건축허가": "건축",
    "건설": "건축", "도시계획": "건축", "주택": "건축",
    "도시개발": "건축", "건축행정": "건축", "개발행위": "건축",
    "토지": "건축", "부동산": "건축",
    "행정": "행정", "행정과": "행정", "일반행정": "행정",
    "총무": "행정", "민원봉사": "행정", "자치행정": "행정",
    "인사": "행정", "기획": "행정", "감사": "행정",
    "법무": "행정", "홍보": "행정", "문화체육": "행정",
    "문화": "행정", "체육": "행정", "관광": "행정",
    "정보통신": "행정", "전산": "행정",
    "세무": "세금", "세금": "세금", "세무과": "세금",
    "재정": "세금", "회계": "세금", "징수": "세금",
    "안전": "안전", "재난안전": "안전", "안전건설": "안전",
    "소방": "안전", "방재": "안전", "민방위": "안전",
    "안전관리": "안전", "재난": "안전",
    "기타": "기타", "경제": "기타", "농업": "기타",
    "축산": "기타", "수산": "기타", "위생": "기타",
    "자동차": "기타",
}

CATEGORY_619_MAP = {
    "건축허가": "건축", "경제": "기타", "공통": "행정",
    "교통": "교통", "농업_축산": "기타", "문화_체육_관광": "행정",
    "보건소": "복지", "복지": "복지", "산림": "환경",
    "상하수도": "환경", "세무": "세금", "안전건설": "안전",
    "위생": "환경", "자동차": "교통", "정보통신": "행정",
    "토지": "건축", "행정": "행정", "환경미화": "환경",
}

DASAN_CATEGORY_MAP = {
    "대중교통 안내": "교통",
    "생활하수도 관련 문의": "환경",
    "일반행정 문의": "행정",
}


# ─── 71847 법률 제목 기반 카테고리 매핑 ─────────────────────────────────
# 키워드 → 카테고리 (우선순위 순서대로 매칭)
LAW_TITLE_CATEGORY_KEYWORDS = {
    "교통": [
        "도로교통", "교통", "자동차", "여객", "운수", "운송", "화물",
        "철도", "항공", "항만", "해운", "선박", "도로법", "고속도로",
        "주차", "면허", "운전",
    ],
    "환경": [
        "환경", "대기", "수질", "폐기물", "소음", "진동", "토양오염",
        "자연환경", "생태", "녹색", "탄소", "기후", "물관리",
        "하수", "상수", "수도", "공원", "녹지", "산림", "산지",
        "야생", "동물보호",
    ],
    "건축": [
        "건축", "주택", "도시계획", "국토", "도시개발", "택지",
        "공동주택", "임대주택", "부동산", "토지", "건설",
        "개발제한", "도시정비", "주거환경", "재건축", "재개발",
        "산업단지", "공장설립", "공유재산",
    ],
    "복지": [
        "복지", "기초생활", "아동", "노인", "장애인", "보육",
        "의료", "건강보험", "국민연금", "연금", "고용보험",
        "산업재해", "보건", "사회보장", "양육", "출산",
        "보훈", "국가유공자", "청년기본",
    ],
    "세금": [
        "세법", "세금", "국세", "지방세", "소득세", "법인세",
        "부가가치세", "상속세", "증여세", "관세", "조세",
        "세징수", "세특례", "세기본",
    ],
    "안전": [
        "안전", "재난", "소방", "방재", "민방위", "위험물",
        "승강기", "원자력", "화재", "방화", "보행안전",
        "어린이 식생활안전",
    ],
}

# P 파일(판결문)의 caseTypeName 매핑
CASE_TYPE_CATEGORY_MAP = {
    "세무": "세금",
    "일반행정": "행정",
    "특허": "기타",
}


def map_71847_category_by_title(title: str) -> str:
    """법률 제목(title) 기반으로 71847 카테고리를 매핑."""
    if not title:
        return "행정"
    for category, keywords in LAW_TITLE_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in title:
                return category
    return "행정"


def map_71847_category_by_agenda(agenda: str) -> str:
    """해석례(H)의 agenda 필드에서 법률명을 추출하여 카테고리 매핑."""
    if not agenda:
        return "행정"
    # 「법률명」 패턴에서 첫 번째 법률명 추출
    matches = re.findall(r'「([^」]+)」', agenda)
    for law_name in matches:
        cat = map_71847_category_by_title(law_name)
        if cat != "행정":
            return cat
    # 법률명 패턴이 없으면 agenda 전체에서 키워드 검색
    return map_71847_category_by_title(agenda)


# ─── 유틸리티 함수 ─────────────────────────────────────────────────────

def map_category(raw_category: str) -> str:
    if not raw_category:
        return "기타"
    raw = raw_category.strip()
    if raw in CATEGORY_MAP:
        return CATEGORY_MAP[raw]
    for key, val in CATEGORY_MAP.items():
        if key in raw or raw in key:
            return val
    return "기타"


def improve_pii_masking_v2(text: str) -> str:
    """PII 마스킹 v2: 토큰 통일 + 특수문자 마스킹 제거."""
    if not text:
        return text
    result = text

    # 표준 PII 태그 치환
    result = re.sub(r'(\[NAME_MASKED\])+', '[이름]', result)
    result = re.sub(r'<NAME>', '[이름]', result)
    result = re.sub(r'<MOBILE_NUMBER>', '[전화번호]', result)
    result = re.sub(r'<PHONE_NUMBER>', '[전화번호]', result)
    result = re.sub(r'<ADDRESS>', '[주소]', result)
    result = re.sub(r'<DATE>', '[날짜]', result)
    result = re.sub(r'<TIME>', '[시간]', result)
    result = re.sub(r'<CHARGE>', '[금액]', result)
    result = re.sub(r'<BIRTH_NUMBER>', '[생년월일]', result)

    # 해시태그 형식 PII
    result = re.sub(r'#@주소#', '[주소]', result)
    result = re.sub(r'#@이름#', '[이름]', result)
    result = re.sub(r'#@전화번호#', '[전화번호]', result)
    result = re.sub(r'#@생년월일#', '[생년월일]', result)
    result = re.sub(r'#@카드번호#', '[카드번호]', result)
    result = re.sub(r'#@계좌번호#', '[계좌번호]', result)

    # v2: 특수문자 마스킹 패턴 제거 (빈 문자열로)
    result = re.sub(r'[▲]{2,}', '', result)
    result = re.sub(r'[○]{2,}', '', result)
    result = re.sub(r'[●]{2,}', '', result)
    result = re.sub(r'[△]{2,}', '', result)
    result = re.sub(r'[□]{2,}', '', result)
    result = re.sub(r'[■]{2,}', '', result)

    # 연속 PII 태그 병합
    result = re.sub(r'(\[이름\])\s*(\[이름\])+', '[이름]', result)
    result = re.sub(r'(\[전화번호\])\s*(\[전화번호\])+', '[전화번호]', result)
    result = re.sub(r'(\[주소\])\s*(\[주소\])+', '[주소]', result)

    # 다중 공백 정리
    result = re.sub(r'  +', ' ', result).strip()
    return result


def calculate_pii_density(text: str) -> float:
    """텍스트 내 PII 토큰 밀도를 계산."""
    if not text:
        return 0.0
    pii_patterns = [
        r'\[이름\]', r'\[전화번호\]', r'\[주소\]', r'\[날짜\]',
        r'\[시간\]', r'\[금액\]', r'\[생년월일\]', r'\[카드번호\]',
        r'\[계좌번호\]', r'\[NAME_MASKED\]',
        r'[○]{2,}', r'[▲]{2,}', r'[●]{2,}',
    ]
    total_len = len(text)
    pii_len = sum(len(m.group()) for pat in pii_patterns for m in re.finditer(pat, text))
    return pii_len / total_len if total_len > 0 else 0.0


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """두 텍스트의 단어 수준 Jaccard 유사도."""
    words_a = set(text_a.split())
    words_b = set(text_b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def has_repetition_pattern(text: str) -> bool:
    """같은 단어가 3회 이상 연속 반복되는지 확인."""
    words = text.split()
    if len(words) < 3:
        return False
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2] and len(words[i]) > 1:
            return True
    # 같은 문장이 3번 이상 반복
    sentences = re.split(r'[.!?。]\s*', text)
    if len(sentences) >= 3:
        sentence_counter = Counter(s.strip() for s in sentences if len(s.strip()) > 5)
        if sentence_counter and sentence_counter.most_common(1)[0][1] >= 3:
            return True
    return False


def is_low_quality(question: str, answer: str, min_answer_len: int = 50) -> Optional[str]:
    """저품질 판정. 제거 사유를 반환. None이면 통과."""
    if len(answer) < min_answer_len:
        return "answer_too_short"
    if len(question) < 10:
        return "question_too_short"
    if jaccard_similarity(question, answer) > 0.8:
        return "jaccard_too_high"
    if has_repetition_pattern(answer):
        return "repetition_pattern"
    if calculate_pii_density(answer) > 0.5:
        return "pii_density_high"
    return None


def parse_consulting_content(content: str) -> Tuple[str, str, str]:
    """71852 consulting_content에서 제목, 질문, 답변을 파싱."""
    if not content:
        return "", "", ""

    title = ""
    title_match = re.search(r"제목\s*:\s*(.+?)(?:\n|Q\s*:)", content, re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()

    q_match = re.search(r"\nQ\s*:\s*", content, re.IGNORECASE)
    a_match = re.search(r"\nA\s*:\s*", content, re.IGNORECASE)

    question, answer = "", ""
    if q_match and a_match:
        question = content[q_match.end():a_match.start()].strip()
        answer = content[a_match.end():].strip()
    elif q_match:
        question = content[q_match.end():].strip()
    elif a_match:
        question = content[:a_match.start()].strip()
        answer = content[a_match.end():].strip()
    else:
        question = content.strip()

    if title and question.startswith(title):
        question = question[len(title):].strip()

    return title, question, answer


def save_jsonl(records: list, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved: {filepath} ({len(records)} records)")


# ─── 1. 71852 데이터 처리 ──────────────────────────────────────────────

def process_71852() -> list:
    print("\n" + "=" * 60)
    print("  1. 71852 데이터 처리 (Primary)")
    print("=" * 60)

    label_files = sorted(glob.glob(os.path.join(DATASET_71852_LABEL, "*.json")))
    source_files = sorted(glob.glob(os.path.join(DATASET_71852_SOURCE, "*.json")))
    all_files = [(f, "label") for f in label_files] + [(f, "source") for f in source_files]

    print(f"Label: {len(label_files)}, Source: {len(source_files)}, Total: {len(all_files)}")

    records = []
    stats = defaultdict(int)

    for filepath, file_type in tqdm(all_files, desc="71852 파싱"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
        except Exception:
            stats["parse_fail"] += 1
            continue

        content = data.get("consulting_content", "")
        raw_category = data.get("consulting_category", "")
        filename = os.path.basename(filepath).replace(".json", "")

        title, question, answer = parse_consulting_content(content)

        # PII 마스킹 (필터링 전에 적용)
        question = improve_pii_masking_v2(question)
        answer = improve_pii_masking_v2(answer)

        # 품질 필터링 (전체 최소 답변 길이 적용)
        reason = is_low_quality(question, answer, min_answer_len=MIN_ANSWER_LEN_GLOBAL)
        if reason:
            stats[reason] += 1
            continue

        category = map_category(raw_category)
        records.append({
            "id": f"71852_{file_type}_{filename}",
            "question": question,
            "answer": answer,
            "category": category,
            "source_dataset": f"71852_{file_type}",
        })

    print(f"유효: {len(records)}")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  제거({k}): {v}")

    if records:
        a_lens = [len(r["answer"]) for r in records]
        print(f"답변 길이 - 평균: {np.mean(a_lens):.0f}, 중앙값: {np.median(a_lens):.0f}")
        cat_dist = Counter(r["category"] for r in records)
        for cat, cnt in cat_dist.most_common():
            print(f"  {cat}: {cnt}")

    return records


# ─── 2. 98 다산콜센터 필터링 ───────────────────────────────────────────

def process_98() -> list:
    print("\n" + "=" * 60)
    print("  2. 98 다산콜센터 필터링")
    print("=" * 60)

    if not os.path.exists(DATASET_98_LABEL):
        print("98 데이터 없음 - 스킵")
        return []

    records = []
    skipped = 0
    filtered = 0

    for fpath in tqdm(sorted(glob.glob(os.path.join(DATASET_98_LABEL, "*.json"))), desc="98 처리"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, list):
            continue

        dialog_groups = defaultdict(list)
        for rec in data:
            dialog_id = rec.get("대화셋일련번호", "")
            dialog_groups[dialog_id].append(rec)

        for dialog_id, turns in dialog_groups.items():
            cat = turns[0].get("카테고리", "")
            if cat not in DASAN_CATEGORY_MAP:
                skipped += 1
                continue

            std_category = DASAN_CATEGORY_MAP[cat]
            questions, answers = [], []

            for turn in sorted(turns, key=lambda x: int(x.get("문장번호", 0))):
                q = turn.get("고객질문(요청)", "").strip()
                a = turn.get("상담사답변", "").strip()
                if q:
                    questions.append(q)
                if a:
                    answers.append(a)

            full_q = " ".join(questions).strip()
            full_a = " ".join(answers).strip()

            full_q = improve_pii_masking_v2(full_q)
            full_a = improve_pii_masking_v2(full_a)

            reason = is_low_quality(full_q, full_a, min_answer_len=MIN_ANSWER_LEN_GLOBAL)
            if reason:
                filtered += 1
                continue

            records.append({
                "id": f"98_{dialog_id}",
                "question": full_q,
                "answer": full_a,
                "category": std_category,
                "source_dataset": "98",
            })

    print(f"민원 관련: {len(records)}, 스킵(비민원): {skipped}, 저품질제거: {filtered}")
    return records


# ─── 3. 71847 행정법 QA 데이터 처리 (카테고리 세분화 + 샘플링) ────────

def process_71847(max_records: int = None) -> list:
    """71847 데이터를 카테고리 세분화하여 처리.

    Args:
        max_records: 최대 반환 레코드 수 (None이면 제한 없음, 이후 sampling에서 제한)
    """
    print("\n" + "=" * 60)
    print("  3. 71847 행정법 QA 데이터 처리 (카테고리 세분화)")
    print("=" * 60)

    if not os.path.exists(DATASET_71847_JSON):
        print("71847 데이터 없음 - 스킵")
        return []

    json_files = sorted(glob.glob(os.path.join(DATASET_71847_JSON, "*.json")))
    print(f"JSON 파일 수: {len(json_files)}")

    records = []
    stats = defaultdict(int)
    cat_stats = defaultdict(int)

    for fpath in tqdm(json_files, desc="71847 처리"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            stats["parse_fail"] += 1
            continue

        info = data.get("info", {})
        label = data.get("label", {})
        question = label.get("input", "").strip()
        answer = label.get("output", "").strip()
        filename = os.path.basename(fpath).replace(".json", "")

        # 유형 추출 (HJ_B_xxx → B)
        parts = filename.split("_")
        law_type = parts[1] if len(parts) > 1 else ""

        # 카테고리 세분화: 소스 유형별 다른 전략
        if law_type == "B":
            # 법령: title 기반 매핑
            title = info.get("title", "")
            category = map_71847_category_by_title(title)
        elif law_type == "P":
            # 판결문: caseTypeName 기반 매핑
            case_type = info.get("caseTypeName", "")
            category = CASE_TYPE_CATEGORY_MAP.get(case_type, "행정")
        elif law_type == "H":
            # 해석례: agenda 필드에서 법률명 추출
            agenda = info.get("agenda", "")
            category = map_71847_category_by_agenda(agenda)
        elif law_type == "K":
            # 결정례: caseName에서 법률 키워드 추출
            case_name = info.get("caseName", "")
            category = map_71847_category_by_title(case_name)
        else:
            category = "행정"

        # 품질 필터링 (71847 전용 최소 답변 길이)
        reason = is_low_quality(question, answer, min_answer_len=MIN_ANSWER_LEN_71847)
        if reason:
            stats[reason] += 1
            continue

        cat_stats[category] += 1
        records.append({
            "id": f"71847_{filename}",
            "question": question,
            "answer": answer,
            "category": category,
            "source_dataset": "71847",
        })

    print(f"품질 필터 통과: {len(records)}")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  제거({k}): {v}")

    print(f"\n카테고리 분포 (필터 전):")
    for cat, cnt in sorted(cat_stats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt} ({cnt / len(records) * 100:.1f}%)")

    if records:
        a_lens = [len(r["answer"]) for r in records]
        print(f"답변 길이 - 평균: {np.mean(a_lens):.0f}, 중앙값: {np.median(a_lens):.0f}")

    return records


def sample_71847(records_71847: list, other_count: int) -> list:
    """71847 데이터를 전체의 25~35% 비율로 샘플링.

    민원 관련 카테고리(교통, 환경, 건축, 복지, 세금, 안전) 우선 선별.
    순수 행정/기타 카테고리는 비율 제한.
    """
    if not records_71847:
        return []

    # 목표: 전체의 MAX_71847_RATIO
    # other_count + sampled_71847 = total
    # sampled_71847 / total = MAX_71847_RATIO
    # sampled_71847 = other_count * MAX_71847_RATIO / (1 - MAX_71847_RATIO)
    target_count = int(other_count * MAX_71847_RATIO / (1 - MAX_71847_RATIO))
    target_count = min(target_count, len(records_71847))

    print(f"\n71847 샘플링: {len(records_71847)} -> {target_count} (목표 비율 {MAX_71847_RATIO*100:.0f}%)")

    # 카테고리별 그룹핑
    cat_groups = defaultdict(list)
    for rec in records_71847:
        cat_groups[rec["category"]].append(rec)

    # 민원 관련 카테고리 우선 (비-행정, 비-기타)
    civil_categories = ["교통", "환경", "건축", "복지", "세금", "안전"]
    admin_categories = ["행정", "기타"]

    # 1단계: 민원 관련 카테고리는 전량 포함 (target 범위 내)
    sampled = []
    civil_total = sum(len(cat_groups.get(cat, [])) for cat in civil_categories)
    admin_total = sum(len(cat_groups.get(cat, [])) for cat in admin_categories)

    print(f"  민원 관련 카테고리: {civil_total}건")
    print(f"  행정/기타 카테고리: {admin_total}건")

    # 민원 관련 카테고리 전량 추가
    for cat in civil_categories:
        group = cat_groups.get(cat, [])
        random.shuffle(group)
        sampled.extend(group)

    # 2단계: 남은 할당량을 행정/기타에서 채움
    remaining = target_count - len(sampled)
    if remaining > 0 and admin_total > 0:
        # 행정/기타 카테고리 비율 배분
        for cat in admin_categories:
            group = cat_groups.get(cat, [])
            if not group:
                continue
            # 비례 배분
            cat_quota = int(remaining * len(group) / admin_total)
            cat_quota = min(cat_quota, len(group))
            random.shuffle(group)
            sampled.extend(group[:cat_quota])
    elif len(sampled) > target_count:
        # 민원 관련만으로도 초과 시, 카테고리별 균등 삭감
        random.shuffle(sampled)
        sampled = sampled[:target_count]

    # 최종 분포 출력
    final_dist = Counter(r["category"] for r in sampled)
    print(f"  샘플링 결과: {len(sampled)}건")
    for cat, cnt in sorted(final_dist.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {cnt}")

    return sampled


# ─── 4. Chat Template 변환 + Split + 저장 ─────────────────────────────

def format_and_split(records_71852: list, records_98: list, records_71847: list = None):
    if records_71847 is None:
        records_71847 = []

    print("\n" + "=" * 60)
    print("  4. 중복 제거 & 71847 샘플링 & Chat Template 변환 & 층화 분할")
    print("=" * 60)

    # 1단계: 먼저 각 소스별 중복 제거 (질문 기준)
    # 71852/98을 먼저 등록하여 우선권 부여 (민원 데이터가 주축)
    seen = set()
    unique_71852 = []
    for rec in records_71852:
        h = hashlib.md5(rec["question"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_71852.append(rec)

    unique_98 = []
    for rec in records_98:
        h = hashlib.md5(rec["question"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_98.append(rec)

    unique_71847_all = []
    for rec in records_71847:
        h = hashlib.md5(rec["question"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_71847_all.append(rec)

    dedup_removed = (len(records_71852) + len(records_98) + len(records_71847)
                     - len(unique_71852) - len(unique_98) - len(unique_71847_all))
    print(f"중복 제거: {dedup_removed}건 제거")
    print(f"  71852: {len(records_71852)} -> {len(unique_71852)}")
    print(f"  98: {len(records_98)} -> {len(unique_98)}")
    print(f"  71847: {len(records_71847)} -> {len(unique_71847_all)}")

    # 2단계: 71847 샘플링 (중복 제거 후 기준)
    other_count = len(unique_71852) + len(unique_98)
    sampled_71847 = sample_71847(unique_71847_all, other_count)

    unique = unique_71852 + unique_98 + sampled_71847
    print(f"\n최종 레코드: {len(unique)} "
          f"(71852: {len(unique_71852)}, 98: {len(unique_98)}, "
          f"71847: {len(sampled_71847)}/{len(unique_71847_all)})")

    # 71847 비율 확인
    ratio_71847 = len(sampled_71847) / len(unique) * 100 if unique else 0
    print(f"71847 비율: {ratio_71847:.1f}% (목표: {MAX_71847_RATIO*100:.0f}%)")

    # EXAONE Chat Template 포맷 변환 (프롬프트 통일)
    formatted = []
    for rec in unique:
        cat = rec["category"]
        # 모든 소스에 동일한 instruction 적용 (프롬프트 통일)
        user_text = f"{INSTRUCTION}\n\n[카테고리: {cat}]\n민원 내용: {rec['question']}"
        text = format_chat_template(SYSTEM_MESSAGE, user_text, rec["answer"])

        formatted.append({
            "text": text,
            "category": cat,
            "id": rec["id"],
            "source": rec.get("source_dataset", ""),
        })

    print(f"최종 레코드: {len(formatted)}")

    # 카테고리 분포
    cat_dist = Counter(r["category"] for r in formatted)
    for cat, cnt in sorted(cat_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt} ({cnt / len(formatted) * 100:.1f}%)")

    # 소스별 분포
    source_dist = Counter(r["source"] for r in formatted)
    print(f"\n소스별 분포:")
    for src, cnt in sorted(source_dist.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt} ({cnt / len(formatted) * 100:.1f}%)")

    # ─── 층화 분할 (테스트셋 균형화) ───
    cat_groups = defaultdict(list)
    for rec in formatted:
        cat_groups[rec["category"]].append(rec)

    train, val, test = [], [], []

    for cat in STANDARD_CATEGORIES:
        data = cat_groups.get(cat, [])
        n = len(data)
        if n == 0:
            continue

        random.shuffle(data)

        if n < 10:
            train.extend(data)
            continue

        # 테스트셋: 카테고리별 균형 (각 카테고리에서 최소 30건)
        test_size = max(MIN_PER_CATEGORY, int(n * 0.1))
        test_size = min(test_size, n // 3)
        val_size = max(MIN_PER_CATEGORY // 2, int(n * 0.1))
        val_size = min(val_size, n // 3)
        train_size = n - test_size - val_size

        if train_size < 1:
            train_size = max(1, n - 2)
            test_size = 1
            val_size = n - train_size - test_size

        # 테스트셋에서 71852 소스 우선 배치
        data_71852 = [r for r in data if r.get("source", "").startswith("71852")]
        data_other = [r for r in data if not r.get("source", "").startswith("71852")]

        # 71852를 테스트셋 앞쪽에 배치하여 우선 선택되도록
        reordered = data_71852 + data_other

        test.extend(reordered[:test_size])
        val.extend(reordered[test_size:test_size + val_size])
        train.extend(reordered[test_size + val_size:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(f"\nTrain: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Test set 카테고리 확인
    test_cats = Counter(r["category"] for r in test)
    for cat in STANDARD_CATEGORIES:
        cnt = test_cats.get(cat, 0)
        status = "OK" if cnt >= MIN_PER_CATEGORY else f"WARN (<{MIN_PER_CATEGORY})"
        print(f"  Test [{cat}]: {cnt} {status}")

    # Test set 소스 분포
    test_sources = Counter(r.get("source", "") for r in test)
    print(f"\nTest 소스 분포:")
    for src, cnt in sorted(test_sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")

    # ─── 데이터 누출 검증 ───
    train_h = {hashlib.md5(r["text"].encode()).hexdigest() for r in train}
    val_h = {hashlib.md5(r["text"].encode()).hexdigest() for r in val}
    test_h = {hashlib.md5(r["text"].encode()).hexdigest() for r in test}
    leak_tv = len(train_h & val_h)
    leak_tt = len(train_h & test_h)
    leak_vt = len(val_h & test_h)
    print(f"\n데이터 누출: train-val={leak_tv}, train-test={leak_tt}, val-test={leak_vt}")

    # ─── 저장 ───
    # 저장 시 source 필드 제거 (학습에 불필요)
    def strip_source(records):
        return [{"text": r["text"], "category": r["category"], "id": r["id"]} for r in records]

    print(f"\n파일 저장 중...")
    save_jsonl(strip_source(train), os.path.join(OUTPUT_DIR, "v2_train.jsonl"))
    save_jsonl(strip_source(val), os.path.join(OUTPUT_DIR, "v2_val.jsonl"))
    save_jsonl(strip_source(test), os.path.join(OUTPUT_DIR, "v2_test.jsonl"))

    # ─── 품질 리포트 ───
    def extract_answer(text):
        marker = "[|assistant|]"
        idx = text.rfind(marker)
        if idx >= 0:
            answer = text[idx + len(marker):]
            answer = answer.replace("[|endofturn|]", "").strip()
            return answer
        return ""

    all_answers = [extract_answer(r["text"]) for r in formatted]
    all_answer_lens = [len(a) for a in all_answers]
    all_pii = [calculate_pii_density(a) for a in all_answers]

    report = {
        "version": "v2.1_balanced",
        "format": "EXAONE chat template with [|endofturn|]",
        "changes_from_v2": [
            "71847 카테고리 세분화 (title/caseTypeName/agenda 기반)",
            f"71847 샘플링 제한 (전체의 {MAX_71847_RATIO*100:.0f}%)",
            "프롬프트 통일 (모든 소스 동일 instruction)",
            f"답변 길이 필터 강화 (71847: {MIN_ANSWER_LEN_71847}자, 전체: {MIN_ANSWER_LEN_GLOBAL}자)",
            "테스트셋 균형화 (71852 우선 + 카테고리 균등)",
        ],
        "total_records": len(formatted),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "sources": {
            "71852": len(records_71852),
            "98": len(records_98),
            "71847_total": len(records_71847),
            "71847_sampled": len(sampled_71847),
            "71847_ratio_percent": round(ratio_71847, 1),
        },
        "category_distribution": dict(cat_dist),
        "source_distribution": dict(source_dist),
        "unique_categories": len(cat_dist),
        "answer_length": {
            "mean": float(np.mean(all_answer_lens)),
            "median": float(np.median(all_answer_lens)),
            "min": int(np.min(all_answer_lens)),
            "max": int(np.max(all_answer_lens)),
        },
        "pii_density_percent": float(np.mean(all_pii) * 100),
        "quality_filters": [
            f"71847 answer < {MIN_ANSWER_LEN_71847} chars removed",
            f"other answer < {MIN_ANSWER_LEN_GLOBAL} chars removed",
            "question < 10 chars removed",
            "Jaccard similarity > 0.8 removed",
            "repetition pattern (3+ consecutive) removed",
            "PII density > 50% removed",
        ],
        "data_leakage": {
            "train_val": leak_tv,
            "train_test": leak_tt,
            "val_test": leak_vt,
        },
        "test_category_counts": dict(test_cats),
        "test_source_counts": dict(test_sources),
    }

    report_path = os.path.join(OUTPUT_DIR, "v2_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nQuality report: {report_path}")

    # 샘플 출력
    print("\n--- 샘플 데이터 (Train 첫 1건) ---")
    if train:
        sample = train[0]
        print(f"ID: {sample['id']}")
        print(f"Category: {sample['category']}")
        print(f"Text (첫 300자):\n{sample['text'][:300]}")
        print(f"...(생략)...")
        print(f"Text (끝 100자):\n{sample['text'][-100:]}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("  데이터 재구성 v2.1 (균형화) 완료")
    print("=" * 60)
    print(f"총 레코드: {report['total_records']:,}")
    print(f"Train/Val/Test: {report['train']:,} / {report['val']:,} / {report['test']:,}")
    print(f"카테고리: {report['unique_categories']}개")
    print(f"71847 비율: {ratio_71847:.1f}% (v2: 85.7% -> v2.1: {ratio_71847:.1f}%)")
    print(f"답변 길이: 평균 {report['answer_length']['mean']:.0f}자, "
          f"중앙값 {report['answer_length']['median']:.0f}자")
    print(f"PII 밀도: {report['pii_density_percent']:.2f}%")

    return formatted, train, val, test


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Issue #70 - 학습 데이터 전면 재구성 v2.1 (균형화)")
    print("=" * 60)
    print(f"Base: {BASE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"포맷: EXAONE Chat Template (text 필드)")
    print(f"71847 비율 제한: {MAX_71847_RATIO*100:.0f}%")
    print(f"답변 최소 길이: 71847={MIN_ANSWER_LEN_71847}자, 기타={MIN_ANSWER_LEN_GLOBAL}자")

    records_71852 = process_71852()
    records_98 = process_98()
    records_71847 = process_71847()
    format_and_split(records_71852, records_98, records_71847)


if __name__ == "__main__":
    main()
