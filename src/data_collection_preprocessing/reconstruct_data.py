"""
Issue #70 - 학습 데이터 전면 재구성 스크립트 (로컬 실행)

근본 원인 분석 결과에 따른 데이터 품질 전면 재구축:
- 71852: 실제 민원 Q&A (consulting_content 전문 답변 활용)
- 619: Q-only 분류 보조 데이터 (카테고리당 500건 샘플링)
- 98: 다산콜센터 민원 관련만 필터링
- 71844: 금융 콜센터 전량 제거

사용법:
    python src/data_collection_preprocessing/reconstruct_data.py
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

MIN_TEST_TOTAL = 300
MIN_PER_CATEGORY = 30

STANDARD_CATEGORIES = ["교통", "환경", "복지", "건축", "행정", "세금", "안전", "기타"]

SYSTEM_MESSAGE = "당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다."
INSTRUCTION = "다음 민원에 대해 공손하고 명확한 답변을 작성하세요."

# ─── 카테고리 매핑 ─────────────────────────────────────────────────────
CATEGORY_MAP = {
    "교통": "교통",
    "교통행정": "교통",
    "교통과": "교통",
    "대중교통": "교통",
    "도로교통": "교통",
    "교통정책": "교통",
    "교통정책과": "교통",
    "도로과": "교통",
    "환경": "환경",
    "환경과": "환경",
    "환경미화": "환경",
    "환경위생": "환경",
    "환경정책": "환경",
    "상하수도": "환경",
    "수도": "환경",
    "하수도": "환경",
    "청소행정": "환경",
    "공원녹지": "환경",
    "산림": "환경",
    "녹지": "환경",
    "복지": "복지",
    "복지과": "복지",
    "복지정책": "복지",
    "사회복지": "복지",
    "보건": "복지",
    "보건소": "복지",
    "보건의료": "복지",
    "노인복지": "복지",
    "아동복지": "복지",
    "장애인복지": "복지",
    "여성가족": "복지",
    "주민생활지원": "복지",
    "건축": "건축",
    "건축과": "건축",
    "건축허가": "건축",
    "건설": "건축",
    "도시계획": "건축",
    "주택": "건축",
    "도시개발": "건축",
    "건축행정": "건축",
    "개발행위": "건축",
    "토지": "건축",
    "부동산": "건축",
    "행정": "행정",
    "행정과": "행정",
    "일반행정": "행정",
    "총무": "행정",
    "민원봉사": "행정",
    "자치행정": "행정",
    "인사": "행정",
    "기획": "행정",
    "감사": "행정",
    "법무": "행정",
    "홍보": "행정",
    "문화체육": "행정",
    "문화": "행정",
    "체육": "행정",
    "관광": "행정",
    "정보통신": "행정",
    "전산": "행정",
    "세무": "세금",
    "세금": "세금",
    "세무과": "세금",
    "재정": "세금",
    "회계": "세금",
    "징수": "세금",
    "안전": "안전",
    "재난안전": "안전",
    "안전건설": "안전",
    "소방": "안전",
    "방재": "안전",
    "민방위": "안전",
    "안전관리": "안전",
    "재난": "안전",
    "기타": "기타",
    "경제": "기타",
    "농업": "기타",
    "축산": "기타",
    "수산": "기타",
    "위생": "기타",
    "자동차": "기타",
}

CATEGORY_619_MAP = {
    "건축허가": "건축",
    "경제": "기타",
    "공통": "행정",
    "교통": "교통",
    "농업_축산": "기타",
    "문화_체육_관광": "행정",
    "보건소": "복지",
    "복지": "복지",
    "산림": "환경",
    "상하수도": "환경",
    "세무": "세금",
    "안전건설": "안전",
    "위생": "환경",
    "자동차": "교통",
    "정보통신": "행정",
    "토지": "건축",
    "행정": "행정",
    "환경미화": "환경",
}

DASAN_CATEGORY_MAP = {
    "대중교통 안내": "교통",
    "생활하수도 관련 문의": "환경",
    "일반행정 문의": "행정",
}

THOUGHT_TEMPLATES = {
    "교통": "이 민원은 교통 분야와 관련됩니다. 도로, 대중교통, 주차, 교통안전 등의 관련 정책과 담당 부서를 확인하고, 민원인에게 구체적인 진행 상황과 계획을 안내해야 합니다.",
    "환경": "이 민원은 환경 분야와 관련됩니다. 환경미화, 상하수도, 폐기물, 소음, 공원녹지 등 관련 조례와 처리 절차를 확인하고, 적절한 조치 방안을 안내해야 합니다.",
    "복지": "이 민원은 복지/보건 분야와 관련됩니다. 사회복지, 보건의료, 노인/아동/장애인 복지 등 관련 제도와 지원 자격 요건을 확인하고, 신청 방법과 절차를 안내해야 합니다.",
    "건축": "이 민원은 건축/도시계획 분야와 관련됩니다. 건축허가, 개발행위, 도시계획, 토지이용 등 관련 법령과 행정 절차를 확인하고, 필요한 서류와 처리 기한을 안내해야 합니다.",
    "행정": "이 민원은 일반 행정 분야와 관련됩니다. 민원 접수, 증명서 발급, 행정 절차 등 관련 규정을 확인하고, 처리 방법과 담당 부서 연락처를 안내해야 합니다.",
    "세금": "이 민원은 세무/재정 분야와 관련됩니다. 지방세, 과세 기준, 감면 요건, 납부 방법 등 관련 세법과 조례를 확인하고, 정확한 세금 정보를 안내해야 합니다.",
    "안전": "이 민원은 안전/재난 분야와 관련됩니다. 재난안전, 소방, 시설물 안전, 민방위 등 관련 법령과 비상 대응 절차를 확인하고, 안전 조치 사항을 안내해야 합니다.",
    "기타": "이 민원의 내용을 분석하여 관련 부서와 정책을 확인하고, 민원인에게 적절한 처리 방안과 담당 부서 정보를 안내해야 합니다.",
}


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


def load_71852_file(filepath: str) -> Optional[dict]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data
    except Exception:
        return None


def parse_consulting_content(content: str) -> Tuple[str, str, str]:
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
        question = content[q_match.end() : a_match.start()].strip()
        answer = content[a_match.end() :].strip()
    elif q_match:
        question = content[q_match.end() :].strip()
    elif a_match:
        question = content[: a_match.start()].strip()
        answer = content[a_match.end() :].strip()
    else:
        question = content.strip()

    if title and question.startswith(title):
        question = question[len(title) :].strip()

    return title, question, answer


def improve_pii_masking(text: str) -> str:
    if not text:
        return text
    result = text
    result = re.sub(r"(\[NAME_MASKED\])+", "[이름]", result)
    result = re.sub(r"<NAME>", "[이름]", result)
    result = re.sub(r"<MOBILE_NUMBER>", "[전화번호]", result)
    result = re.sub(r"<PHONE_NUMBER>", "[전화번호]", result)
    result = re.sub(r"<ADDRESS>", "[주소]", result)
    result = re.sub(r"<DATE>", "[날짜]", result)
    result = re.sub(r"<TIME>", "[시간]", result)
    result = re.sub(r"<CHARGE>", "[금액]", result)
    result = re.sub(r"<BIRTH_NUMBER>", "[생년월일]", result)
    result = re.sub(r"#@주소#", "[주소]", result)
    result = re.sub(r"#@이름#", "[이름]", result)
    result = re.sub(r"#@전화번호#", "[전화번호]", result)
    result = re.sub(r"#@생년월일#", "[생년월일]", result)
    result = re.sub(r"#@카드번호#", "[카드번호]", result)
    result = re.sub(r"#@계좌번호#", "[계좌번호]", result)
    result = re.sub(r"(\[이름\])\s*(\[이름\])+", "[이름]", result)
    result = re.sub(r"(\[전화번호\])\s*(\[전화번호\])+", "[전화번호]", result)
    result = re.sub(r"(\[주소\])\s*(\[주소\])+", "[주소]", result)
    return result


def calculate_pii_density(text: str) -> float:
    if not text:
        return 0.0
    pii_patterns = [
        r"\[이름\]",
        r"\[전화번호\]",
        r"\[주소\]",
        r"\[날짜\]",
        r"\[시간\]",
        r"\[금액\]",
        r"\[생년월일\]",
        r"\[카드번호\]",
        r"\[계좌번호\]",
        r"\[NAME_MASKED\]",
        r"\u25cb{2,}",
        r"\u25b2{2,}",
    ]
    total_len = len(text)
    pii_len = sum(len(m.group()) for pat in pii_patterns for m in re.finditer(pat, text))
    return pii_len / total_len if total_len > 0 else 0.0


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
    parse_failures = 0
    no_answer = 0

    for filepath, file_type in tqdm(all_files, desc="71852 파싱"):
        rec = load_71852_file(filepath)
        if rec is None:
            parse_failures += 1
            continue

        content = rec.get("consulting_content", "")
        raw_category = rec.get("consulting_category", "")
        source_region = rec.get("source", "")
        filename = os.path.basename(filepath).replace(".json", "")

        title, question, answer = parse_consulting_content(content)

        if not answer or len(answer) < 30:
            no_answer += 1
            continue
        if not question or len(question) < 10:
            continue

        category = map_category(raw_category)
        question = improve_pii_masking(question)
        answer = improve_pii_masking(answer)

        records.append(
            {
                "id": f"71852_{file_type}_{filename}",
                "question": question,
                "answer": answer,
                "title": title,
                "category": category,
                "raw_category": raw_category,
                "source_dataset": f"71852_{file_type}",
                "q_len": len(question),
                "a_len": len(answer),
            }
        )

    print(f"파싱 성공: {len(records)}, 실패: {parse_failures}, 답변부재: {no_answer}")

    if records:
        a_lens = [r["a_len"] for r in records]
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

    records = []
    skipped = 0

    for fpath in tqdm(glob.glob(os.path.join(DATASET_98_LABEL, "*.json")), desc="98 처리"):
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

            if not full_q or len(full_q) < 10 or not full_a or len(full_a) < 30:
                continue

            full_q = improve_pii_masking(full_q)
            full_a = improve_pii_masking(full_a)

            records.append(
                {
                    "id": f"98_{dialog_id}",
                    "question": full_q,
                    "answer": full_a,
                    "title": "",
                    "category": std_category,
                    "raw_category": cat,
                    "source_dataset": "98",
                    "q_len": len(full_q),
                    "a_len": len(full_a),
                }
            )

    print(f"민원 관련: {len(records)}, 스킵: {skipped}")
    return records


# ─── 3. 619 데이터 (Q-only) ────────────────────────────────────────────


def process_619() -> list:
    print("\n" + "=" * 60)
    print("  3. 619 데이터 처리 (Q-only, 분류 보조)")
    print("=" * 60)

    MAX_PER_CAT = 500
    records = []

    if not os.path.exists(DATASET_619_LABEL):
        print("619 데이터 없음 - 스킵")
        return records

    for cat_dir in tqdm(sorted(os.listdir(DATASET_619_LABEL)), desc="619 처리"):
        cat_path = os.path.join(DATASET_619_LABEL, cat_dir)
        if not os.path.isdir(cat_path):
            continue

        std_category = CATEGORY_619_MAP.get(cat_dir, "기타")
        cat_docs = []

        for fpath in glob.glob(os.path.join(cat_path, "*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for doc in data.get("documents", []):
                    q = doc.get("Q_refined", "").strip()
                    if q and len(q) >= 10:
                        q = improve_pii_masking(q)
                        cat_docs.append(
                            {
                                "id": f"619_{cat_dir}_{doc.get('id', '')}",
                                "question": q,
                                "category": std_category,
                                "raw_category": cat_dir,
                                "source": "619",
                            }
                        )
            except Exception:
                pass

        if len(cat_docs) > MAX_PER_CAT:
            random.shuffle(cat_docs)
            cat_docs = cat_docs[:MAX_PER_CAT]

        records.extend(cat_docs)
        print(f"  {cat_dir} -> {std_category}: {len(cat_docs)}건")

    print(f"총 619 샘플: {len(records)}")
    return records


# ─── 4. Instruction Format 변환 + Split + 저장 ────────────────────────


def format_and_split(records_71852: list, records_98: list, records_619: list):
    print("\n" + "=" * 60)
    print("  4. Instruction Format 변환 & Split")
    print("=" * 60)

    all_qa = records_71852 + records_98
    print(f"Q&A 레코드: {len(all_qa)} (71852: {len(records_71852)}, 98: {len(records_98)})")

    # 중복 제거
    seen = set()
    unique = []
    for rec in all_qa:
        h = hashlib.md5(rec["question"].encode(), usedforsecurity=False).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(rec)
    print(f"중복 제거 후: {len(unique)} (제거: {len(all_qa) - len(unique)})")

    # Instruction 포맷 변환 (기본: thought 없음)
    formatted = []
    formatted_thought = []
    for rec in unique:
        cat = rec["category"]
        input_text = f"[카테고리: {cat}]\n민원 내용: {rec['question']}"

        formatted.append(
            {
                "id": rec["id"],
                "instruction": INSTRUCTION,
                "input": input_text,
                "output": rec["answer"],
                "category": cat,
                "source": rec.get("source_dataset", ""),
            }
        )

        thought = THOUGHT_TEMPLATES.get(cat, THOUGHT_TEMPLATES["기타"])
        formatted_thought.append(
            {
                "id": rec["id"],
                "instruction": INSTRUCTION,
                "input": input_text,
                "output": f"<thought>\n{thought}\n</thought>\n{rec['answer']}",
                "category": cat,
                "source": rec.get("source_dataset", ""),
            }
        )

    print(f"최종 레코드: {len(formatted)}")

    # 카테고리 분포
    cat_dist = Counter(r["category"] for r in formatted)
    for cat, cnt in sorted(cat_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt} ({cnt / len(formatted) * 100:.1f}%)")

    # ─── 층화 분할 ───
    cat_groups = defaultdict(list)
    for rec in formatted:
        cat_groups[rec["category"]].append(rec)

    train, val, test = [], [], []

    for cat in STANDARD_CATEGORIES:
        data = cat_groups[cat]
        n = len(data)
        if n == 0:
            continue

        random.shuffle(data)

        if n < 10:
            train.extend(data)
            continue

        test_size = max(MIN_PER_CATEGORY, int(n * 0.1))
        test_size = min(test_size, n // 3)
        val_size = max(MIN_PER_CATEGORY // 2, int(n * 0.1))
        val_size = min(val_size, n // 3)
        train_size = n - test_size - val_size

        if train_size < 1:
            train_size = max(1, n - 2)
            test_size = 1
            val_size = n - train_size - test_size

        test.extend(data[:test_size])
        val.extend(data[test_size : test_size + val_size])
        train.extend(data[test_size + val_size :])

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

    # ─── 데이터 누출 검증 ───
    train_h = {hashlib.md5(r["input"].encode(), usedforsecurity=False).hexdigest() for r in train}
    val_h = {hashlib.md5(r["input"].encode(), usedforsecurity=False).hexdigest() for r in val}
    test_h = {hashlib.md5(r["input"].encode(), usedforsecurity=False).hexdigest() for r in test}
    print(
        f"\n데이터 누출: train-val={len(train_h & val_h)}, train-test={len(train_h & test_h)}, val-test={len(val_h & test_h)}"
    )

    # ─── 저장 ───
    print(f"\n파일 저장 중...")
    save_jsonl(train, os.path.join(OUTPUT_DIR, "civil_complaint_train.jsonl"))
    save_jsonl(val, os.path.join(OUTPUT_DIR, "civil_complaint_val.jsonl"))
    save_jsonl(test, os.path.join(OUTPUT_DIR, "civil_complaint_test.jsonl"))

    # thought 포함 버전
    thought_map = {r["id"]: r for r in formatted_thought}
    save_jsonl(
        [thought_map[r["id"]] for r in train if r["id"] in thought_map],
        os.path.join(OUTPUT_DIR, "civil_complaint_train_with_thought.jsonl"),
    )
    save_jsonl(
        [thought_map[r["id"]] for r in val if r["id"] in thought_map],
        os.path.join(OUTPUT_DIR, "civil_complaint_val_with_thought.jsonl"),
    )
    save_jsonl(
        [thought_map[r["id"]] for r in test if r["id"] in thought_map],
        os.path.join(OUTPUT_DIR, "civil_complaint_test_with_thought.jsonl"),
    )

    # 619 Q-only
    if records_619:
        save_jsonl(records_619, os.path.join(OUTPUT_DIR, "civil_complaint_619_qonly.jsonl"))

    # ─── 품질 리포트 ───
    all_output_lens = [len(r["output"]) for r in formatted]
    final_pii = [calculate_pii_density(r["input"] + " " + r["output"]) for r in formatted]

    report = {
        "version": "v2_reconstructed",
        "total_records": len(formatted),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "sources": {
            "71852_label": len([r for r in formatted if r["source"] == "71852_label"]),
            "71852_source": len([r for r in formatted if r["source"] == "71852_source"]),
            "98": len([r for r in formatted if r["source"] == "98"]),
            "71844": "전량 제거",
        },
        "category_distribution": dict(cat_dist),
        "unique_categories": len(cat_dist),
        "avg_output_length": float(np.mean(all_output_lens)),
        "median_output_length": float(np.median(all_output_lens)),
        "pii_density_percent": float(np.mean(final_pii) * 100),
    }

    report_path = os.path.join(OUTPUT_DIR, "civil_complaint_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nQuality report: {report_path}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("  데이터 재구성 완료")
    print("=" * 60)
    print(f"총 레코드: {report['total_records']:,}")
    print(f"카테고리: {report['unique_categories']}개")
    print(f"답변 길이: 평균 {report['avg_output_length']:.0f}자 (기존 97자)")
    print(f"PII 밀도: {report['pii_density_percent']:.2f}% (기존 23.35%)")

    return formatted, train, val, test


# ─── Main ──────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  Issue #70 - 학습 데이터 전면 재구성")
    print("=" * 60)
    print(f"Base: {BASE_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    records_71852 = process_71852()
    records_98 = process_98()
    records_619 = process_619()
    format_and_split(records_71852, records_98, records_619)


if __name__ == "__main__":
    main()
