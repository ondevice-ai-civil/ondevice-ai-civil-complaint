"""
CodeRabbit 리뷰 반영: RAG 테스트 데이터셋 품질 수정 스크립트

수정 항목:
1. 공문서 - 추진기간 날짜 역전, 협조기한이 생산일 이전
2. 공시자료 - 분기/제출일 불일치, 잘못된 행정구역, 반기/분기 충돌
3. 메뉴얼   - 전화번호 마스킹
4. manifest.json - format_distribution 전체 합계로 수정
5. README.md - 코드블록 언어 지정, 파일 끝 개행
"""

import json
import re
import os
from datetime import date, timedelta
import random

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'rag_test_dataset')
DATASET_DIR = os.path.abspath(DATASET_DIR)


# ─────────────────────────────────────────────
# 1. manifest.json: format_distribution 전체 합계
# ─────────────────────────────────────────────
def fix_manifest():
    path = os.path.join(DATASET_DIR, 'manifest.json')
    with open(path, encoding='utf-8') as f:
        m = json.load(f)

    m['format_distribution'] = {'txt': 160, 'pdf': 100, 'docx': 80, 'xlsx': 60}

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print('✅ manifest.json: format_distribution 수정 완료')


# ─────────────────────────────────────────────
# 2. README.md: 코드블록 언어, 파일 끝 개행
# ─────────────────────────────────────────────
def fix_readme():
    path = os.path.join(DATASET_DIR, 'README.md')
    with open(path, encoding='utf-8') as f:
        content = f.read()

    # 코드블록에 언어 추가 (``` 만 있는 경우 → ```text)
    content = re.sub(r'^```\s*$', '```text', content, flags=re.MULTILINE)

    # 합성 데이터 면책 문구 보강 (전화번호 관련)
    disclaimer = '\n- 모든 전화번호·인명·주소는 합성 데이터이며 실존 인물·기관과 무관합니다.\n'
    if '전화번호·인명·주소는 합성' not in content:
        content = content.replace(
            '- 상업적 사용 시 관련 법령을 확인하십시오.',
            '- 모든 전화번호·인명·주소는 합성 데이터이며 실존 인물·기관과 무관합니다.\n- 상업적 사용 시 관련 법령을 확인하십시오.'
        )

    # 파일 끝 단일 개행
    content = content.rstrip('\n') + '\n'

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✅ README.md: 코드블록 언어·개행·면책문구 수정 완료')


# ─────────────────────────────────────────────
# 3. 공문서 TXT: 날짜 논리 오류 수정
# ─────────────────────────────────────────────
KO_DATE_PAT = re.compile(r'(\d{4})년\s*(\d{2})월\s*(\d{2})일')

def parse_ko_date(s):
    m = KO_DATE_PAT.search(s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None

def fmt_ko_date(d):
    return f'{d.year}년 {d.month:02d}월 {d.day:02d}일'

def fix_gongmunso_dates(text):
    lines = text.split('\n')
    prod_date = None
    doc_number_line_idx = None

    # 생산일자 파악
    for line in lines:
        if '생산일자:' in line:
            prod_date = parse_ko_date(line)
            break

    changed = False
    new_lines = []
    for i, line in enumerate(lines):
        # 문서번호 연도 ↔ 생산일자 연도 불일치 수정
        if '문서번호:' in line and prod_date:
            # 문서번호에서 연도 추출 및 생산일자 연도로 맞추기
            num_match = re.search(r'(\w+)\s+(\d{4})-', line)
            if num_match:
                doc_year = int(num_match.group(2))
                if abs(doc_year - prod_date.year) > 1:
                    line = re.sub(r'(\w+\s+)\d{4}(-)', lambda m: m.group(1) + str(prod_date.year) + m.group(2), line, count=1)
                    changed = True

        # 추진 기간 날짜 역전 수정
        if '추진 기간:' in line:
            dates = KO_DATE_PAT.findall(line)
            if len(dates) == 2:
                d1 = date(int(dates[0][0]), int(dates[0][1]), int(dates[0][2]))
                d2 = date(int(dates[1][0]), int(dates[1][1]), int(dates[1][2]))
                if d1 > d2:
                    # 날짜 순서 교환
                    line = line.replace(
                        f'{dates[0][0]}년 {dates[0][1]}월 {dates[0][2]}일',
                        '__START__', 1
                    ).replace(
                        f'{dates[1][0]}년 {dates[1][1]}월 {dates[1][2]}일',
                        '__END__', 1
                    ).replace('__START__', fmt_ko_date(d2)).replace('__END__', fmt_ko_date(d1))
                    changed = True

        # 협조 기한이 생산일 이전인 경우 수정
        if prod_date and ('자료 제출 기한:' in line or '회의 참석 여부 회신:' in line):
            d = parse_ko_date(line)
            if d and d < prod_date:
                # 생산일 + 14~60일 사이의 임의 날짜로 대체
                new_d = prod_date + timedelta(days=random.randint(14, 60))
                line = KO_DATE_PAT.sub(fmt_ko_date(new_d), line, count=1)
                changed = True

        new_lines.append(line)

    return '\n'.join(new_lines), changed


def fix_all_gongmunso():
    folder = os.path.join(DATASET_DIR, '공문서')
    fixed = 0
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(folder, fname)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        new_content, changed = fix_gongmunso_dates(content)
        if changed:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            fixed += 1
    print(f'✅ 공문서: {fixed}개 파일 날짜 수정 완료')


# ─────────────────────────────────────────────
# 4. 공시자료 TXT: 분기/제출일, 행정구역, 보고서 타입 수정
# ─────────────────────────────────────────────

# 잘못된 행정구역 조합 → 올바른 조합으로 매핑
INVALID_ADDR_FIXES = {
    '광주광역시 영등포구': '서울특별시 영등포구',
    '대전광역시 강남구':   '서울특별시 강남구',
    '부산광역시 강남구':   '서울특별시 강남구',
    '인천광역시 영등포구': '서울특별시 영등포구',
    '울산광역시 영등포구': '서울특별시 영등포구',
    '대구광역시 영등포구': '서울특별시 영등포구',
    '대구광역시 강남구':   '서울특별시 강남구',
    '광주광역시 강남구':   '서울특별시 강남구',
    '인천광역시 강남구':   '서울특별시 강남구',
    '울산광역시 강남구':   '서울특별시 강남구',
    '부산광역시 영등포구': '서울특별시 영등포구',
    '대전광역시 영등포구': '서울특별시 영등포구',
    '광주광역시 서초구':   '서울특별시 서초구',
    '인천광역시 서초구':   '서울특별시 서초구',
    '울산광역시 서초구':   '서울특별시 서초구',
}

# 분기 → 해당 분기 종료 후 제출 가능한 최소 날짜
QUARTER_SUBMIT_AFTER = {
    '1분기': lambda y: date(y, 4, 1),
    '2분기': lambda y: date(y, 7, 1),
    '3분기': lambda y: date(y, 10, 1),
    '4분기': lambda y: date(y + 1, 1, 15),
    '상반기': lambda y: date(y, 7, 1),
    '하반기': lambda y: date(y + 1, 1, 15),
}

def fix_gongsijaryeo(text):
    changed = False

    # 행정구역 오류 수정
    for wrong, correct in INVALID_ADDR_FIXES.items():
        if wrong in text:
            text = text.replace(wrong, correct)
            changed = True

    # 보고서 타입/기간 불일치 수정 (반기보고서 + 분기 → 분기보고서로 통일)
    if '반기보고서' in text:
        # 사업연도에서 분기 키워드 찾기
        quarter_match = re.search(r'사업연도:\s*(\d{4})년\s*(1분기|2분기|3분기|4분기)', text)
        if quarter_match:
            text = text.replace('반기보고서', '분기보고서', 1)
            changed = True

    # 분기/제출일 시계열 수정
    period_match = re.search(
        r'사업연도:\s*(\d{4})년\s*(1분기|2분기|3분기|4분기|상반기|하반기)\s+제출일:\s*(\d{4})년\s*(\d{2})월\s*(\d{2})일',
        text
    )
    if period_match:
        year = int(period_match.group(1))
        period = period_match.group(2)
        sub_year = int(period_match.group(3))
        sub_month = int(period_match.group(4))
        sub_day = int(period_match.group(5))
        sub_date = date(sub_year, sub_month, sub_day)
        min_date = QUARTER_SUBMIT_AFTER.get(period, lambda y: date(y, 1, 1))(year)

        if sub_date < min_date:
            # 제출일을 분기 종료 후 30~60일 내로 수정
            new_sub = min_date + timedelta(days=random.randint(5, 45))
            old_str = f'{sub_year}년 {sub_month:02d}월 {sub_day:02d}일'
            new_str = fmt_ko_date(new_sub)
            # 제출일 부분만 교체 (첫 번째 날짜만)
            text = text.replace(
                f'사업연도: {year}년 {period}  제출일: {old_str}',
                f'사업연도: {year}년 {period}  제출일: {new_str}',
                1
            )
            changed = True

    return text, changed


def fix_all_gongsijaryeo():
    folder = os.path.join(DATASET_DIR, '공시자료')
    fixed = 0
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(folder, fname)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        new_content, changed = fix_gongsijaryeo(content)
        if changed:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            fixed += 1
    print(f'✅ 공시자료: {fixed}개 파일 수정 완료')


# ─────────────────────────────────────────────
# 5. 메뉴얼 TXT: 전화번호 마스킹
# ─────────────────────────────────────────────
PHONE_PAT = re.compile(r'(☎\s*)(\d{2,3})-(\d{3,4})-(\d{4})')

def mask_phones(text):
    new_text = PHONE_PAT.sub(r'\g<1>\2-000-0000', text)
    return new_text, new_text != text


def fix_all_manyual():
    folder = os.path.join(DATASET_DIR, '메뉴얼')
    fixed = 0
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(folder, fname)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        new_content, changed = mask_phones(content)
        if changed:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            fixed += 1
    print(f'✅ 메뉴얼: {fixed}개 파일 전화번호 마스킹 완료')


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(42)
    fix_manifest()
    fix_readme()
    fix_all_gongmunso()
    fix_all_gongsijaryeo()
    fix_all_manyual()
    print('\n✅ 전체 수정 완료')
