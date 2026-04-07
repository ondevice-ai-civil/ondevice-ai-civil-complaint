"""
CodeRabbit 2차 리뷰 반영: RAG 테스트 데이터셋 추가 품질 수정

수정 항목:
1. 공문서 - 담당자 실명/전화번호 마스킹, 중복 표현 제거, 본문 시행일↔추진기간 불일치
2. 공시자료 - 추가 잘못된 행정구역, 기업명↔업종 불일치, 제출일 최대 범위 초과
3. 메뉴얼   - 비밀번호 재사용 유도 보안 문구 수정
4. README.md - 모순된 합성데이터 면책 문구 통일
"""

import re
import os
import random
from datetime import date, timedelta

DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'rag_test_dataset')
)

KO_DATE_PAT = re.compile(r'(\d{4})년\s*(\d{2})월\s*(\d{2})일')

def fmt_ko_date(d):
    return f'{d.year}년 {d.month:02d}월 {d.day:02d}일'

def parse_ko_date(s):
    m = KO_DATE_PAT.search(s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


# ─────────────────────────────────────────────
# 1. 공문서: 담당자/연락처 마스킹 + 중복표현 + 시행일 불일치
# ─────────────────────────────────────────────
# 담당자 패턴: "담당자: 홍길동  연락처: 02-123-4567"
MANAGER_PAT = re.compile(
    r'(담당자:\s*)([가-힣]{2,4})(\s+연락처:\s*)\d{2,3}-\d{3,4}-\d{4}'
)

def fix_gongmunso(text):
    changed = False

    # 담당자 실명 + 연락처 마스킹
    new_text = MANAGER_PAT.sub(r'\g<1>[담당자]\g<3>02-000-0000', text)
    if new_text != text:
        changed = True
        text = new_text

    # 중복 표현 제거 (X 추진 추진 → X 추진, X 계획 추진 계획서 → X 추진 계획서)
    before = text
    text = re.sub(r'(\S+)\s+(추진|계획)\s+\1', r'\1', text)
    text = re.sub(r'(추진)\s+(추진)', r'\1', text)
    text = re.sub(r'(계획)\s+(추진 계획서)', r'추진 계획서', text)
    if text != before:
        changed = True

    # 본문 시행일과 추진기간 시작일이 불일치하는 경우:
    # "YYYY년 MM월 DD일부터 본 부처에서 시행" 날짜를 추진기간 시작일로 통일
    lines = text.split('\n')
    impl_start = None
    for line in lines:
        if '추진 기간:' in line:
            ds = KO_DATE_PAT.findall(line)
            if ds:
                impl_start = date(int(ds[0][0]), int(ds[0][1]), int(ds[0][2]))
            break

    if impl_start:
        new_lines = []
        for line in lines:
            if '부터 본 부처에서 시행' in line:
                m = KO_DATE_PAT.search(line)
                if m:
                    body_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    if body_date != impl_start:
                        line = line.replace(
                            f'{m.group(1)}년 {m.group(2)}월 {m.group(3)}일',
                            fmt_ko_date(impl_start), 1
                        )
                        changed = True
            new_lines.append(line)
        text = '\n'.join(new_lines)

    return text, changed


def fix_all_gongmunso():
    folder = os.path.join(DATASET_DIR, '공문서')
    fixed = 0
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(folder, fname)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        new_content, changed = fix_gongmunso(content)
        if changed:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            fixed += 1
    print(f'✅ 공문서: {fixed}개 파일 수정 완료 (담당자 마스킹·중복표현·시행일)')


# ─────────────────────────────────────────────
# 2. 공시자료: 추가 행정구역, 기업↔업종, 제출일 최대 범위
# ─────────────────────────────────────────────
# 추가 잘못된 행정구역 매핑
ADDR_FIXES = {
    # 이미 1차에서 처리된 것들 (혹시 남은 것 대비)
    '광주광역시 영등포구': '서울특별시 영등포구',
    '대전광역시 강남구':   '서울특별시 강남구',
    '부산광역시 강남구':   '서울특별시 강남구',
    '인천광역시 영등포구': '서울특별시 영등포구',
    '울산광역시 영등포구': '서울특별시 영등포구',
    '울산광역시 강남구':   '서울특별시 강남구',
    '인천광역시 강남구':   '서울특별시 강남구',
    '부산광역시 영등포구': '서울특별시 영등포구',
    '대구광역시 강남구':   '서울특별시 강남구',
    '대구광역시 영등포구': '서울특별시 영등포구',
    '대전광역시 영등포구': '서울특별시 영등포구',
    '광주광역시 서초구':   '서울특별시 서초구',
    # 2차 신규
    '수원시 종로구':          '경기도 수원시 팔달구',
    '광주광역시 종로구':      '광주광역시 동구',
    '창원시 중구':            '경상남도 창원시 의창구',
    '세종특별자치시 중구':    '세종특별자치시 조치원읍',
    '세종특별자치시 강남구':  '세종특별자치시 조치원읍',
    '세종특별자치시 영등포구':'세종특별자치시 조치원읍',
}

# 기업명 → 올바른 업종 설명
COMPANY_INDUSTRY = {
    '한국가스공사':   '가스 공급 및 에너지 관련 제품 및 서비스 제공',
    'GS칼텍스㈜':    '석유 정제 및 윤활유 관련 제품 및 서비스 제공',
    '셀트리온㈜':    '바이오의약품 연구개발 및 제조·판매',
    'LG전자㈜':      '가전·전자 제품 및 솔루션 관련 제품 및 서비스 제공',
    '롯데쇼핑㈜':    '유통·소매·백화점 관련 제품 및 서비스 제공',
    'SK하이닉스㈜':  '반도체 메모리 관련 제품 및 서비스 제공',
    '우리금융지주㈜':'금융서비스 관련 제품 및 서비스 제공',
    'KB금융지주㈜':  '금융서비스 관련 제품 및 서비스 제공',
    '현대자동차㈜':  '자동차 및 부품 관련 제품 및 서비스 제공',
    '삼성전자㈜':    '전자·반도체·가전 관련 제품 및 서비스 제공',
    'POSCO홀딩스㈜': '철강 관련 제품 및 서비스 제공',
    '한국전력공사':  '전력 생산 및 송배전 관련 서비스 제공',
    'KT㈜':          '유무선 통신 서비스 관련 제품 및 서비스 제공',
    'SK텔레콤㈜':    '무선통신 서비스 관련 제품 및 서비스 제공',
    '카카오㈜':      '인터넷·플랫폼 관련 제품 및 서비스 제공',
    '네이버㈜':      '인터넷·검색 플랫폼 관련 제품 및 서비스 제공',
    '현대건설㈜':    '건설 및 엔지니어링 관련 제품 및 서비스 제공',
    'CJ제일제당㈜':  '식품 및 바이오 관련 제품 및 서비스 제공',
    '한화솔루션㈜':  '화학·에너지·방산 관련 제품 및 서비스 제공',
    'HD현대㈜':      '중공업·해양 관련 제품 및 서비스 제공',
    '삼성물산㈜':    '건설·상사·패션 관련 제품 및 서비스 제공',
    'LG화학㈜':      '화학·배터리 관련 제품 및 서비스 제공',
    'SK이노베이션㈜':'에너지·화학 관련 제품 및 서비스 제공',
}

# 분기별 제출일 범위 (최소, 최대)
QUARTER_RANGE = {
    '1분기': (lambda y: date(y, 4, 1),  lambda y: date(y, 5, 31)),
    '2분기': (lambda y: date(y, 7, 1),  lambda y: date(y, 8, 31)),
    '3분기': (lambda y: date(y, 10, 1), lambda y: date(y, 11, 30)),
    '4분기': (lambda y: date(y+1, 1, 15), lambda y: date(y+1, 3, 31)),
    '상반기': (lambda y: date(y, 7, 1),  lambda y: date(y, 9, 30)),
    '하반기': (lambda y: date(y+1, 1, 15), lambda y: date(y+1, 3, 31)),
}

# 업종 변형 패턴 (기업에 맞지 않는 업종들 → 실제 업종으로 교체)
WRONG_INDUSTRY_PATTERNS = [
    r'금융서비스 관련 제품 및 서비스 제공',
    r'자동차 관련 제품 및 서비스 제공',
    r'반도체 관련 제품 및 서비스 제공',
    r'화학 관련 제품 및 서비스 제공',
    r'에너지 관련 제품 및 서비스 제공',
    r'IT 서비스 관련 제품 및 서비스 제공',
    r'건설 관련 제품 및 서비스 제공',
    r'식품 관련 제품 및 서비스 제공',
]


def fix_gongsijaryeo(text):
    changed = False

    # 행정구역 오류 수정 (전체 매핑)
    for wrong, correct in ADDR_FIXES.items():
        if wrong in text:
            text = text.replace(wrong, correct)
            changed = True

    # 기업명↔업종 불일치 수정
    for company, correct_industry in COMPANY_INDUSTRY.items():
        if company in text:
            # 현재 업종 라인 찾기
            for wrong_pattern in WRONG_INDUSTRY_PATTERNS:
                if re.search(wrong_pattern, text):
                    # 해당 업종이 이 기업의 올바른 업종이 아닌 경우에만 교체
                    if wrong_pattern.replace(r'관련 제품 및 서비스 제공', '').strip().rstrip('\\') not in correct_industry:
                        text = re.sub(wrong_pattern, correct_industry, text, count=1)
                        changed = True
                        break

    # 분기/제출일 범위 초과 수정 (최소~최대 범위 벗어난 경우)
    m = re.search(
        r'사업연도:\s*(\d{4})년\s*(1분기|2분기|3분기|4분기|상반기|하반기)\s+제출일:\s*(\d{4})년\s*(\d{2})월\s*(\d{2})일',
        text
    )
    if m:
        year = int(m.group(1))
        period = m.group(2)
        sub_year = int(m.group(3))
        sub_month = int(m.group(4))
        sub_day = int(m.group(5))
        try:
            sub_date = date(sub_year, sub_month, sub_day)
        except ValueError:
            sub_date = None

        if sub_date and period in QUARTER_RANGE:
            min_fn, max_fn = QUARTER_RANGE[period]
            min_date = min_fn(year)
            max_date = max_fn(year)

            if sub_date < min_date or sub_date > max_date:
                new_sub = min_date + timedelta(days=random.randint(5, 40))
                old_str = f'{sub_year}년 {sub_month:02d}월 {sub_day:02d}일'
                new_str = fmt_ko_date(new_sub)
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
    print(f'✅ 공시자료: {fixed}개 파일 수정 완료 (행정구역·업종·제출일)')


# ─────────────────────────────────────────────
# 3. 메뉴얼: 비밀번호 재사용 유도 문구 수정
# ─────────────────────────────────────────────
def fix_manual_security(text):
    old = 'ID/PW는 행정전산망 계정과 동일하게 사용'
    new = 'ID는 행정전산망 계정을 사용하되, 비밀번호는 시스템별 고유하게 설정'
    if old in text:
        return text.replace(old, new), True
    return text, False


def fix_all_manual():
    folder = os.path.join(DATASET_DIR, '메뉴얼')
    fixed = 0
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(folder, fname)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        new_content, changed = fix_manual_security(content)
        if changed:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            fixed += 1
    print(f'✅ 메뉴얼: {fixed}개 파일 보안 문구 수정 완료')


# ─────────────────────────────────────────────
# 4. README.md: 모순 면책 문구 통일
# ─────────────────────────────────────────────
def fix_readme():
    path = os.path.join(DATASET_DIR, 'README.md')
    with open(path, encoding='utf-8') as f:
        content = f.read()

    # 모순된 두 줄 → 하나로 통일
    old = (
        '- 실제 기관명, 인명, 연락처 등은 테스트 목적으로만 사용된 것입니다.\n'
        '- 모든 전화번호·인명·주소는 합성 데이터이며 실존 인물·기관과 무관합니다.'
    )
    new = '- 모든 기관명, 인명, 전화번호, 주소는 합성 데이터이며 실존 인물·기관과 무관합니다.'
    if old in content:
        content = content.replace(old, new)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print('✅ README.md: 면책 문구 통일 완료')
    else:
        print('ℹ️  README.md: 이미 수정된 상태')


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(99)
    fix_all_gongmunso()
    fix_all_gongsijaryeo()
    fix_all_manual()
    fix_readme()
    print('\n✅ 2차 전체 수정 완료')
