"""
한국 공공기관 실제 문서 크롤러 (RAG 테스트용)

소스:
- 보도자료: korea.kr/briefing/pressReleaseList.do
- 공문서:   korea.kr/archive/expDocMainList.do (정책자료실)
- 메뉴얼:   korea.kr/archive/expDocMainList.do (매뉴얼/지침 필터)
- 공시자료: dart.fss.or.kr 공시목록 → 원문 PDF
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
import urllib3
urllib3.disable_warnings()

BASE = 'https://www.korea.kr'
DART_BASE = 'https://dart.fss.or.kr'
DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'rag_test_dataset')
)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  'Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
}

TARGET = 100   # 카테고리별 목표
DELAY  = 0.5   # 요청 간 딜레이(초)
MIN_SIZE = 2 * 1024       # 2KB 미만 스킵
MAX_SIZE = 50 * 1024 * 1024  # 50MB 초과 스킵


def sess():
    s = requests.Session()
    s.headers.update(HEADERS)
    s.verify = False
    return s


def safe_filename(name, ext=''):
    name = re.sub(r'[\\/:*?"<>|]', '_', name).strip()
    name = name[:100]
    if ext and not name.endswith(ext):
        name += ext
    return name


def get_ext(filename, url=''):
    for src in [filename, url]:
        m = re.search(r'\.(hwp|hwpx|pdf|doc|docx|xls|xlsx|txt|ppt|pptx)$', src, re.I)
        if m:
            return '.' + m.group(1).lower()
    return ''


def download_file(session, url, save_dir, filename_hint=''):
    """파일 URL → 저장 경로. 성공 시 경로 반환, 실패 시 None."""
    try:
        r = session.get(url, timeout=20, stream=True)
        if r.status_code != 200:
            return None

        # Content-Disposition에서 파일명 추출
        cd = r.headers.get('Content-Disposition', '')
        fname = ''
        m = re.search(r"filename\*?=['\"]?(?:UTF-8'')?([^'\";\r\n]+)", cd)
        if m:
            fname = requests.utils.unquote(m.group(1)).strip()

        if not fname:
            fname = filename_hint or os.path.basename(urlparse(url).path) or 'file'

        ext = get_ext(fname, url)
        if not ext:
            ct = r.headers.get('Content-Type', '')
            ext_map = {
                'application/pdf': '.pdf',
                'application/haansofthwp': '.hwp',
                'application/x-hwp': '.hwp',
                'application/vnd.openxmlformats-officedocument.wordprocessingml': '.docx',
                'application/msword': '.doc',
            }
            for mime, e in ext_map.items():
                if mime in ct:
                    ext = e
                    break
            if not ext:
                ext = '.bin'

        fname = safe_filename(fname, ext)
        save_path = os.path.join(save_dir, fname)

        # 중복 파일명 처리
        if os.path.exists(save_path):
            base, e = os.path.splitext(fname)
            i = 1
            while os.path.exists(save_path):
                save_path = os.path.join(save_dir, f'{base}_{i}{e}')
                i += 1

        content = b''
        for chunk in r.iter_content(65536):
            content += chunk
            if len(content) > MAX_SIZE:
                return None

        if len(content) < MIN_SIZE:
            return None

        with open(save_path, 'wb') as f:
            f.write(content)
        return save_path

    except Exception as e:
        return None


# ─────────────────────────────────────────────
# 보도자료 (korea.kr/briefing)
# ─────────────────────────────────────────────
def collect_pressrelease(session, save_dir, target=TARGET):
    os.makedirs(save_dir, exist_ok=True)
    collected = []
    page = 1

    while len(collected) < target:
        url = f'{BASE}/briefing/pressReleaseList.do?pageIndex={page}'
        try:
            r = session.get(url, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
        except Exception:
            break

        # 보도자료 상세 링크 추출
        detail_links = list(set(
            a.get('href', '') for a in soup.find_all('a')
            if 'pressReleaseView.do' in a.get('href', '') and 'newsId' in a.get('href', '')
        ))

        if not detail_links:
            break

        for detail_url in detail_links:
            if len(collected) >= target:
                break
            full_url = urljoin(BASE, detail_url)
            try:
                dr = session.get(full_url, timeout=10)
                dsoup = BeautifulSoup(dr.text, 'html.parser')
                time.sleep(DELAY)
            except Exception:
                continue

            # 첨부파일 다운로드 링크 찾기
            for a in dsoup.find_all('a', href=True):
                href = a.get('href', '')
                if 'download.do' not in href:
                    continue
                fname = a.get_text(strip=True)
                ext = get_ext(fname)
                if ext in ('.hwp', '.hwpx', '.pdf', '.docx', '.doc'):
                    dl_url = urljoin(BASE, href)
                    path = download_file(session, dl_url, save_dir, fname)
                    if path:
                        collected.append(path)
                        print(f'  [보도자료 {len(collected):3d}] {os.path.basename(path)}')
                        if len(collected) >= target:
                            break
                    time.sleep(DELAY)

        page += 1
        time.sleep(DELAY)

    return collected


# ─────────────────────────────────────────────
# 정책자료실 (공문서 / 메뉴얼)
# ─────────────────────────────────────────────
def collect_expdoc(session, save_dir, keyword='', target=TARGET):
    """
    keyword: 검색어 (예: '메뉴얼', '지침', '훈령', '' = 전체)
    """
    os.makedirs(save_dir, exist_ok=True)
    collected = []
    page = 1

    while len(collected) < target:
        params = f'pageIndex={page}'
        if keyword:
            params += f'&srchWord={keyword}'
        url = f'{BASE}/archive/expDocMainList.do?{params}'
        try:
            r = session.get(url, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
        except Exception:
            break

        detail_links = list(set(
            a.get('href', '') for a in soup.find_all('a')
            if 'expDocView.do' in a.get('href', '') and 'docId' in a.get('href', '')
        ))

        if not detail_links:
            break

        for detail_url in detail_links:
            if len(collected) >= target:
                break
            full_url = urljoin(BASE, detail_url)
            try:
                dr = session.get(full_url, timeout=10)
                dsoup = BeautifulSoup(dr.text, 'html.parser')
                time.sleep(DELAY)
            except Exception:
                continue

            for a in dsoup.find_all('a', href=True):
                href = a.get('href', '')
                if 'download.do' not in href:
                    continue
                fname = a.get_text(strip=True)
                if fname in ('내려받기', '다운로드', ''):
                    fname = ''
                ext = get_ext(fname)
                if ext in ('.hwp', '.hwpx', '.pdf', '.docx', '.doc', '.txt'):
                    dl_url = urljoin(BASE, href)
                    path = download_file(session, dl_url, save_dir, fname)
                    if path:
                        collected.append(path)
                        cat = os.path.basename(save_dir)
                        print(f'  [{cat} {len(collected):3d}] {os.path.basename(path)}')
                        if len(collected) >= target:
                            break
                    time.sleep(DELAY)

        page += 1
        time.sleep(DELAY)

    return collected


# ─────────────────────────────────────────────
# 공시자료 (DART)
# ─────────────────────────────────────────────
def collect_dart(session, save_dir, target=TARGET):
    os.makedirs(save_dir, exist_ok=True)
    collected = []
    page = 1

    while len(collected) < target:
        # 공시 목록
        url = f'{DART_BASE}/dsac001/mainAll.do?selectKey=&autoSearch=false'
        try:
            r = session.get(url, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
        except Exception:
            break

        rcp_links = list(set(
            a.get('href', '') for a in soup.find_all('a')
            if 'rcpNo=' in a.get('href', '')
        ))

        if not rcp_links:
            break

        for rcp_url in rcp_links:
            if len(collected) >= target:
                break

            rcp_no = re.search(r'rcpNo=(\d+)', rcp_url)
            if not rcp_no:
                continue
            rcp_no = rcp_no.group(1)

            # 공시 원문 뷰어
            view_url = f'{DART_BASE}/dsaf001/main.do?rcpNo={rcp_no}'
            try:
                vr = session.get(view_url, timeout=10)
                vsoup = BeautifulSoup(vr.text, 'html.parser')
                time.sleep(DELAY)
            except Exception:
                continue

            # PDF/HWP 직링크 탐색
            for a in vsoup.find_all('a', href=True):
                href = a.get('href', '')
                ext = get_ext(href)
                if ext in ('.pdf', '.hwp', '.hwpx'):
                    dl_url = urljoin(DART_BASE, href)
                    path = download_file(session, dl_url, save_dir)
                    if path:
                        collected.append(path)
                        print(f'  [공시자료 {len(collected):3d}] {os.path.basename(path)}')
                        break
                    time.sleep(DELAY)

        page += 1
        time.sleep(DELAY * 2)

    return collected


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
if __name__ == '__main__':
    s = sess()

    cats = {
        '보도자료': (collect_pressrelease, {}),
        '공문서':   (collect_expdoc,       {'keyword': ''}),
        '메뉴얼':   (collect_expdoc,       {'keyword': '매뉴얼'}),
        '공시자료': (collect_dart,         {}),
    }

    for cat, (fn, kwargs) in cats.items():
        save_dir = os.path.join(DATASET_DIR, cat)
        print(f'\n▶ {cat} 수집 시작...')
        results = fn(s, save_dir, **kwargs)
        print(f'  → {cat}: {len(results)}개 수집 완료')

    # 결과 요약
    print('\n=== 최종 결과 ===')
    from collections import Counter
    for cat in ['보도자료', '공문서', '메뉴얼', '공시자료']:
        d = os.path.join(DATASET_DIR, cat)
        if os.path.exists(d):
            files = os.listdir(d)
            exts = Counter(os.path.splitext(f)[1].lower() for f in files)
            print(f'{cat}: {len(files)}개  {dict(exts)}')
