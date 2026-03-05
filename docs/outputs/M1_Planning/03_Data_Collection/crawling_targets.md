# Crawling Targets: 지자체 공개민원 게시판 목록

본 문서는 QLoRA 파인튜닝을 위한 민원-답변 데이터 수집 대상 지자체 및 게시판 목록을 정의합니다.

## 1. 광역자치단체 (Major Cities & Provinces)

| 지자체명 | 게시판 이름 | 메인 URL | 크롤링 대상 URL (공개목록) | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **서울특별시** | 응답소 민원사례 | [링크](https://eungdapso.seoul.go.kr/) | [바로가기](https://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lst.jsp) | 민원사례 중심, 데이터 품질 우수 |
| **부산광역시** | 민원처리 확인 | [링크](https://www.busan.go.kr/minwon/) | [바로가기](https://www.busan.go.kr/minwon/mhconfirm01) | 공개 설정된 민원 중심 수집 |
| **경기도** | 민원상담 결과 | [링크](https://www.gg.go.kr/) | [바로가기](https://www.gg.go.kr/contents/contents.do?ciIdx=611) | 상담 사례 위주 |
| **대구광역시** | 민원처리 공개 | [링크](https://dudeuriso.daegu.go.kr/) | [바로가기](https://minwon.daegu.go.kr/) | 두드리소 시스템 기반 |
| **인천광역시** | 민원상담 | [링크](https://www.incheon.go.kr/) | [바로가기](https://www.incheon.go.kr/minwon/MW030101) | 국민신문고 연계 형태 |
| **대전광역시** | 민원처리 공개 | [링크](https://www.daejeon.go.kr/) | [바로가기](https://www.daejeon.go.kr/drh/drhContentsHtmlView.do?menuSeq=151) | - |
| **광주광역시** | 시민의 소리 | [링크](https://www.gwangju.go.kr/) | [바로가기](https://www.gwangju.go.kr/contentsView.do?menuId=gwangju0301020000) | - |
| **울산광역시** | 민원목록 조회 | [링크](https://www.ulsan.go.kr/) | [바로가기](https://www.ulsan.go.kr/u/ulsan/contents.ulsan?mId=001003001002000000) | - |

## 2. 주요 API 엔드포인트 및 대규모 데이터 소스

100,000건 이상의 고품질 데이터 확보를 위해 다음 API 및 데이터셋을 우선적으로 활용합니다.

| 소스명 | 제공 형태 | 예상 데이터 규모 | 주요 엔드포인트 / URL | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **AI Hub** | 데이터셋(Download) | **50만 건+** | [민원 질의-응답 데이터](https://aihub.or.kr/) | 학습용 정제 완료, 최우선 활용 |
| **서울 열린데이터 광장** | Open API (JSON) | **10만 건+** | `http://openAPI.seoul.go.kr:8088/` | `S_EUNGDAPSO_CASE_INFO` 등 |
| **공공데이터포털** | Open API (REST) | **100만 건+** | `http://apis.data.go.kr/` | 전국 지자체 통합 민원 분석 정보 |

## 3. 수집 가능성 분석 (100,000건+ Target)

- **정적 데이터셋 (AI Hub)**: 이미 비식별화 및 Q&A 쌍 구성이 완료된 50만 건 이상의 데이터를 즉시 확보 가능하여 **가장 안정적인 소스**임.
- **실시간 API (Seoul/Data.go.kr)**: 최근 1~2년 내의 최신 민원 트렌드를 반영하기 위해 사용. API 호출당 1,000건 제한이 있으므로 `civil-complaint-crawler`의 반복 호출 로직 활용 필요.
- **결론**: AI Hub 데이터를 베이스라인(80%)으로 삼고, Open API 수집 데이터(20%)를 추가하여 최신성을 보완함으로써 10만 건 이상의 목표치 달성이 충분히 가능함.

## 4. 수집 데이터 필드 (Data Fields)

모든 타겟에서 다음 필드를 공통적으로 수집하는 것을 목표로 합니다.

1.  **ID**: 게시글 고유 번호
2.  **Category**: 민원 분류 (예: 도로, 교통, 환경 등)
3.  **Title**: 민원 제목
4.  **Question (Body)**: 민원 상세 내용
5.  **Answer**: 담당 부서의 공식 답변
6.  **Department**: 담당 부서명
7.  **Date**: 작성일 및 처리일

## 3. 기술적 분석 및 대응 전략

### 3.1 공통 구조 분석
- **Paging**: 대부분 `pageIndex` 또는 `currPage` 파라미터를 사용한 GET 요청 방식.
- **Detail View**: 게시글 클릭 시 상세 페이지로 이동하며, `seq` 또는 `nttId`와 같은 고유 ID를 인자로 가짐.
- **Authentication**: 상당수 지자체가 상세 내용 확인 시 실명인증을 요구하나, '공개민원' 메뉴에서는 인증 없이 접근 가능한 사례 위주로 수집.

### 3.2 크롤링 도구 및 라이브러리
- **Python + BeautifulSoup4**: 정적 페이지 크롤링에 최우선 사용.
- **Selenium/Playwright**: 상세 내용이 JavaScript로 렌더링되거나 클릭 이벤트가 필요한 경우 보조적으로 사용.
- **Requests**: API 엔드포인트가 노출된 경우 직접 호출하여 속도 최적화.

### 3.3 주의사항
- **개인정보 보호**: 수집 직후 PII(개인정보) 탐지 로직을 거쳐 마스킹 처리 필수.
- **Robots.txt 준수**: 각 지자체의 `robots.txt` 설정을 확인하고, `User-Agent` 및 `Delay` 설정 준수.
- **데이터 품질**: 답변이 없는 단순 민원이나, 20자 미만의 짧은 텍스트는 수집 단계에서 필터링.

## 4. API 인증키 및 데이터셋 발급 안내

데이터 수집을 위해 다음 플랫폼에서 인증키(API Key) 발급 및 활용 신청이 필요합니다.

### 4.1 서울 열린데이터 광장 (Seoul Open Data)
- **대상**: 서울시 응답소 민원사례, 서울시 자치구별 민원 통계 등
- **URL**: [https://data.seoul.go.kr/](https://data.seoul.go.kr/)
- **발급 절차**:
    1. 회원가입 및 로그인
    2. 데이터셋 검색 (예: `응답소 민원사례`)
    3. 상세 페이지 하단 **[인증키 신청]** 클릭
    4. 활용 목적(AI 모델 학습 등) 및 URL(`localhost` 등) 입력 후 즉시 발급
    5. **마이페이지 > 인증키 관리**에서 일반 인증키 확인

### 4.2 공공데이터포털 (data.go.kr)
- **대상**: 전국 지자체 통합 민원 분석 정보, 특정 지자체 민원 처리 현황 API
- **URL**: [https://www.data.go.kr/](https://www.data.go.kr/)
- **발급 절차**:
    1. 회원가입 및 로그인
    2. 데이터셋 검색 (예: `민원분석정보`)
    3. **오픈 API** 탭에서 **[활용신청]** 버튼 클릭
    4. 활용 목적 및 동의사항 체크 후 신청 (대부분 자동 승인)
    5. **마이페이지 > 오픈API > 개발계정**에서 인증키(Encoding/Decoding) 확인

### 4.3 AI Hub (대용량 데이터셋)
- **대상**: 민원 질의-응답 데이터, 용도별 목적대화 데이터 (정제 완료된 JSON/CSV)
- **URL**: [https://aihub.or.kr/](https://aihub.or.kr/)
- **확보 절차**:
    1. 회원가입 및 로그인
    2. 데이터셋 검색 (예: `민원 질의`) 후 **[데이터 활용 신청]**
    3. 승인 확인 후 전용 다운로더(Agent) 설치
    4. 대용량 압축 파일(ZIP/JSON) 일괄 다운로드 및 로컬 정제

## 5. 향후 계획
- [ ] 각 타겟별 세부 HTML Selector 분석 완료 (W3)
- [x] 크롤러 프로토타입 개발 및 샘플링 테스트 (W3)
- [ ] 전체 데이터 수집 및 비식별화 (W4)
