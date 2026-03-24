# 프론트엔드 UI 라이브러리 및 CSS 프레임워크 선정 가이드

## 배경

GovOn은 일반 정보 제공형 웹사이트가 아니라, 사용자가 질문하면 실시간으로 답변을 생성하는 챗봇/에이전트형 인터페이스가 핵심인 서비스입니다. 따라서 UI 라이브러리 선정은 단순히 컴포넌트 수가 많은지보다 아래 요구사항을 얼마나 잘 만족하는지가 더 중요합니다.

- 채팅/에이전트 UI 구현 적합성
- 스트리밍 응답 UI 구성 용이성
- 반응형 레이아웃 대응
- 접근성(A11y)
- Figma 기반 디자인 시스템과의 연결성
- 커스터마이징 자유도
- 번들 크기와 개발 성능
- macOS 환경에서의 개발 안정성

특히 macOS에서 잘 돌아가는지를 고려하면, Node/Vite/Next 기반 표준 React 생태계에서 무리 없이 동작하고, 네이티브 의존성이 적으며, 로컬 개발 서버가 무겁지 않은 조합이 유리합니다. 이 관점에서 GovOn에는 `Tailwind CSS + shadcn/ui + Radix UI`가 가장 적합하고, 빠른 디자인-개발 핸드오프가 더 중요하다면 `Chakra UI`가 차선책입니다.

## React 기반 UI 라이브러리 후보 조사

검토 대상:

- MUI
- Ant Design
- Chakra UI
- shadcn/ui
- Radix UI
- Mantine
- Headless UI
- Ark UI
- NextUI(HeroUI)
- Park UI
- DaisyUI

참고:

- `shadcn/ui`는 전통적인 npm 패키지형 UI 라이브러리라기보다 코드를 프로젝트로 가져와 직접 소유하는 방식입니다.
- `Radix UI`는 완성형 디자인 라이브러리라기보다 접근성 좋은 low-level primitive입니다.
- 실제 실무 조합은 보통 `Tailwind CSS + shadcn/ui + Radix UI`처럼 함께 사용합니다.

## 후보별 비교 분석표

| 후보 | 장점 | 단점 | 챗봇/에이전트 UI 적합성 | 디자인 시스템 호환성 | 커스터마이징 | 번들/성능 | macOS 개발 적합성 |
|------|------|------|--------------------------|----------------------|--------------|-----------|-------------------|
| MUI | 컴포넌트 매우 풍부, 문서/생태계 강함, Figma 리소스 좋음 | 기본 Material 느낌이 강함, 브랜드 커스텀 비용 큼, 개발 시 import 패턴 주의 필요 | 중 | 높음 | 중 | 중~무거움 | 양호 |
| Ant Design | 엔터프라이즈용 화면 구성 빠름, 테이블/폼 강함 | 관리자 대시보드 느낌이 강함, GovOn 같은 대화형 제품 톤과 거리 있음, Figma는 공식보다 서드파티 비중 큼 | 낮음~중 | 중 | 중 | 무거운 편 | 양호 |
| Chakra UI | 접근성 좋음, 공식 Figma Kit 있음, 토큰 기반 테마 관리 편함 | 스타일 시스템은 좋지만 GovOn 고유 UI를 크게 만들 때 shadcn보다 자유도는 낮음 | 중~높음 | 높음 | 높음 | 중 | 좋음 |
| shadcn/ui | 오픈 코드, 제품 맞춤형 UI 제작에 최적, 에이전트 UI 설계 자유도 높음 | 팀이 직접 컴포넌트 자산을 관리해야 함, 공식 Figma 키트가 아니라 커뮤니티 중심 | 매우 높음 | 중 | 매우 높음 | 가벼움 | 매우 좋음 |
| Radix UI | 접근성 매우 강함, low-level primitive라 자유도 높음, 필요한 것만 가져가서 가벼움 | 단독으로는 화면을 빨리 만들기 어려움, 스타일링 직접 해야 함 | 매우 높음 | 중 | 매우 높음 | 가벼움 | 매우 좋음 |

## 추가 후보 라이브러리 검토

기존 비교 대상(MUI, Ant Design, Chakra UI, shadcn/ui, Radix UI)만으로는 선정 근거가 다소 제한적일 수 있으므로, 아래 후보도 동일 기준으로 추가 검토합니다.

- Mantine
- Headless UI
- Ark UI
- NextUI(HeroUI)
- Park UI
- DaisyUI

## 추가 후보 비교표

| 후보 | 장점 | 단점 | 챗봇/에이전트 UI 적합성 | 디자인 시스템 호환성 | 커스터마이징 | 번들/성능 | macOS 개발 적합성 |
|------|------|------|--------------------------|----------------------|--------------|-----------|-------------------|
| Mantine | React 전용, 컴포넌트와 Hooks가 풍부함, 문서 품질이 좋음 | 기본 스타일 존재감이 있고 완전 Headless 구조는 아님 | 중~높음 | 중~높음 | 높음 | 중 | 좋음 |
| Headless UI | Tailwind와 궁합이 매우 좋고 Headless 패턴에 충실함 | 제공 범위가 넓지 않고 직접 조립해야 하는 비중이 큼 | 높음 | 중 | 매우 높음 | 가벼움 | 매우 좋음 |
| Ark UI | Chakra 팀이 만든 차세대 Headless UI, 접근성과 상태 관리가 강함 | 상대적으로 생태계와 도입 사례가 적음 | 높음 | 중 | 매우 높음 | 가벼움 | 매우 좋음 |
| NextUI(HeroUI) | Next.js 친화적이고 기본 완성도가 높음 | 제품 고유 톤으로 깊게 커스터마이징할 때 제약이 생길 수 있음 | 중 | 중 | 중~높음 | 중 | 좋음 |
| Park UI | Ark UI 기반, 디자인 토큰 확장성이 좋음 | 바로 쓰기보다 설계와 조합 작업이 필요함 | 높음 | 높음 | 매우 높음 | 가벼움 | 매우 좋음 |
| DaisyUI | Tailwind 플러그인 방식이라 빠르게 시안 제작 가능, Chat bubble 등 대화형 UI 요소가 있음 | 클래스 중심이라 장기적인 디자인 시스템 자산화에는 보완이 필요함 | 중~높음 | 중 | 중 | 가벼움 | 매우 좋음 |

## 프로젝트 요구사항 기준 평가

### 1. 챗봇 UI 적합성

GovOn은 다음 UI 패턴이 중요합니다.

- 채팅 메시지 버블
- 입력창과 멀티라인 prompt textarea
- 스트리밍 중 토큰/문장 단위 응답 표시
- 로딩, 생성 중, 실패, 재시도 상태
- 세션 목록/대화 이력
- 에이전트 사이드바
- 참고 문서/근거 패널
- 피드백 버튼(좋아요/싫어요)

이 기준에서는 다음과 같이 평가할 수 있습니다.

- 최상: `shadcn/ui + Radix UI`
- 차선: `Chakra UI`
- 보류: `MUI`, `Ant Design`

이유:

- 챗봇/에이전트 UI는 CRUD 페이지보다 레이아웃과 상태 표현을 세밀하게 직접 설계해야 합니다.
- `shadcn/ui`는 코드 자체를 가져와 수정하는 방식이라 GovOn 전용 화면에 맞추기 가장 좋습니다.
- `Radix UI`는 Dialog, Tooltip, Tabs, Dropdown, ScrollArea 같은 상호작용 컴포넌트 품질이 좋아 대화형 앱에 잘 맞습니다.

### 2. 반응형 대응

- `Tailwind CSS` 계열은 모바일/태블릿/데스크탑 분기 작성이 직관적입니다.
- `Chakra UI`도 responsive prop이 좋아 반응형이 편합니다.
- `MUI`와 `Ant Design`도 가능하지만, GovOn 같은 커스텀 반응형 제품 UI에서는 상대적으로 덜 유연할 수 있습니다.

평가:

- 상: `Tailwind + shadcn/ui`, `Chakra UI`
- 중상: `MUI`
- 중: `Ant Design`

### 3. 접근성(A11y)

- `Radix UI`는 공식적으로 WAI-ARIA 패턴, 포커스 관리, 키보드 내비게이션을 강하게 지원합니다.
- `Chakra UI`도 접근성을 강하게 가져갑니다.
- `MUI`도 성숙한 접근성 지원이 있습니다.
- `Ant Design`은 가능하지만 GovOn처럼 세밀한 인터랙션을 만들 때 직접 보완할 부분이 더 생길 수 있습니다.

평가:

- 최상: `Radix UI`
- 상: `Chakra UI`, `MUI`
- 중상: `Ant Design`

## 디자인 시스템과의 호환성 검토

### 1. Figma 연계

- `MUI`: 공식 Figma kit와 design resource가 강합니다.
- `Chakra UI`: 공식 Figma Kit가 있습니다.
- `Ant Design`: Figma 리소스는 있으나 공식보다는 서드파티 비중이 큽니다.
- `shadcn/ui`: 공식 코드 문서는 좋지만 Figma는 커뮤니티 키트 중심입니다.
- `Radix UI`: primitive 중심이라 Figma 대응은 팀이 직접 시스템화해야 합니다.

평가:

- 상: `Chakra UI`, `MUI`
- 중: `Ant Design`
- 중~낮음: `shadcn/ui`, `Radix UI` 단독

### 2. GovOn 디자인 시스템 확장성

GovOn은 초기에 빠르게 만드는 것보다, 이후에 아래를 자산화해야 합니다.

- 버튼/입력/카드/사이드바
- 채팅 메시지
- 에이전트 상태 표시
- 근거 문서 패널
- 세션 리스트
- 알림/토스트
- 피드백 UI

이 기준에서는 다음 조합이 가장 유리합니다.

- 가장 유리: `shadcn/ui + Radix UI`
- 차선: `Chakra UI`
- 덜 적합: `MUI`, `Ant Design`

이유:

- GovOn은 범용 디자인 시스템을 그대로 가져다 쓰는 것보다 우리 서비스용 시스템을 키워가는 구조가 더 맞습니다.

## 커스터마이징 용이성 및 번들 크기 비교

### 1. 커스터마이징

- `shadcn/ui`: 최상
  - 컴포넌트 코드를 직접 소유하므로 디자인 시스템화하기 가장 좋습니다.
- `Radix UI`: 최상
  - 스타일이 거의 없고 구조/접근성만 제공해서 완전 자유롭게 입힐 수 있습니다.
- `Chakra UI`: 상
  - 토큰 기반 테마가 좋고 커스텀이 쉽습니다.
- `MUI`: 중상
  - 테마 시스템은 강하지만 기본 Material 성격이 강합니다.
- `Ant Design`: 중
  - 토큰 커스터마이징은 가능하지만 제품 개성 강한 UI로 바꾸는 데 비용이 큽니다.

### 2. 번들 크기/성능

정확한 수치는 버전마다 바뀌므로 최종 도입 직전 Bundlephobia로 다시 확인하는 것이 맞습니다. 다만 구조적으로 보면 다음과 같습니다.

- 가벼운 편: `Radix UI`, `shadcn/ui`
  - 필요한 primitive/컴포넌트만 가져가는 구조
  - 불필요한 런타임 오버헤드가 적음
- 중간: `Chakra UI`
  - 공식 문서도 bundle optimization 가이드를 제공합니다.
- 중~무거움: `MUI`
  - 공식 문서에서 barrel import가 개발 성능에 불리하다고 안내합니다.
- 무거운 편: `Ant Design`
  - 컴포넌트 범위가 넓고 제품 특성상 대형 엔터프라이즈 UI 성격이 강합니다.

### 3. macOS 개발 적합성

여기서 중요한 건 mac에서 설치가 되느냐보다 아래입니다.

- Apple Silicon 환경에서 무거운 dev server가 아닌지
- 네이티브 빌드 의존성이 적은지
- Vite/Next + pnpm 조합에서 무리 없는지
- CSS-in-JS 런타임 부담이 과하지 않은지

실무적으로 보면 다음과 같이 볼 수 있습니다.

- 가장 안정적: `Tailwind CSS + shadcn/ui + Radix UI`
  - 순수 JS/TS 중심
  - Vite/Next와 궁합이 좋음
  - macOS에서 흔한 네이티브 빌드 이슈가 적은 편
- 안정적: `Chakra UI`
- 가능하지만 상대적으로 무거움: `MUI`, `Ant Design`

여기서 "mac에서 잘 돌아간다"는 기준으로는 `shadcn/ui + Radix UI + Tailwind`가 가장 무난합니다. 이건 공식 문서에 mac 전용 최적화가 적혀 있어서가 아니라, 구조적으로 가볍고 표준 React/Vite/Next 흐름에 잘 맞기 때문입니다.

## 추천 결론

### 1순위

`Tailwind CSS + shadcn/ui + Radix UI`

선정 이유:

- GovOn 핵심 요구사항인 챗봇/에이전트 UI와 가장 잘 맞음
- 실시간 스트리밍 응답, 상태 표시, 세션/사이드바 UI를 제품 맞춤형으로 설계하기 좋음
- 접근성 좋은 primitive를 확보하면서도 디자인 자유도가 높음
- 장기적으로 GovOn 고유 디자인 시스템 구축에 유리함
- macOS 개발 환경에서 가볍고 안정적으로 운용하기 좋음

### 2순위

`Chakra UI`

선정 이유:

- 공식 Figma Kit가 있어 디자인-개발 핸드오프가 비교적 쉬움
- 접근성/반응형/토큰 관리가 안정적임
- 빠른 MVP 제작에는 강함

보류 이유:

- GovOn만의 개성 있는 에이전트형 UI를 만들 때 1순위보다 자유도가 낮음

### 비추천 우선순위

- `MUI`: 너무 범용적이고 Material 성격이 강해서 GovOn 톤과 다를 가능성이 큼
- `Ant Design`: 엔터프라이즈 관리도구 성격이 강해 챗봇형 서비스와 톤이 덜 맞음

## 최종 제안 문장

GovOn의 핵심은 사용자의 질의에 실시간으로 응답하는 챗봇/에이전트형 인터페이스이므로, 단순한 범용 컴포넌트 수보다 대화형 UI 설계 자유도, 접근성, 반응형 대응, 디자인 시스템 확장성, macOS 기반 개발 안정성을 우선해야 합니다. 이에 따라 UI 라이브러리 스택은 `Tailwind CSS + shadcn/ui + Radix UI`를 1순위로 채택하고, 디자인-개발 핸드오프 효율을 더 우선할 경우 `Chakra UI`를 차선책으로 검토합니다.

## 참고 링크

- MUI bundle 최적화 가이드: https://mui.com/material-ui/guides/minimizing-bundle-size/
- MUI Figma 리소스: https://mui.com/material-ui/design-resources/material-ui-for-figma/
- Chakra UI Figma Kit: https://chakra-ui.com/docs/get-started/figma
- Chakra UI bundle 최적화: https://chakra-ui.com/guides/component-bundle-optimization
- Ant Design theme customization: https://ant.design/docs/react/customize-theme
- Ant Design resources: https://ant.design/docs/resources/
- Radix Primitives 소개: https://www.radix-ui.com/primitives/docs/overview/introduction
- Radix 접근성: https://www.radix-ui.com/primitives/docs/overview/accessibility
- Radix Themes: https://www.radix-ui.com/themes/docs/overview/getting-started
- shadcn/ui 소개: https://ui.shadcn.com/docs
- shadcn/ui Figma 리소스: https://ui.shadcn.com/docs/figma
- shadcn CLI: https://ui.shadcn.com/docs/cli
