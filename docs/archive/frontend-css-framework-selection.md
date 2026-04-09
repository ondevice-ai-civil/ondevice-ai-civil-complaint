# 프론트엔드 CSS 프레임워크 및 스타일링 스택 선정 가이드

## 배경

GovOn 프론트엔드는 일반적인 정보 제공 페이지보다 챗봇/에이전트형 인터페이스에 더 가깝습니다. 따라서 CSS 스택도 단순히 빠르게 화면을 만드는 용도보다, 다음 조건을 안정적으로 만족하는 방향으로 선택해야 합니다.

- 대화형 UI에 맞는 세밀한 레이아웃 제어
- 스트리밍 응답 상태 표현 및 조건부 스타일링 용이성
- 반응형 웹 대응
- 디자인 시스템 자산화 가능성
- 선택한 UI 라이브러리와의 호환성
- 번들 크기 및 런타임 성능 부담
- macOS 기반 React 개발 환경과의 안정성

특히 GovOn 프론트엔드는 팀 로컬 개발 환경인 macOS에서 안정적으로 실행되고 관리되어야 하므로, Apple Silicon 환경에서 네이티브 빌드 의존성이 과도하지 않고, Vite/Next 기반 React 개발 흐름과 잘 맞으며, 로컬 개발 서버 구성이 무겁지 않은 스택이 유리합니다.

또한 앞선 이슈 `#146`과 PR `#168`에서 GovOn에 적합한 UI 라이브러리 조합을 `Tailwind CSS + shadcn/ui + Radix UI` 중심으로 정리한 만큼, 이번 문서는 그 결론과 충돌하지 않고 자연스럽게 이어지는 CSS 프레임워크 및 보조 라이브러리 스택을 선정하는 것을 목표로 합니다.

## 관련 문서 및 연계

- 관련 이슈: `#147 feat: 오픈소스 CSS 프레임워크 리서치 및 선정`
- 선행 이슈: `#146 feat: 레퍼런스 UI/UX 라이브러리 리서치 및 선정`
- 연계 문서: `docs/frontend-ui-library-selection.md`

이번 문서는 UI 라이브러리 선정 문서의 결론을 전제로 작성합니다. 즉, CSS 스택은 독립적으로 따로 고르는 것이 아니라, `shadcn/ui + Radix UI` 조합을 가장 효율적으로 뒷받침할 수 있는지를 기준으로 판단합니다.

## CSS 프레임워크 후보 조사

검토 대상:

- Tailwind CSS
- UnoCSS
- Bootstrap
- Bulma
- Panda CSS
- styled-components
- Emotion

참고:

- `styled-components`, `Emotion`은 전통적인 CSS 프레임워크라기보다 CSS-in-JS 스타일링 도구에 가깝습니다.
- `Bootstrap`, `Bulma`는 전통적 CSS 프레임워크에 가깝고, `Tailwind CSS`, `UnoCSS`, `Panda CSS`는 utility-first 또는 atomic CSS 계열로 볼 수 있습니다.
- GovOn은 관리자 화면보다 제품형 UI에 가깝기 때문에, 기본 스타일을 많이 덮어써야 하는 도구는 불리할 수 있습니다.

## 스타일링 방식 비교

### 1. Utility-first

대표 후보: `Tailwind CSS`

특징:

- 작은 단위의 utility class를 조합해 UI를 구성
- 반응형, 상태, 다크모드, 조건부 스타일을 한 흐름 안에서 관리하기 쉬움
- shadcn/ui와 결합할 때 가장 자연스러운 방식

GovOn 적합성:

- 챗봇 메시지 버블, 세션 리스트, 사이드바, 스트리밍 상태 배지 등 반복적인 UI 패턴을 빠르게 조합하기 좋습니다.
- 화면 구조를 직접 설계하는 제품형 UI와 잘 맞습니다.
- 디자인 토큰과 조합하면 확장 가능한 디자인 시스템으로 발전시키기 쉽습니다.

### 2. 전통적 CSS 프레임워크

대표 후보: `Bootstrap`, `Bulma`

특징:

- 미리 정해진 레이아웃과 컴포넌트 클래스가 풍부함
- 빠른 마크업에는 강하지만, 제품 고유 톤으로 깊게 바꾸려면 오버라이드가 늘어날 수 있음

GovOn 적합성:

- 랜딩 페이지나 단순 폼에는 빠를 수 있으나, 대화형 앱의 세밀한 상태 표현에는 상대적으로 제약이 큽니다.
- 기본 컴포넌트 룩앤필이 강할수록 GovOn 전용 톤을 만들기 위해 오버라이드 계층이 두꺼워질 가능성이 큽니다.

### 3. CSS-in-JS

대표 후보: `styled-components`, `Emotion`

특징:

- 컴포넌트 단위로 스타일을 코드 안에서 관리
- 동적 스타일링과 테마 관리는 편리하지만, 런타임 비용과 SSR 설정 부담이 생길 수 있음

GovOn 적합성:

- 동적 상태 표현은 편하지만, Tailwind + shadcn/ui 조합과 비교하면 지금 프로젝트의 기준 스택과는 방향이 다릅니다.
- 디자인 시스템 초반 구축 단계에서 오히려 스타일 계층이 복잡해질 수 있습니다.

## CSS-in-JS vs Utility-first vs 전통적 CSS 방식 비교 분석

| 비교 항목 | Utility-first | 전통적 CSS 프레임워크 | CSS-in-JS |
|------|------|------|------|
| 대표 도구 | `Tailwind CSS`, `UnoCSS`, `Panda CSS` | `Bootstrap`, `Bulma` | `styled-components`, `Emotion` |
| 기본 철학 | 작은 단위의 utility를 조합해 직접 UI를 설계 | 미리 준비된 컴포넌트/클래스 규칙을 활용 | 컴포넌트 로직과 스타일을 함께 관리 |
| 초기 개발 속도 | 빠름 | 매우 빠름 | 보통 |
| 제품 고유 UI 자유도 | 매우 높음 | 중 | 높음 |
| 상태 기반 UI 표현 | 매우 쉬움 | 보통 | 쉬움 |
| 디자인 시스템 확장성 | 높음 | 중 | 높음 |
| 런타임 성능 | 유리함 | 유리함 | 상대적으로 불리할 수 있음 |
| shadcn/ui 연계성 | 매우 높음 | 낮음 | 보통 |
| 팀 규칙 일관성 확보 | 토큰/클래스 규칙 정리 시 쉬움 | 기본 규칙은 쉽지만 오버라이드가 누적되기 쉬움 | 패턴을 잘못 열어두면 스타일 분산 가능성 큼 |
| GovOn 적합도 | 가장 높음 | 제한적 | 조건부 적합 |

해석:

- `Utility-first`는 제품 UI를 직접 설계해야 하는 GovOn에 가장 잘 맞습니다. 챗봇 메시지 상태, 패널 전환, 반응형 레이아웃을 한 흐름 안에서 조합하기 쉽고, `shadcn/ui`와도 바로 연결됩니다.
- `전통적 CSS 프레임워크`는 CRUD 화면이나 단순 폼 중심 서비스에는 여전히 효율적이지만, GovOn처럼 고유한 제품 톤과 세밀한 상태 표현이 필요한 경우에는 기본 스타일을 많이 걷어내야 합니다.
- `CSS-in-JS`는 컴포넌트 단위 응집도가 장점이지만, 현재 GovOn의 후보 UI 스택과 조합했을 때 얻는 이점보다 런타임/SSR/스타일 계층 복잡도가 더 크게 작용할 가능성이 있습니다.

## 후보별 비교 분석표

| 후보 | 장점 | 단점 | 반응형 지원 | UI 라이브러리 호환성 | 커스터마이징 | 성능/번들 | macOS 개발 적합성 |
|------|------|------|-------------|----------------------|--------------|-----------|-------------------|
| Tailwind CSS | utility-first 구조, 반응형 클래스 직관적, shadcn/ui와 궁합 좋음 | 클래스 길이가 길어질 수 있고 초기 규칙 정리가 필요함 | 매우 높음 | 매우 높음 | 매우 높음 | 가벼운 편 | 매우 좋음 |
| UnoCSS | Tailwind 유사 사용감, on-demand 생성, preset/variant 확장성 높음 | 팀 표준 자료와 예제가 Tailwind보다 적고 shadcn/ui 기본 전제와는 거리 있음 | 매우 높음 | 높음 | 매우 높음 | 매우 가벼움 | 매우 좋음 |
| Bootstrap | 빠른 화면 제작, 문서와 생태계 성숙 | 기본 스타일 존재감이 강하고 제품 고유 톤 유지가 어려움 | 높음 | 중 | 중 | 중~무거움 | 양호 |
| Bulma | 가볍고 문법이 단순함, mobile-first | 생태계와 조합 폭이 좁고 대화형 제품 UI 확장성은 제한적 | 높음 | 중 | 중 | 가벼움 | 좋음 |
| Panda CSS | 타입 세이프 토큰, zero-runtime 추구, 디자인 시스템 구축 친화적 | 초기 설계 비용이 있고 shadcn/ui와 바로 맞물리지는 않음 | 높음 | 중~높음 | 매우 높음 | 가벼운 편 | 좋음 |
| styled-components | 컴포넌트 단위 스타일 관리, 테마 지원 | 런타임 비용, SSR/툴링 고려 필요 | 높음 | 중~높음 | 높음 | 중 | 양호 |
| Emotion | 유연한 API, css prop 지원, 라이브러리 연동 폭 넓음 | CSS-in-JS 특성상 런타임/설정 복잡도 존재 | 높음 | 높음 | 높음 | 중 | 양호 |

## 프로젝트 요구사항 기준 평가

### 1. 챗봇/에이전트 UI 적합성

GovOn에서 자주 나오는 UI 패턴은 다음과 같습니다.

- 좌우 정렬이 다른 메시지 버블
- 전송 중, 생성 중, 오류 상태 표시
- 참고 문서 패널과 에이전트 사이드바
- 세션 리스트 및 상태 뱃지
- 모바일에서 재배치되는 입력창/패널 구조

이 기준에서는 `Tailwind CSS`가 가장 유리합니다.

이유:

- 작은 유틸리티를 조합해 복잡한 상태를 빠르게 표현할 수 있습니다.
- 스트리밍 UI처럼 상태 변화가 잦은 화면에서 클래스 조건 분기가 자연스럽습니다.
- shadcn/ui 예제와 실무 패턴 상당수가 Tailwind 기반입니다.

### 2. 반응형 대응

- `Tailwind CSS`는 breakpoint prefix 기반 반응형 작성이 매우 직관적입니다.
- `UnoCSS`도 유사한 atomic utility 흐름이라 반응형 작성성이 좋습니다.
- `Bootstrap`도 responsive utility가 잘 갖춰져 있습니다.
- `Bulma`는 mobile-first 기반과 breakpoint mixin이 장점입니다.
- `Panda CSS`는 토큰 중심 설계와 recipe 패턴으로 반응형 일관성을 가져가기 좋습니다.
- `styled-components`, `Emotion`도 가능하지만 반응형 체계를 팀이 직접 강하게 통일해야 합니다.

평가:

- 최상: `Tailwind CSS`
- 상: `UnoCSS`, `Bootstrap`, `Bulma`, `Panda CSS`
- 중상: `Emotion`, `styled-components`

### 3. 디자인 시스템 확장성

GovOn은 단기적으로 페이지 몇 개를 만드는 것보다, 이후에 재사용 가능한 토큰과 패턴을 쌓아야 합니다.

- 버튼, 입력, 카드, 배지
- 채팅 메시지 패턴
- 상태 컬러와 간격 체계
- 패널, 사이드바, 토스트

이 기준에서는 `Tailwind CSS + CSS Variables` 조합이 가장 적합합니다.

이유:

- 토큰은 CSS Variables로 두고, 실제 UI 조합은 Tailwind utility로 구성하면 유지보수와 확장성 균형이 좋습니다.
- shadcn/ui도 같은 접근을 기본 전제로 둡니다.
- `Panda CSS`도 디자인 시스템 관점에서는 경쟁력 있지만, 현재 GovOn이 이미 참조 중인 UI 레퍼런스 및 조합 예시와의 직접 호환성은 Tailwind보다 약합니다.

### 4. 선택한 UI 라이브러리와의 호환성

앞선 이슈 #146 기준 1순위 조합은 `shadcn/ui + Radix UI`입니다.

이 조합과의 호환성은 다음과 같이 평가할 수 있습니다.

- 매우 높음: `Tailwind CSS`
- 높음: `UnoCSS`
- 보통: `Emotion`, `styled-components`, `Panda CSS`
- 낮음~중: `Bootstrap`, `Bulma`

이유:

- shadcn/ui는 Tailwind 기반으로 설계되어 있어 사실상 가장 자연스러운 선택입니다.
- `UnoCSS`는 atomic utility라는 점에서 접근은 유사하지만, shadcn/ui 기본 문서/예제와 1:1로 맞물리는 수준은 아닙니다.
- `Panda CSS`는 설계 철학상 디자인 시스템 친화적이지만, shadcn/ui 생태계의 기본 사용 흐름을 그대로 가져오기는 어렵습니다.
- Bootstrap, Bulma는 자체 스타일 체계가 강해 조합 시 일관성 관리 비용이 커질 수 있습니다.

이 항목은 단순한 기술 선호가 아니라, PR #168에서 이미 합의된 UI 라이브러리 선정 방향과 실제 구현 스택을 일치시키기 위한 기준입니다. UI 라이브러리는 `shadcn/ui + Radix UI`로 정리해 두고, CSS 프레임워크는 별도로 Bootstrap이나 Bulma를 선택하면 문서 간 결론이 충돌하게 됩니다. 따라서 #147의 결론은 #146의 결론을 보강하는 방향이어야 합니다.

## 번들 크기 및 성능 관점 검토

- `Tailwind CSS`는 빌드 시 사용 클래스 중심으로 최적화되는 구조라 런타임 부담이 적습니다.
- `UnoCSS`도 on-demand 생성 방식이라 매우 가볍고 빠른 편입니다.
- `Bootstrap`은 범용 UI 전체를 포함하는 경우 불필요한 스타일까지 같이 들어오기 쉽습니다.
- `Bulma`는 비교적 가볍지만, GovOn에서 필요한 제품형 UI를 만들기 위해 추가 스타일 계층이 늘어날 수 있습니다.
- `Panda CSS`는 zero-runtime 성격이 장점이지만, 초기 토큰/recipe 설계 비용을 별도로 감수해야 합니다.
- `styled-components`, `Emotion`은 동적 스타일링 강점이 있지만 런타임 기반 비용을 고려해야 합니다.

GovOn처럼 스트리밍 UI와 조건부 상태가 많은 인터페이스에서는, CSS 런타임보다는 정적 유틸리티 중심 구성이 더 단순하고 안정적입니다.

## macOS 개발 환경 기준 검토

GovOn 팀의 기본 로컬 개발 환경은 macOS이므로, 다음 기준을 함께 봐야 합니다.

- Apple Silicon 환경에서 네이티브 의존성 이슈가 적은지
- Vite/Next + pnpm 기반 프론트엔드 개발 흐름과 잘 맞는지
- 로컬 개발 서버와 HMR이 무겁지 않은지
- CSS 런타임 오버헤드가 과하지 않은지

이 기준에서는 `Tailwind CSS`가 가장 안정적입니다.

이유:

- 정적 유틸리티 기반이라 런타임 스타일 계산 부담이 적습니다.
- shadcn/ui, Radix UI, Lucide, Motion과 함께 사용할 때 일반적인 React 생태계 흐름과 잘 맞습니다.
- `UnoCSS`도 macOS 개발 환경에서 가볍게 운용할 수 있지만, 팀 온보딩과 레퍼런스 확보 측면에서는 Tailwind보다 보수적인 선택이 아닙니다.
- `Panda CSS`는 타입 안전성과 토큰 설계 측면의 장점이 있으나, 현재 단계의 GovOn보다 한 단계 더 성숙한 디자인 시스템 운영 조직에 적합합니다.
- CSS-in-JS 계열은 나쁜 선택은 아니지만, SSR 설정과 스타일 런타임까지 고려해야 해 운영 복잡도가 더 높아질 수 있습니다.
- Bootstrap, Bulma는 도입은 단순하지만, 결과적으로 GovOn 전용 UI 톤으로 커스터마이징하는 추가 비용이 생깁니다.

## 아이콘 라이브러리 후보 조사

검토 대상:

- Lucide
- Heroicons
- Phosphor Icons

### 비교 요약

| 후보 | 장점 | 단점 | GovOn 적합성 |
|------|------|------|--------------|
| Lucide | 가볍고 일관된 스트로크 스타일, React 사용 편함 | 굵직한 일러스트형 표현은 약함 | 매우 높음 |
| Heroicons | Tailwind 생태계와 잘 맞음, 단순하고 정제된 스타일 | 아이콘 종류 폭은 Lucide/Phosphor보다 좁게 느껴질 수 있음 | 높음 |
| Phosphor Icons | 종류가 풍부하고 스타일 선택 폭 넓음 | 제품 톤 통일을 위해 기준을 강하게 잡아야 함 | 중~높음 |

선정 의견:

- 기본 아이콘 라이브러리는 `Lucide`를 우선 추천합니다.
- Tailwind/shadcn 생태계와 함께 쓸 때 가장 무난하고, 채팅/사이드바/상태 아이콘에 잘 맞습니다.

## 애니메이션 라이브러리 후보 조사

검토 대상:

- Motion
- React Spring

### 비교 요약

| 후보 | 장점 | 단점 | GovOn 적합성 |
|------|------|------|--------------|
| Motion | React UI 애니메이션에 최적화, 레이아웃 전환과 등장 효과 구현이 쉬움 | 과도하게 쓰면 UI가 무거워질 수 있음 | 매우 높음 |
| React Spring | 물리 기반 애니메이션이 강력함 | 단순 UI 전환 기준으로는 Motion보다 진입 비용이 큼 | 중 |

선정 의견:

- GovOn에서는 `Motion`을 우선 추천합니다.
- 페이지 전환, 패널 슬라이드, 메시지 등장, 로딩 상태 전환 등 제품 UI 중심 애니메이션에 더 적합합니다.

## 최종 CSS 스택 제안

### 1순위

`Tailwind CSS + CSS Variables + Lucide + Motion`

선정 이유:

- `shadcn/ui + Radix UI` 조합과 가장 자연스럽게 연결됨
- 이슈 `#146`과 PR `#168`에서 정리한 UI 라이브러리 결론과 충돌하지 않음
- 챗봇/에이전트형 UI에 필요한 세밀한 조합형 레이아웃을 빠르게 구현할 수 있음
- 반응형, 상태 기반 UI, 디자인 토큰 관리에 모두 유리함
- 런타임 스타일링 부담이 적고 macOS 기반 개발 환경에서도 안정적임
- 아이콘과 애니메이션까지 포함한 프론트엔드 기본 스택으로 확장하기 좋음

### 2순위

`UnoCSS + CSS Variables`

선정 이유:

- Tailwind와 유사한 atomic utility 접근을 유지하면서도 매우 가벼운 구성 가능
- 다만 현재 GovOn의 UI 라이브러리 선택 결과와 팀 레퍼런스 측면에서는 Tailwind보다 이점이 명확하지 않음

### 3순위

`Panda CSS + CSS Variables`

선정 이유:

- 장기적으로 강한 디자인 토큰 체계와 zero-runtime 지향 구성이 가능
- 다만 지금 단계에서 GovOn이 바로 선택하기에는 초기 설계 비용이 높고, `shadcn/ui` 중심 흐름과의 정합성도 Tailwind보다 낮음

### 비추천 우선순위

- `Bootstrap`: 기본 스타일이 강해서 GovOn 고유 UI 톤을 만들기 어렵습니다.
- `Bulma`: 단순하고 가볍지만, 장기적인 확장성과 생태계 측면에서 우선순위가 낮습니다.
- `styled-components`: 나쁜 선택은 아니지만, 현재 선택한 UI 라이브러리 방향과는 결이 다릅니다.
- `Emotion`: 유연성은 좋지만, 지금 프로젝트에서는 CSS-in-JS 도입 복잡도를 감수할 실익이 크지 않습니다.

## 최종 제안 문장

GovOn 프론트엔드는 챗봇/에이전트형 인터페이스를 중심으로 구성되므로, CSS 스택 역시 범용 페이지 제작보다 제품 맞춤형 UI 설계 자유도, 반응형 대응, 디자인 시스템 확장성, 성능 안정성, macOS 기반 개발 편의성을 우선해야 합니다. 또한 이 결론은 이슈 `#146` 및 PR `#168`에서 정리한 `shadcn/ui + Radix UI` 중심 방향과 일관되게 이어져야 합니다. 이에 따라 CSS 프레임워크는 `Tailwind CSS`를 1순위로 채택하고, 디자인 토큰은 `CSS Variables`, 아이콘은 `Lucide`, 애니메이션은 `Motion` 조합으로 가져가는 것이 가장 적합합니다.

## 참고 링크

- Tailwind CSS Responsive Design: https://tailwindcss.com/docs/responsive-design
- UnoCSS Guide: https://unocss.dev/guide/
- Bootstrap CSS Variables: https://getbootstrap.com/docs/5.3/customize/css-variables/
- Bulma Documentation: https://bulma.io/documentation/
- Bulma Responsiveness: https://bulma.io/documentation/start/responsiveness/
- Panda CSS Overview: https://panda-css.com/docs/overview/why-panda
- styled-components Tooling / SSR: https://styled-components.com/docs/tooling
- Emotion CSS Package: https://emotion.sh/docs/@emotion/css
- Lucide React Guide: https://lucide.dev/guide/packages/lucide-react
- Heroicons: https://heroicons.com/
- Phosphor Icons React: https://phosphoricons.com/
- Motion for React: https://motion.dev/docs/react
- React Spring: https://react-spring.dev/
