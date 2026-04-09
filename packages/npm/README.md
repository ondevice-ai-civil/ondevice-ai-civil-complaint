# govon (npm)

npm wrapper for the [GovOn](https://github.com/umyunsang/GovOn) CLI.

GovOn은 행정 지원 및 민원 처리 워크플로우를 위한 셸 퍼스트 로컬 에이전트 런타임입니다.

## 요구 사항

- **Node.js** 18 이상
- **Python** 3.10 이상
- **pip** (Python 패키지 관리자)

## 설치

```bash
npm install -g govon
```

설치 후 Python 환경이 자동으로 확인됩니다.  
`govon` Python 패키지가 설치되어 있지 않다면 아래 명령어로 설치하세요.

```bash
pip install govon
```

## 사용법

```bash
govon --help
```

## 동작 방식

이 패키지는 Python CLI(`govon`)를 감싸는 thin wrapper입니다.

1. `govon` 명령어 실행 시 Python 3.10+ 설치 여부를 확인합니다.
2. `govon` CLI(`pip install govon`)가 설치되어 있는지 확인합니다.
3. 조건이 충족되면 `child_process.spawn`을 통해 Python CLI에 실행을 위임합니다.
4. 미충족 시 명확한 설치 안내 메시지를 출력하고 종료합니다.

> Python 자동 설치는 보안 및 권한 이슈로 지원하지 않습니다.

## 라이선스

MIT
