# GovOn Homebrew Tap

GovOn을 macOS/Linux에서 Homebrew로 설치하는 방법을 안내합니다.

> **주의**: PyPI 패키지(#499)가 배포된 이후 실제 설치가 가능합니다.
> 현재 formula는 구조 준비 단계이며 `sha256` 및 `url`은 placeholder입니다.

## 설치

```bash
brew tap govon-org/govon
brew install govon
```

## 업그레이드

```bash
brew upgrade govon
```

## 제거

```bash
brew uninstall govon
brew untap govon-org/govon
```

## 수동 설치 (Tap 없이)

```bash
brew install --formula https://raw.githubusercontent.com/GovOn-org/GovOn/main/homebrew/govon.rb
```

## 사용법

설치 후 다음 명령어로 CLI를 실행합니다.

```bash
govon --help
```

## 요구사항

- macOS 12 이상 또는 Linux (glibc 2.17+)
- Python 3.12 (Homebrew가 자동으로 설치)

## 문제 해결

### `govon: command not found`

```bash
brew link govon
```

또는 PATH에 Homebrew bin 경로가 포함되어 있는지 확인하세요.

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

### Formula sha256 오류

PyPI 릴리즈 후 formula가 업데이트되지 않은 경우 아래를 실행하세요.

```bash
brew update
brew upgrade govon
```

## 관련 링크

- [GovOn GitHub](https://github.com/GovOn-org/GovOn)
- [PyPI 패키지](https://pypi.org/project/GovOn/) (배포 예정)
- [이슈 트래커](https://github.com/GovOn-org/GovOn/issues)
