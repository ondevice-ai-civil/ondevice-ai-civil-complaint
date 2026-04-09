# GitHub Copilot PR Review Instructions

## Review Tag System

Every review comment must begin with one of the following tags.

| Tag | Criteria | Comment Required |
|-----|----------|-----------------|
| `[MUST]` | Bugs, security vulnerabilities, data loss risk, runtime errors | **Always required** |
| `[SHOULD]` | Code quality issues, maintainability problems, performance issues, potential regressions | **Always required** |
| `[NITS]` | Style, naming, comment wording, minor formatting | **Do not write** |
| `[QUESTION]` | Questions to understand intent | Write when necessary |

## Review Criteria

### Report (`[MUST]` / `[SHOULD]` tags)

- `[MUST]`: Bugs, security vulnerabilities, data loss risk, runtime errors
- `[SHOULD]`: Code quality issues, maintainability problems, performance issues, potential regressions

### Do Not Report

- `[NITS]`-level: Style preferences, variable/method naming preferences, comment wording, minor formatting
- Preference-based suggestions on already-working code
- Pointing out code that is consistent with existing codebase patterns

## When to Skip Review

If there are no `[MUST]` or `[SHOULD]` level issues, **do not write any review comments**.
