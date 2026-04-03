# DORA 메트릭

GovOn은 아직 MVP 구축 단계이므로 DORA는 운영 자동화보다 품질 게이트 정렬을 먼저 끝낸 뒤 수집 범위를 넓혀야 한다.

## 우선순위

1. 실패한 PR 게이트를 우회한 merge와 배포를 막는다.
2. required checks와 branch protection을 저장소 설정에 연결한다.
3. 그 다음부터 deployment frequency, change failure rate, MTTR를 안정적으로 측정한다.

## 현재 해석 기준

- Deployment frequency: `Docker Publish` 또는 수동 Cloud Run deploy 횟수
- Lead time for changes: PR 생성부터 `PR Gate` 통과 및 merge까지의 시간
- Change failure rate: publish 이후 rollback 또는 hotfix가 필요한 비율
- MTTR: demo/runtime 장애 감지부터 복구까지 걸린 시간

## 메모

- 10분 단위 운영 모니터링은 local-install 중심 MVP에는 과하다.
- PR Gate와 deploy lane이 먼저 정리되어야 DORA 숫자도 왜곡되지 않는다.
