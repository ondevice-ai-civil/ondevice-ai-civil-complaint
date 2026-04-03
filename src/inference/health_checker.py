"""상세 헬스체크.

컴포넌트별 상태를 확인하고 종합 상태를 판정한다:
- 컴포넌트별 상태 확인 (model, faiss_index, bm25_index, database)
- 상세/간략 모드 (detail_level: "minimal" | "detailed")
- 종합 상태 판정 (healthy, degraded, unhealthy)
- 마지막 체크 타임스탬프
- 비동기 체크
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from loguru import logger


class HealthStatus:
    """상태 상수."""

    HEALTHY: str = "healthy"
    DEGRADED: str = "degraded"
    UNHEALTHY: str = "unhealthy"


# 상태 우선순위 (높을수록 심각)
_STATUS_PRIORITY = {
    HealthStatus.HEALTHY: 0,
    HealthStatus.DEGRADED: 1,
    HealthStatus.UNHEALTHY: 2,
}


@dataclass
class ComponentStatus:
    """개별 컴포넌트 상태."""

    name: str
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class OverallStatus:
    """종합 상태."""

    status: str
    timestamp: datetime
    components: Optional[List[ComponentStatus]] = None


# 체크 함수 타입: 인자 없이 호출, ComponentStatus를 반환하는 비동기 함수
CheckFunction = Callable[[], Awaitable[ComponentStatus]]


class HealthChecker:
    """상세 헬스체크기."""

    def __init__(self) -> None:
        self.registered_checks: Dict[str, CheckFunction] = {}
        self.last_check_timestamp: Optional[datetime] = None

    def register_check(self, name: str, check_fn: CheckFunction) -> None:
        """컴포넌트 체크 함수를 등록한다."""
        self.registered_checks[name] = check_fn

    async def check(
        self, detail_level: Literal["minimal", "detailed"] = "minimal"
    ) -> OverallStatus:
        """등록된 컴포넌트를 체크하고 종합 상태를 반환한다."""
        now = datetime.now(timezone.utc)
        self.last_check_timestamp = now

        if not self.registered_checks:
            return OverallStatus(
                status=HealthStatus.HEALTHY,
                timestamp=now,
                components=[],
            )

        component_results: List[ComponentStatus] = []
        worst_status = HealthStatus.HEALTHY

        for name, check_fn in self.registered_checks.items():
            try:
                result = await check_fn()
                component_results.append(result)
            except Exception as exc:
                logger.error(f"헬스체크 실패 [{name}]: {exc}")
                component_results.append(
                    ComponentStatus(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"체크 실패: {exc}",
                    )
                )

        # 종합 상태: 가장 심각한 상태를 채택
        for comp in component_results:
            if _STATUS_PRIORITY.get(comp.status, 2) > _STATUS_PRIORITY.get(worst_status, 0):
                worst_status = comp.status

        if detail_level == "minimal":
            return OverallStatus(
                status=worst_status,
                timestamp=now,
                components=[],
            )

        return OverallStatus(
            status=worst_status,
            timestamp=now,
            components=component_results,
        )
