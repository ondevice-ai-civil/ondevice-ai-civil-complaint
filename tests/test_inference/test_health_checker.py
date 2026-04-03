# TDD: Red-Green-Refactor cycle로 구현됨
# TDD Phase: RED - 테스트 먼저 작성
"""HealthChecker 단위 테스트.

상세 헬스체크의 핵심 기능을 테스트한다:
- 컴포넌트별 상태 확인
- 상세/간략 모드
- 종합 상태 판정
- 마지막 체크 타임스탬프
- 비동기 체크
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.inference.health_checker import (
    ComponentStatus,
    HealthChecker,
    HealthStatus,
    OverallStatus,
)


@pytest.fixture
def checker() -> HealthChecker:
    return HealthChecker()


@pytest.fixture
def checker_with_mocks() -> HealthChecker:
    """모든 컴포넌트를 Mock으로 등록한 HealthChecker."""
    hc = HealthChecker()
    hc.register_check(
        "model",
        AsyncMock(
            return_value=ComponentStatus(
                name="model", status=HealthStatus.HEALTHY, message="모델 정상"
            )
        ),
    )
    hc.register_check(
        "faiss_index",
        AsyncMock(
            return_value=ComponentStatus(
                name="faiss_index", status=HealthStatus.HEALTHY, message="FAISS 정상"
            )
        ),
    )
    hc.register_check(
        "database",
        AsyncMock(
            return_value=ComponentStatus(
                name="database", status=HealthStatus.HEALTHY, message="DB 정상"
            )
        ),
    )
    return hc


class TestHealthStatus:
    """상태 열거형."""

    def test_healthy_status(self):
        assert HealthStatus.HEALTHY == "healthy"

    def test_degraded_status(self):
        assert HealthStatus.DEGRADED == "degraded"

    def test_unhealthy_status(self):
        assert HealthStatus.UNHEALTHY == "unhealthy"


class TestComponentStatus:
    """컴포넌트별 상태."""

    def test_component_status_creation(self):
        status = ComponentStatus(
            name="model",
            status=HealthStatus.HEALTHY,
            message="모델 로드 완료",
        )
        assert status.name == "model"
        assert status.status == HealthStatus.HEALTHY

    def test_component_status_with_details(self):
        status = ComponentStatus(
            name="faiss_index",
            status=HealthStatus.DEGRADED,
            message="인덱스 일부 누락",
            details={"loaded": 3, "expected": 4},
        )
        assert status.details["loaded"] == 3


class TestRegisterCheck:
    """컴포넌트 체크 등록."""

    def test_register_check(self, checker: HealthChecker):
        check_fn = AsyncMock(
            return_value=ComponentStatus(name="model", status=HealthStatus.HEALTHY, message="OK")
        )
        checker.register_check("model", check_fn)
        assert "model" in checker.registered_checks

    def test_register_multiple_checks(self, checker: HealthChecker):
        for name in ["model", "faiss_index", "database"]:
            checker.register_check(
                name,
                AsyncMock(
                    return_value=ComponentStatus(
                        name=name, status=HealthStatus.HEALTHY, message="OK"
                    )
                ),
            )
        assert len(checker.registered_checks) == 3


class TestOverallStatusDetermination:
    """종합 상태 판정."""

    @pytest.mark.asyncio
    async def test_all_healthy_returns_healthy(self, checker_with_mocks: HealthChecker):
        result = await checker_with_mocks.check(detail_level="minimal")
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_one_degraded_returns_degraded(self, checker: HealthChecker):
        checker.register_check(
            "model",
            AsyncMock(
                return_value=ComponentStatus(
                    name="model", status=HealthStatus.HEALTHY, message="OK"
                )
            ),
        )
        checker.register_check(
            "faiss_index",
            AsyncMock(
                return_value=ComponentStatus(
                    name="faiss_index", status=HealthStatus.DEGRADED, message="일부 누락"
                )
            ),
        )
        result = await checker.check(detail_level="minimal")
        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_one_unhealthy_returns_unhealthy(self, checker: HealthChecker):
        checker.register_check(
            "model",
            AsyncMock(
                return_value=ComponentStatus(
                    name="model", status=HealthStatus.UNHEALTHY, message="로드 실패"
                )
            ),
        )
        checker.register_check(
            "faiss_index",
            AsyncMock(
                return_value=ComponentStatus(
                    name="faiss_index", status=HealthStatus.HEALTHY, message="OK"
                )
            ),
        )
        result = await checker.check(detail_level="minimal")
        assert result.status == HealthStatus.UNHEALTHY


class TestDetailLevel:
    """상세/간략 모드."""

    @pytest.mark.asyncio
    async def test_minimal_mode(self, checker_with_mocks: HealthChecker):
        result = await checker_with_mocks.check(detail_level="minimal")
        assert result.status is not None
        assert result.components is None or len(result.components) == 0

    @pytest.mark.asyncio
    async def test_detailed_mode(self, checker_with_mocks: HealthChecker):
        result = await checker_with_mocks.check(detail_level="detailed")
        assert result.components is not None
        assert len(result.components) == 3

    @pytest.mark.asyncio
    async def test_detailed_includes_component_info(self, checker_with_mocks: HealthChecker):
        result = await checker_with_mocks.check(detail_level="detailed")
        names = [c.name for c in result.components]
        assert "model" in names
        assert "faiss_index" in names
        assert "database" in names


class TestTimestamp:
    """마지막 체크 타임스탬프."""

    @pytest.mark.asyncio
    async def test_timestamp_recorded(self, checker_with_mocks: HealthChecker):
        result = await checker_with_mocks.check(detail_level="minimal")
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_last_check_timestamp_updated(self, checker_with_mocks: HealthChecker):
        await checker_with_mocks.check(detail_level="minimal")
        ts = checker_with_mocks.last_check_timestamp
        assert ts is not None


class TestAsyncCheck:
    """비동기 체크."""

    @pytest.mark.asyncio
    async def test_check_handles_exception(self, checker: HealthChecker):
        async def failing_check():
            raise RuntimeError("연결 실패")

        checker.register_check("database", failing_check)
        result = await checker.check(detail_level="detailed")
        # 예외 발생 시 해당 컴포넌트는 unhealthy
        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_no_checks_registered(self, checker: HealthChecker):
        result = await checker.check(detail_level="minimal")
        assert result.status == HealthStatus.HEALTHY


class TestOverallStatusDataclass:
    """OverallStatus dataclass 검증."""

    def test_overall_status_creation(self):
        overall = OverallStatus(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(timezone.utc),
            components=[],
        )
        assert overall.status == HealthStatus.HEALTHY
        assert isinstance(overall.timestamp, datetime)
