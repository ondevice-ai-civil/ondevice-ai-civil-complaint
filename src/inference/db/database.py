"""
SQLAlchemy 2.0 데이터베이스 엔진/세션 설정.

동기 세션 기반으로 구성하며, FastAPI 의존성 주입(get_db)을 제공한다.
DATABASE_URL 환경변수에서 PostgreSQL 연결 문자열을 읽는다.
"""

import logging
import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 엔진 & 세션 팩토리
# ---------------------------------------------------------------------------

_DEFAULT_DATABASE_URL = "postgresql://govon:govon@localhost:5432/govon"

DATABASE_URL: str = os.getenv("DATABASE_URL", _DEFAULT_DATABASE_URL)

if DATABASE_URL == _DEFAULT_DATABASE_URL:
    logger.warning(
        "DATABASE_URL 환경변수가 설정되지 않아 기본값을 사용합니다. "
        "프로덕션 환경에서는 반드시 DATABASE_URL을 설정하세요."
    )

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # 연결 유효성 사전 검사
    pool_recycle=3600,  # 1시간마다 커넥션 재활용
    echo=os.getenv("SQL_ECHO", "").lower() in ("1", "true"),
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# FastAPI 의존성 주입
# ---------------------------------------------------------------------------


def get_db() -> Generator[Session, None, None]:
    """FastAPI Depends()용 세션 제너레이터.

    사용 예시::

        @router.get("/docs")
        def list_docs(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()
