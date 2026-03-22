"""
SQLAlchemy 2.0 데이터베이스 엔진/세션 설정.

동기 세션 기반으로 구성하며, FastAPI 의존성 주입(get_db)을 제공한다.
DATABASE_URL 환경변수에서 PostgreSQL 연결 문자열을 읽는다.
"""

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# ---------------------------------------------------------------------------
# 엔진 & 세션 팩토리
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://govon:govon@localhost:5432/govon",
)

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,          # 연결 유효성 사전 검사
    pool_recycle=3600,           # 1시간마다 커넥션 재활용
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
        db.close()
