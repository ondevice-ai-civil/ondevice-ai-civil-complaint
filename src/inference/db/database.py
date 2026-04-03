"""
SQLAlchemy 2.0 데이터베이스 엔진/세션 설정.

동기 세션 기반으로 구성하며, FastAPI 의존성 주입(get_db)을 제공한다.
기본값은 로컬 단일 사용자 MVP에 맞춰 GovOn 홈 디렉터리 아래 SQLite 파일을 사용한다.
"""

import logging
import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 엔진 & 세션 팩토리
# ---------------------------------------------------------------------------

_DEFAULT_GOVON_HOME = Path(os.getenv("GOVON_HOME", Path.home() / ".govon"))
_DEFAULT_DATABASE_URL = f"sqlite:///{_DEFAULT_GOVON_HOME / 'metadata.sqlite3'}"

DATABASE_URL: str = os.getenv("DATABASE_URL", _DEFAULT_DATABASE_URL)

if DATABASE_URL == _DEFAULT_DATABASE_URL:
    logger.warning(
        "DATABASE_URL 환경변수가 설정되지 않아 로컬 SQLite 기본값을 사용합니다. "
        "별도 RDBMS를 사용하려면 DATABASE_URL을 명시적으로 설정하세요."
    )

engine_kwargs = {
    "echo": os.getenv("SQL_ECHO", "").lower() in ("1", "true"),
}
if DATABASE_URL.startswith("sqlite:///"):
    _DEFAULT_GOVON_HOME.mkdir(parents=True, exist_ok=True)
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs.update(
        {
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
        }
    )

engine = create_engine(DATABASE_URL, **engine_kwargs)

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
