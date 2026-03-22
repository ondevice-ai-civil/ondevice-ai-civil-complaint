"""
Alembic 환경 설정.

DATABASE_URL 환경변수를 통해 PostgreSQL 연결 문자열을 주입받는다.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# 프로젝트 루트를 sys.path에 추가하여 모델 import 가능하게 함
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from src.inference.db.models import Base  # noqa: E402

# Alembic Config 객체
config = context.config

# 로깅 설정
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 메타데이터 설정 (자동 마이그레이션 생성용)
target_metadata = Base.metadata

# 환경변수에서 DB URL 가져오기
database_url = os.getenv(
    "DATABASE_URL",
    "postgresql://govon:govon@localhost:5432/govon",
)
config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    """오프라인 모드: DB 연결 없이 SQL 스크립트만 생성."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """온라인 모드: DB에 직접 연결하여 마이그레이션 실행."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
