"""Pool de conexões e bootstrap do schema/seeds (idempotente)."""

from pathlib import Path

from psycopg import AsyncConnection
from psycopg.rows import class_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

DB_DIR = Path(__file__).resolve().parents[2] / "db"
SCHEMA_SQL = (DB_DIR / "schema.sql").read_text(encoding="utf-8")
SEEDS_SQL = (DB_DIR / "seeds.sql").read_text(encoding="utf-8")


class Count(BaseModel):
    total: int


def create_pool(database_url: str) -> AsyncConnectionPool:
    return AsyncConnectionPool(database_url, open=False, min_size=1, max_size=10)


async def apply_schema(conn: AsyncConnection) -> None:
    await conn.execute(SCHEMA_SQL)
    await conn.commit()


async def seed_if_empty(conn: AsyncConnection) -> bool:
    async with conn.cursor(row_factory=class_row(Count)) as cur:
        await cur.execute("SELECT count(*) AS total FROM contas")
        count = await cur.fetchone()
    if count is not None and count.total > 0:
        return False
    await conn.execute(SEEDS_SQL)
    await conn.commit()
    return True


async def reset_demo(conn: AsyncConnection) -> None:
    """Volta o banco ao estado inicial da demo (para retomar de qualquer passo)."""
    async with conn.transaction():
        await conn.execute("TRUNCATE lancamentos, intents, contas RESTART IDENTITY CASCADE")
        await conn.execute(SEEDS_SQL)


async def prepare_database(pool: AsyncConnectionPool) -> bool:
    async with pool.connection() as conn:
        await apply_schema(conn)
        return await seed_if_empty(conn)
