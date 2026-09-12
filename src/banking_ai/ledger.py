"""Core bancário fake: o Postgres é o último guardrail.

Saldo é derivado (nunca UPDATE), execução roda em transação com
`SELECT ... FOR UPDATE` na conta de origem e o `UNIQUE (intent_id)` em
`lancamentos` garante idempotência contra requisições duplicadas.
"""

from datetime import datetime
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.errors import UniqueViolation
from psycopg.rows import class_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from banking_ai.models.domain import (
    Account,
    Balance,
    BoletoPayload,
    CancellationResult,
    ConfirmationResult,
    Contact,
    Intent,
    IntentPayload,
    LedgerEntry,
    PixPayload,
    StatementEntry,
)

TREASURY_ACCOUNT_ID = UUID("00000000-0000-0000-0000-000000000001")
BOLETOS_ACCOUNT_ID = UUID("00000000-0000-0000-0000-000000000002")
INTERNAL_ACCOUNT_SUFFIX = "@banco.demo"

INTENT_COLUMNS = "id, user_id, tipo, payload, status, criada_em, expira_em"
ENTRY_COLUMNS = "id, intent_id, conta_debito, conta_credito, valor_centavos, criado_em"


class Now(BaseModel):
    now: datetime


class BalanceRow(BaseModel):
    saldo_centavos: int


async def db_now(conn: AsyncConnection) -> datetime:
    async with conn.cursor(row_factory=class_row(Now)) as cur:
        await cur.execute("SELECT now() AS now")
        row = await cur.fetchone()
    if row is None:
        raise RuntimeError("SELECT now() sem linha")
    return row.now


async def account_by_pix_key(conn: AsyncConnection, pix_key: str) -> Account | None:
    async with conn.cursor(row_factory=class_row(Account)) as cur:
        await cur.execute("SELECT id, nome, chave_pix FROM contas WHERE chave_pix = %s", (pix_key,))
        return await cur.fetchone()


async def account_by_id(conn: AsyncConnection, account_id: UUID) -> Account | None:
    async with conn.cursor(row_factory=class_row(Account)) as cur:
        await cur.execute("SELECT id, nome, chave_pix FROM contas WHERE id = %s", (account_id,))
        return await cur.fetchone()


async def get_balance(conn: AsyncConnection, account_id: UUID) -> Balance:
    """Saldo = SUM(créditos) − SUM(débitos), calculado sob demanda."""
    async with conn.cursor(row_factory=class_row(BalanceRow)) as cur:
        await cur.execute(
            """
            SELECT (
              COALESCE(SUM(CASE WHEN conta_credito = %(id)s THEN valor_centavos ELSE 0 END), 0)
              - COALESCE(SUM(CASE WHEN conta_debito = %(id)s THEN valor_centavos ELSE 0 END), 0)
            )::BIGINT AS saldo_centavos
            FROM lancamentos
            WHERE conta_credito = %(id)s OR conta_debito = %(id)s
            """,
            {"id": account_id},
        )
        row = await cur.fetchone()
    return Balance(conta_id=account_id, saldo_centavos=0 if row is None else row.saldo_centavos)


async def search_contacts(conn: AsyncConnection, term: str, exclude_id: UUID) -> list[Contact]:
    """Candidatos reais por nome (sem acento) ou chave. Nunca inventa destinatário."""
    pattern = f"%{term.strip()}%"
    async with conn.cursor(row_factory=class_row(Contact)) as cur:
        await cur.execute(
            """
            SELECT nome, chave_pix FROM contas
            WHERE id <> %(exclude)s
              AND chave_pix NOT LIKE %(internal)s
              AND (unaccent(nome) ILIKE unaccent(%(pattern)s) OR chave_pix ILIKE %(pattern)s)
            ORDER BY nome
            LIMIT 10
            """,
            {"exclude": exclude_id, "internal": f"%{INTERNAL_ACCOUNT_SUFFIX}", "pattern": pattern},
        )
        return await cur.fetchall()


async def resolve_recipient(conn: AsyncConnection, term: str, exclude_id: UUID) -> Account | list[Contact]:
    """Chave exata ou nome com um único match viram a conta; senão, a lista de candidatos (talvez vazia)."""
    exact = await account_by_pix_key(conn, term.strip())
    if exact is not None and not exact.chave_pix.endswith(INTERNAL_ACCOUNT_SUFFIX):
        return exact
    candidates = await search_contacts(conn, term, exclude_id)
    if len(candidates) != 1:
        return candidates
    only = await account_by_pix_key(conn, candidates[0].chave_pix)
    return candidates if only is None else only


async def create_intent(conn: AsyncConnection, user_id: UUID, payload: IntentPayload) -> Intent:
    """Fase 1 do two-phase commit: grava a intenção validada, sem mover dinheiro."""
    async with conn.cursor(row_factory=class_row(Intent)) as cur:
        await cur.execute(
            f"""
            INSERT INTO intents (user_id, tipo, payload)
            VALUES (%s, %s, %s)
            RETURNING {INTENT_COLUMNS}
            """,
            (user_id, payload.tipo, Jsonb(payload.model_dump(mode="json"))),
        )
        intent = await cur.fetchone()
    await conn.commit()
    if intent is None:
        raise RuntimeError("INSERT em intents sem RETURNING")
    return intent


async def get_intent(conn: AsyncConnection, intent_id: UUID, user_id: UUID) -> Intent | None:
    async with conn.cursor(row_factory=class_row(Intent)) as cur:
        await cur.execute(
            f"SELECT {INTENT_COLUMNS} FROM intents WHERE id = %s AND user_id = %s",
            (intent_id, user_id),
        )
        return await cur.fetchone()


async def _entry_for_intent(conn: AsyncConnection, intent_id: UUID) -> LedgerEntry | None:
    async with conn.cursor(row_factory=class_row(LedgerEntry)) as cur:
        await cur.execute(f"SELECT {ENTRY_COLUMNS} FROM lancamentos WHERE intent_id = %s", (intent_id,))
        return await cur.fetchone()


def _recipient_name(payload: IntentPayload) -> str:
    match payload:
        case PixPayload():
            return payload.nome_recebedor
        case BoletoPayload():
            return payload.beneficiario


def _credit_account(payload: IntentPayload) -> UUID:
    match payload:
        case PixPayload():
            return payload.conta_recebedor_id
        case BoletoPayload():
            return BOLETOS_ACCOUNT_ID


async def confirm_intent(conn: AsyncConnection, intent_id: UUID, user_id: UUID) -> ConfirmationResult:
    """Fase 2: executa o payload GRAVADO NO BANCO, nunca valores reinterpretados.

    O INSERT roda num savepoint: o UNIQUE (intent_id) barra a duplicata sem derrubar a transação.
    """
    async with conn.transaction():
        async with conn.cursor(row_factory=class_row(Intent)) as cur:
            await cur.execute(
                f"SELECT {INTENT_COLUMNS} FROM intents WHERE id = %s AND user_id = %s FOR UPDATE",
                (intent_id, user_id),
            )
            intent = await cur.fetchone()
        if intent is None:
            return ConfirmationResult(status="nao_encontrada", intent_id=intent_id)

        amount = intent.payload.valor_centavos
        recipient = _recipient_name(intent.payload)
        if intent.status == "confirmada":
            existing = await _entry_for_intent(conn, intent_id)
            return ConfirmationResult(
                status="ja_executada",
                intent_id=intent_id,
                lancamento_id=None if existing is None else existing.id,
                valor_centavos=amount,
                destinatario=recipient,
                observacao="Requisição duplicada ignorada: a intent já foi executada uma vez.",
            )
        if intent.status == "cancelada":
            return ConfirmationResult(status="cancelada", intent_id=intent_id, destinatario=recipient)
        now = await db_now(conn)
        if intent.status == "expirada" or intent.expira_em <= now:
            await conn.execute("UPDATE intents SET status = 'expirada' WHERE id = %s", (intent_id,))
            return ConfirmationResult(
                status="expirada",
                intent_id=intent_id,
                valor_centavos=amount,
                destinatario=recipient,
                observacao="A intent expirou; proponha novamente se o usuário quiser.",
            )

        await conn.execute("SELECT id FROM contas WHERE id = %s FOR UPDATE", (user_id,))
        balance = await get_balance(conn, user_id)
        if balance.saldo_centavos < amount:
            await conn.execute("UPDATE intents SET status = 'cancelada' WHERE id = %s", (intent_id,))
            return ConfirmationResult(
                status="saldo_insuficiente",
                intent_id=intent_id,
                valor_centavos=amount,
                destinatario=recipient,
                saldo_centavos=balance.saldo_centavos,
                observacao="Saldo insuficiente; a intent foi cancelada.",
            )

        try:
            async with conn.transaction(), conn.cursor(row_factory=class_row(LedgerEntry)) as cur:
                await cur.execute(
                    f"""
                    INSERT INTO lancamentos (intent_id, conta_debito, conta_credito, valor_centavos)
                    VALUES (%s, %s, %s, %s)
                    RETURNING {ENTRY_COLUMNS}
                    """,
                    (intent_id, user_id, _credit_account(intent.payload), amount),
                )
                entry = await cur.fetchone()
        except UniqueViolation:
            existing = await _entry_for_intent(conn, intent_id)
            return ConfirmationResult(
                status="ja_executada",
                intent_id=intent_id,
                lancamento_id=None if existing is None else existing.id,
                valor_centavos=amount,
                destinatario=recipient,
                observacao="UNIQUE (intent_id) barrou um lançamento duplicado.",
            )
        if entry is None:
            raise RuntimeError("INSERT em lancamentos sem RETURNING")
        await conn.execute("UPDATE intents SET status = 'confirmada' WHERE id = %s", (intent_id,))
        new_balance = await get_balance(conn, user_id)
        return ConfirmationResult(
            status="executada",
            intent_id=intent_id,
            lancamento_id=entry.id,
            valor_centavos=amount,
            destinatario=recipient,
            saldo_centavos=new_balance.saldo_centavos,
        )


async def cancel_intent(conn: AsyncConnection, intent_id: UUID, user_id: UUID) -> CancellationResult:
    async with conn.transaction():
        intent = await get_intent(conn, intent_id, user_id)
        if intent is None:
            return CancellationResult(status="nao_encontrada", intent_id=intent_id)
        if intent.status != "pendente":
            return CancellationResult(status="nao_pendente", intent_id=intent_id)
        await conn.execute("UPDATE intents SET status = 'cancelada' WHERE id = %s", (intent_id,))
        return CancellationResult(status="cancelada", intent_id=intent_id)


async def statement(conn: AsyncConnection, account_id: UUID, limit: int = 20) -> list[StatementEntry]:
    async with conn.cursor(row_factory=class_row(StatementEntry)) as cur:
        await cur.execute(
            """
            SELECT l.id, l.intent_id,
                   CASE WHEN l.conta_debito = %(id)s THEN 'debit' ELSE 'credit' END AS direction,
                   c.nome AS counterparty,
                   l.valor_centavos AS amount_cents,
                   l.criado_em AS created_at
            FROM lancamentos l
            JOIN contas c ON c.id = CASE WHEN l.conta_debito = %(id)s THEN l.conta_credito ELSE l.conta_debito END
            WHERE l.conta_debito = %(id)s OR l.conta_credito = %(id)s
            ORDER BY l.id DESC
            LIMIT %(limit)s
            """,
            {"id": account_id, "limit": limit},
        )
        return await cur.fetchall()
