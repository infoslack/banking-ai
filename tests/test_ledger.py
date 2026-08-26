"""O Postgres é o último guardrail: saldo derivado, FOR UPDATE, UNIQUE (intent_id)."""

import asyncio

import pytest
from psycopg.errors import UniqueViolation
from psycopg_pool import AsyncConnectionPool

from banking_ai import ledger
from banking_ai.models import Account, BoletoPayload, Contact, PixPayload

ALBERTO_SOUZA_PIX_KEY = "+5511999990001"


async def _pix_payload(pool: AsyncConnectionPool, amount: int) -> PixPayload:
    async with pool.connection() as conn:
        alberto = await ledger.account_by_pix_key(conn, ALBERTO_SOUZA_PIX_KEY)
    if alberto is None:
        raise AssertionError("seed do Alberto Souza ausente")
    return PixPayload(chave_pix=alberto.chave_pix, valor_centavos=amount, nome_recebedor=alberto.nome, conta_recebedor_id=alberto.id)


async def _balance(pool: AsyncConnectionPool, account: Account) -> int:
    async with pool.connection() as conn:
        return (await ledger.get_balance(conn, account.id)).saldo_centavos


async def test_balance_is_derived_from_seeds(pool: AsyncConnectionPool, daniel: Account) -> None:
    assert await _balance(pool, daniel) == 250_000


async def test_contact_search_ignores_accents_and_internal_accounts(pool: AsyncConnectionPool, daniel: Account) -> None:
    async with pool.connection() as conn:
        albertos = await ledger.search_contacts(conn, "alberto", exclude_id=daniel.id)
        albertos_with_accent = await ledger.search_contacts(conn, "albérto", exclude_id=daniel.id)
        internal = await ledger.search_contacts(conn, "tesouraria", exclude_id=daniel.id)
        self_account = await ledger.search_contacts(conn, "daniel", exclude_id=daniel.id)
    assert [c.nome for c in albertos] == ["Alberto Pereira", "Alberto Souza"]
    assert albertos == albertos_with_accent
    assert internal == []
    assert self_account == []


async def test_two_phase_commit_proposing_moves_no_money(pool: AsyncConnectionPool, daniel: Account) -> None:
    payload = await _pix_payload(pool, 5000)
    async with pool.connection() as conn:
        intent = await ledger.create_intent(conn, daniel.id, payload)
    assert intent.status == "pendente"
    assert await _balance(pool, daniel) == 250_000


async def test_confirm_executes_the_stored_payload(pool: AsyncConnectionPool, daniel: Account) -> None:
    payload = await _pix_payload(pool, 5000)
    async with pool.connection() as conn:
        intent = await ledger.create_intent(conn, daniel.id, payload)
        result = await ledger.confirm_intent(conn, intent.id, daniel.id)
        alberto = await ledger.account_by_pix_key(conn, ALBERTO_SOUZA_PIX_KEY)
    assert result.status == "executada"
    assert result.saldo_centavos == 245_000
    assert result.destinatario == "Alberto Souza"
    assert alberto is not None
    assert await _balance(pool, alberto) == 35_000


async def test_duplicate_request_creates_no_second_entry(pool: AsyncConnectionPool, daniel: Account) -> None:
    payload = await _pix_payload(pool, 5000)
    async with pool.connection() as conn:
        intent = await ledger.create_intent(conn, daniel.id, payload)
        first = await ledger.confirm_intent(conn, intent.id, daniel.id)
        second = await ledger.confirm_intent(conn, intent.id, daniel.id)
        entries = await ledger.statement(conn, daniel.id)
    assert first.status == "executada"
    assert second.status == "ja_executada"
    assert second.lancamento_id == first.lancamento_id
    assert [e.intent_id for e in entries].count(intent.id) == 1
    assert await _balance(pool, daniel) == 245_000


async def test_concurrent_confirmations_execute_once(pool: AsyncConnectionPool, daniel: Account) -> None:
    payload = await _pix_payload(pool, 5000)
    async with pool.connection() as conn:
        intent = await ledger.create_intent(conn, daniel.id, payload)

    async def confirm() -> str:
        async with pool.connection() as conn:
            return (await ledger.confirm_intent(conn, intent.id, daniel.id)).status

    statuses = await asyncio.gather(*(confirm() for _ in range(5)))
    assert sorted(statuses) == ["executada", "ja_executada", "ja_executada", "ja_executada", "ja_executada"]
    assert await _balance(pool, daniel) == 245_000


async def test_unique_intent_id_blocks_direct_duplicate_insert(pool: AsyncConnectionPool, daniel: Account) -> None:
    payload = await _pix_payload(pool, 5000)
    async with pool.connection() as conn:
        intent = await ledger.create_intent(conn, daniel.id, payload)
        await ledger.confirm_intent(conn, intent.id, daniel.id)
        with pytest.raises(UniqueViolation):
            await conn.execute(
                "INSERT INTO lancamentos (intent_id, conta_debito, conta_credito, valor_centavos) VALUES (%s, %s, %s, %s)",
                (intent.id, daniel.id, payload.conta_recebedor_id, 5000),
            )


async def test_insufficient_balance_cancels_without_entry(pool: AsyncConnectionPool, daniel: Account) -> None:
    payload = await _pix_payload(pool, 400_000)
    async with pool.connection() as conn:
        intent = await ledger.create_intent(conn, daniel.id, payload)
        result = await ledger.confirm_intent(conn, intent.id, daniel.id)
        after = await ledger.get_intent(conn, intent.id, daniel.id)
    assert result.status == "saldo_insuficiente"
    assert result.saldo_centavos == 250_000
    assert after is not None and after.status == "cancelada"
    assert await _balance(pool, daniel) == 250_000


async def test_expired_intent_does_not_execute(pool: AsyncConnectionPool, daniel: Account) -> None:
    payload = await _pix_payload(pool, 5000)
    async with pool.connection() as conn:
        intent = await ledger.create_intent(conn, daniel.id, payload)
        await conn.execute("UPDATE intents SET expira_em = now() - interval '1 minute' WHERE id = %s", (intent.id,))
        await conn.commit()
        result = await ledger.confirm_intent(conn, intent.id, daniel.id)
        after = await ledger.get_intent(conn, intent.id, daniel.id)
    assert result.status == "expirada"
    assert after is not None and after.status == "expirada"
    assert await _balance(pool, daniel) == 250_000


async def test_cancel_then_confirm(pool: AsyncConnectionPool, daniel: Account) -> None:
    payload = await _pix_payload(pool, 5000)
    async with pool.connection() as conn:
        intent = await ledger.create_intent(conn, daniel.id, payload)
        cancelled = await ledger.cancel_intent(conn, intent.id, daniel.id)
        result = await ledger.confirm_intent(conn, intent.id, daniel.id)
        again = await ledger.cancel_intent(conn, intent.id, daniel.id)
    assert cancelled.status == "cancelada"
    assert result.status == "cancelada"
    assert again.status == "nao_pendente"


async def test_intent_of_another_user_is_not_found(pool: AsyncConnectionPool, daniel: Account) -> None:
    payload = await _pix_payload(pool, 5000)
    async with pool.connection() as conn:
        intent = await ledger.create_intent(conn, daniel.id, payload)
        other = await ledger.account_by_pix_key(conn, "11122233344")
        assert other is not None
        result = await ledger.confirm_intent(conn, intent.id, other.id)
    assert result.status == "nao_encontrada"


async def test_boleto_credits_internal_boletos_account(pool: AsyncConnectionPool, daniel: Account) -> None:
    payload = BoletoPayload(linha_digitavel="2" * 47, valor_centavos=15_000, beneficiario="Energia Luz S.A.")
    async with pool.connection() as conn:
        intent = await ledger.create_intent(conn, daniel.id, payload)
        result = await ledger.confirm_intent(conn, intent.id, daniel.id)
        boletos = await ledger.get_balance(conn, ledger.BOLETOS_ACCOUNT_ID)
    assert result.status == "executada"
    assert result.destinatario == "Energia Luz S.A."
    assert boletos.saldo_centavos == 15_000
    assert await _balance(pool, daniel) == 235_000


async def test_resolve_recipient(pool: AsyncConnectionPool, daniel: Account) -> None:
    async with pool.connection() as conn:
        by_key = await ledger.resolve_recipient(conn, ALBERTO_SOUZA_PIX_KEY, daniel.id)
        by_name = await ledger.resolve_recipient(conn, "alberto souza", daniel.id)
        by_surname = await ledger.resolve_recipient(conn, "Souza", daniel.id)
        ambiguous = await ledger.resolve_recipient(conn, "alberto", daniel.id)
        none = await ledger.resolve_recipient(conn, "fulana", daniel.id)
        internal = await ledger.resolve_recipient(conn, "tesouraria@banco.demo", daniel.id)
    assert by_key == by_name == by_surname
    assert by_key.nome == "Alberto Souza"  # type: ignore[union-attr]  # SAFETY: as três resoluções acima são iguais e a primeira é por chave exata, logo Account
    assert ambiguous == [
        Contact(nome="Alberto Pereira", chave_pix="alberto.pereira@email.com"),
        Contact(nome="Alberto Souza", chave_pix=ALBERTO_SOUZA_PIX_KEY),
    ]
    assert none == []
    assert internal == []
