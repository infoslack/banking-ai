-- Core bancário fake: ledger append-only de dupla entrada.
-- Nunca UPDATE em saldo; saldo é derivado dos lançamentos.
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE IF NOT EXISTS contas (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  nome TEXT NOT NULL,
  chave_pix TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS intents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES contas(id),
  tipo TEXT NOT NULL CHECK (tipo IN ('pix', 'boleto')),
  payload JSONB NOT NULL,                  -- valores canônicos, já validados
  status TEXT NOT NULL DEFAULT 'pendente'
    CHECK (status IN ('pendente', 'confirmada', 'expirada', 'cancelada')),
  criada_em TIMESTAMPTZ NOT NULL DEFAULT now(),
  expira_em TIMESTAMPTZ NOT NULL DEFAULT now() + interval '5 minutes'
);

CREATE TABLE IF NOT EXISTS lancamentos (
  id BIGSERIAL PRIMARY KEY,
  intent_id UUID UNIQUE REFERENCES intents(id),  -- UNIQUE = idempotência
  conta_debito UUID NOT NULL REFERENCES contas(id),
  conta_credito UUID NOT NULL REFERENCES contas(id),
  valor_centavos BIGINT NOT NULL CHECK (valor_centavos > 0),
  criado_em TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (conta_debito <> conta_credito)
);

CREATE INDEX IF NOT EXISTS lancamentos_debito_idx ON lancamentos (conta_debito);
CREATE INDEX IF NOT EXISTS lancamentos_credito_idx ON lancamentos (conta_credito);
CREATE INDEX IF NOT EXISTS intents_user_status_idx ON intents (user_id, status);
