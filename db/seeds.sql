-- Estado inicial da demo. Idempotente: só roda quando `contas` está vazia
-- (ver banking_ai.db.semear_se_vazio) ou após o reset administrativo.
INSERT INTO contas (id, nome, chave_pix) VALUES
  ('00000000-0000-0000-0000-000000000001', 'Tesouraria',     'tesouraria@banco.demo'),
  ('00000000-0000-0000-0000-000000000002', 'Boletos',        'boletos@banco.demo'),
  ('10000000-0000-0000-0000-000000000001', 'Daniel Romero',      'daniel@email.com'),
  ('10000000-0000-0000-0000-000000000002', 'Alberto Souza',     '+5511999990001'),
  ('10000000-0000-0000-0000-000000000003', 'Alberto Pereira',   'alberto.pereira@email.com'),
  ('10000000-0000-0000-0000-000000000004', 'Maria Oliveira', '11122233344'),
  ('10000000-0000-0000-0000-000000000005', 'Carlos Lima',    '+5521988887777');

-- Saldos iniciais entram como lançamentos vindos da Tesouraria (intent_id NULL:
-- o UNIQUE admite vários NULLs, então aportes iniciais não colidem).
INSERT INTO lancamentos (conta_debito, conta_credito, valor_centavos) VALUES
  ('00000000-0000-0000-0000-000000000001', '10000000-0000-0000-0000-000000000001', 250000),
  ('00000000-0000-0000-0000-000000000001', '10000000-0000-0000-0000-000000000002',  30000),
  ('00000000-0000-0000-0000-000000000001', '10000000-0000-0000-0000-000000000003',  12000),
  ('00000000-0000-0000-0000-000000000001', '10000000-0000-0000-0000-000000000004', 100000),
  ('00000000-0000-0000-0000-000000000001', '10000000-0000-0000-0000-000000000005',   5000);
