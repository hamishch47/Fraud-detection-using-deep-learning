-- schema.sql
-- Database schema for the Live Fraud Analyst Dashboard.
-- Compatible with PostgreSQL 14+.
--
-- Usage:
--   psql -U postgres -d fraud -f schema.sql

-- ---------------------------------------------------------------------------
-- Transactions
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS transactions (
    id           TEXT        PRIMARY KEY,
    amount       NUMERIC(14, 2) NOT NULL,
    merchant     TEXT        NOT NULL DEFAULT '',
    location     TEXT        NOT NULL DEFAULT '',
    risk_score   NUMERIC(5, 2)  NOT NULL DEFAULT 0,
    status       TEXT        NOT NULL DEFAULT 'Pending Review'
                             CHECK (status IN ('Pending Review', 'Auto-Approved', 'Fraud', 'Approved')),
    context      TEXT        NOT NULL DEFAULT '',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_transactions_status      ON transactions (status);
CREATE INDEX IF NOT EXISTS idx_transactions_created_at  ON transactions (created_at DESC);

-- ---------------------------------------------------------------------------
-- Analyst actions / audit log
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS analyst_actions (
    id          BIGSERIAL   PRIMARY KEY,
    txn_id      TEXT        NOT NULL REFERENCES transactions (id) ON DELETE CASCADE,
    action      TEXT        NOT NULL,   -- e.g. 'Fraud', 'Approved'
    actor       TEXT        NOT NULL DEFAULT 'analyst-ui',
    action_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    note        TEXT
);

CREATE INDEX IF NOT EXISTS idx_analyst_actions_txn_id ON analyst_actions (txn_id);
