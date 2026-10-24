-- Database schema. Applied on every start; IF NOT EXISTS makes that safe.
-- JSON columns are stored as TEXT: we never filter by them in SQL.
-- Times are ISO 8601 strings in UTC.

-- Vacancies fetched from Arbeitnow (a cache).
CREATE TABLE IF NOT EXISTS vacancies (
    id          TEXT PRIMARY KEY,               -- Arbeitnow slug
    title       TEXT NOT NULL,
    company     TEXT NOT NULL,
    location    TEXT NOT NULL,
    remote      INTEGER NOT NULL CHECK (remote IN (0, 1)),
    url         TEXT NOT NULL,
    description TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '[]',     -- JSON list
    job_types   TEXT NOT NULL DEFAULT '[]',     -- JSON list
    level       TEXT NOT NULL
                CHECK (level IN ('student', 'intern', 'entry', 'experienced', 'unknown')),
    posted_at   TEXT NOT NULL,
    fetched_at  TEXT NOT NULL                   -- the same for every row of one refresh
);

-- Result of score_match for one vacancy and one CV.
-- The CV itself is not stored, only a hash of its text.
CREATE TABLE IF NOT EXISTS scores (
    id          INTEGER PRIMARY KEY,
    vacancy_id  TEXT NOT NULL REFERENCES vacancies (id),
    cv_hash     TEXT NOT NULL,
    score       INTEGER NOT NULL CHECK (score BETWEEN 0 AND 100),
    matched     TEXT NOT NULL,                  -- JSON: [{skill, evidence}]
    gaps        TEXT NOT NULL,                  -- JSON: [skill or requirement]
    reasons     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (vacancy_id, cv_hash)
);

-- Cover letter drafts. `text` cites CV facts like [p1.f2].
CREATE TABLE IF NOT EXISTS drafts (
    id          INTEGER PRIMARY KEY,
    vacancy_id  TEXT NOT NULL REFERENCES vacancies (id),
    cv_hash     TEXT NOT NULL,
    text        TEXT NOT NULL,
    edited_text TEXT,                           -- the user's edit, if any
    cited_facts TEXT NOT NULL,                  -- JSON: {fact_id: fact text}
    status      TEXT NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft', 'approved', 'rejected')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Human decisions. Rows are only ever added, never changed.
CREATE TABLE IF NOT EXISTS approvals (
    id          INTEGER PRIMARY KEY,
    draft_id    INTEGER NOT NULL REFERENCES drafts (id),
    action      TEXT NOT NULL CHECK (action IN ('mark_ready', 'export')),
    decision    TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    comment     TEXT,
    decided_at  TEXT NOT NULL
);

-- Agent runs: the trace of every step, rate limiting and the saved demo run.
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    goal        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running'
                CHECK (status IN ('running', 'done', 'failed', 'limited')),
    ip_hash     TEXT,                           -- hash, never the raw IP
    trace       TEXT NOT NULL DEFAULT '[]',     -- JSON: list of steps
    is_demo     INTEGER NOT NULL DEFAULT 0 CHECK (is_demo IN (0, 1)),
    started_at  TEXT NOT NULL,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS runs_by_ip ON runs (ip_hash, started_at);
