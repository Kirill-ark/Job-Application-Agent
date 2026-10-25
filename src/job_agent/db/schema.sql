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
    components  TEXT NOT NULL,                  -- JSON: {skills, level, german, location}
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
    issues      TEXT NOT NULL DEFAULT '[]',     -- JSON: groundedness problems left after repair
    status      TEXT NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft', 'approved', 'rejected')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Approval log. Rows are only ever added: the triggers below forbid UPDATE and DELETE.
-- A request ('requested' or 'refused') comes from the agent; a decision
-- ('approved' or 'rejected') comes from the human and points to its request.
-- A request with no decision row is pending.
CREATE TABLE IF NOT EXISTS approvals (
    id          INTEGER PRIMARY KEY,
    request_id  INTEGER REFERENCES approvals (id),  -- set on decisions only
    draft_id    INTEGER NOT NULL REFERENCES drafts (id),
    action      TEXT NOT NULL,
    event       TEXT NOT NULL
                CHECK (event IN ('requested', 'refused', 'approved', 'rejected')),
    comment     TEXT,
    created_at  TEXT NOT NULL,
    -- Unknown actions ('send', ...) can only be logged as refused.
    CHECK (event = 'refused' OR action IN ('mark_ready', 'export')),
    CHECK ((request_id IS NULL) = (event IN ('requested', 'refused')))
);

-- One decision per request: deciding twice is an error.
CREATE UNIQUE INDEX IF NOT EXISTS one_decision_per_request
    ON approvals (request_id) WHERE request_id IS NOT NULL;

CREATE TRIGGER IF NOT EXISTS approvals_no_update BEFORE UPDATE ON approvals
BEGIN
    SELECT RAISE(ABORT, 'the approval log is append-only');
END;

CREATE TRIGGER IF NOT EXISTS approvals_no_delete BEFORE DELETE ON approvals
BEGIN
    SELECT RAISE(ABORT, 'the approval log is append-only');
END;

-- Agent runs: the trace of every step, rate limiting and the saved demo run.
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    goal        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running'
                CHECK (status IN ('running', 'done', 'failed', 'limited')),
    ip_hash     TEXT,                           -- hash, never the raw IP
    token_hash  TEXT,                           -- hash of the browser's secret for this run
    trace       TEXT NOT NULL DEFAULT '[]',     -- JSON: list of steps
    answer      TEXT,                           -- the agent's final message
    error       TEXT,                           -- why the run failed
    is_demo     INTEGER NOT NULL DEFAULT 0 CHECK (is_demo IN (0, 1)),
    own_cv      INTEGER NOT NULL DEFAULT 0 CHECK (own_cv IN (0, 1)),  -- a visitor's CV
    started_at  TEXT NOT NULL,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS runs_by_ip ON runs (ip_hash, started_at);
