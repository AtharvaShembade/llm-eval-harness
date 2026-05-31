import sqlite3
from datetime import datetime

DB_PATH = "eval.db"


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            git_sha   TEXT,
            notes     TEXT
        );

        CREATE TABLE IF NOT EXISTS results (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id             INTEGER NOT NULL REFERENCES runs(id),
            question           TEXT    NOT NULL,
            answer             TEXT    NOT NULL,
            faithfulness       REAL    NOT NULL,
            answer_relevance   REAL    NOT NULL,
            context_precision  REAL    NOT NULL,
            passed             INTEGER NOT NULL,
            reasoning          TEXT    NOT NULL
        );
    """)
    con.commit()
    con.close()


def create_run(git_sha: str | None = None, notes: str | None = None) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.execute(
        "INSERT INTO runs (timestamp, git_sha, notes) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), git_sha, notes),
    )
    run_id = cur.lastrowid
    con.commit()
    con.close()
    return run_id


def save_result(
    run_id: int,
    question: str,
    answer: str,
    faithfulness: float,
    answer_relevance: float,
    context_precision: float,
    passed: bool,
    reasoning: str,
):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO results
           (run_id, question, answer, faithfulness, answer_relevance,
            context_precision, passed, reasoning)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, question, answer, faithfulness, answer_relevance,
         context_precision, int(passed), reasoning),
    )
    con.commit()
    con.close()


def get_last_run(before_run_id: int | None = None) -> int | None:
    con = sqlite3.connect(DB_PATH)
    if before_run_id is not None:
        row = con.execute(
            "SELECT id FROM runs WHERE id < ? ORDER BY id DESC LIMIT 1",
            (before_run_id,),
        ).fetchone()
    else:
        row = con.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return row[0] if row else None


def get_results(run_id: int) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM results WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]
