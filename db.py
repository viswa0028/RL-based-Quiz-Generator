# -*- coding: utf-8 -*-
"""
db.py

SQLite persistence for the Adaptive Quiz System.

This is what makes the RL policy actually "learn" across sessions instead
of resetting every time the process restarts: each user's Q-table and
concept-mastery history are saved after every quiz and reloaded next time
they log in. It also stores a per-attempt history so a dashboard can show
progress over time.

Uses only the standard library (sqlite3, json) plus werkzeug's password
hashing helpers, which are already a Flask dependency — no new packages
required.
"""

import sqlite3
import json
from datetime import datetime, timezone
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

import quiz_engine as qe

DB_PATH = "quiz_app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_state (
    user_id INTEGER PRIMARY KEY,
    q_table_json TEXT NOT NULL,
    concept_mastery_json TEXT NOT NULL,
    current_difficulty INTEGER NOT NULL DEFAULT 1,
    previous_score REAL NOT NULL DEFAULT 5,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    score REAL NOT NULL,
    concept_breakdown_json TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
"""


@contextmanager
def get_conn(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# User accounts
# ---------------------------------------------------------------------------

def create_user(username, password, db_path=DB_PATH):
    """Returns the new user's id. Raises ValueError if the username is taken."""
    with get_conn(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            raise ValueError("Username already taken.")

        password_hash = generate_password_hash(password)
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.now(timezone.utc).isoformat()),
        )
        user_id = cur.lastrowid

        # Seed initial state so get_user_state() always has a row to read.
        conn.execute(
            "INSERT INTO user_state (user_id, q_table_json, concept_mastery_json, "
            "current_difficulty, previous_score) VALUES (?, ?, ?, ?, ?)",
            (user_id, json.dumps(qe.new_q_table().tolist()), json.dumps({}), 1, 5),
        )
        return user_id


def verify_user(username, password, db_path=DB_PATH):
    """Returns the user's id if credentials are correct, else None."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row and check_password_hash(row["password_hash"], password):
        return row["id"]
    return None


def get_username(user_id, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    return row["username"] if row else None


# ---------------------------------------------------------------------------
# Per-user RL / mastery state
# ---------------------------------------------------------------------------

def get_user_state(user_id, db_path=DB_PATH):
    """
    Returns a dict: {q_table (np.ndarray), concept_mastery (defaultdict(list)),
    current_difficulty (int), previous_score (float)}.
    """
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM user_state WHERE user_id = ?", (user_id,)
        ).fetchone()

    if row is None:
        # Shouldn't normally happen (create_user seeds this), but don't crash.
        return {
            "q_table": qe.new_q_table(),
            "concept_mastery": defaultdict(list),
            "current_difficulty": 1,
            "previous_score": 5,
        }

    q_table = np.array(json.loads(row["q_table_json"]))
    concept_mastery = defaultdict(list, json.loads(row["concept_mastery_json"]))
    return {
        "q_table": q_table,
        "concept_mastery": concept_mastery,
        "current_difficulty": row["current_difficulty"],
        "previous_score": row["previous_score"],
    }


def save_user_state(user_id, q_table, concept_mastery, current_difficulty, previous_score, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE user_state
            SET q_table_json = ?, concept_mastery_json = ?, current_difficulty = ?, previous_score = ?
            WHERE user_id = ?
            """,
            (
                json.dumps(q_table.tolist()),
                json.dumps(dict(concept_mastery)),
                current_difficulty,
                previous_score,
                user_id,
            ),
        )


def record_attempt(user_id, difficulty_label, score, concept_breakdown, db_path=DB_PATH):
    """concept_breakdown: dict of {concept: [scores]} for this single attempt."""
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO attempts (user_id, created_at, difficulty, score, concept_breakdown_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                user_id,
                datetime.now(timezone.utc).isoformat(),
                difficulty_label,
                score,
                json.dumps(concept_breakdown),
            ),
        )


def get_attempts(user_id, db_path=DB_PATH):
    """Returns attempts newest-first as a list of dicts."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT created_at, difficulty, score, concept_breakdown_json "
            "FROM attempts WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [
        {
            "created_at": r["created_at"],
            "difficulty": r["difficulty"],
            "score": r["score"],
            "concept_breakdown": json.loads(r["concept_breakdown_json"]),
        }
        for r in rows
    ]


def get_dashboard_summary(user_id, db_path=DB_PATH):
    """
    Aggregates attempts into stats for the dashboard: average score, best/worst
    concept by accuracy, and total attempts. Returns None if no attempts yet.
    """
    attempts = get_attempts(user_id, db_path)
    if not attempts:
        return None

    avg_score = sum(a["score"] for a in attempts) / len(attempts)

    concept_totals = defaultdict(lambda: [0, 0])  # concept -> [correct_sum, count]
    for a in attempts:
        for concept, scores in a["concept_breakdown"].items():
            for s in scores:
                concept_totals[concept][0] += s
                concept_totals[concept][1] += 1

    concept_accuracy = {
        c: (correct / count) for c, (correct, count) in concept_totals.items() if count > 0
    }

    best_concept = max(concept_accuracy, key=concept_accuracy.get) if concept_accuracy else None
    worst_concept = min(concept_accuracy, key=concept_accuracy.get) if concept_accuracy else None

    return {
        "total_attempts": len(attempts),
        "avg_score": round(avg_score, 2),
        "best_concept": best_concept,
        "worst_concept": worst_concept,
        "concept_accuracy": concept_accuracy,
        "recent_attempts": attempts[:10],
    }