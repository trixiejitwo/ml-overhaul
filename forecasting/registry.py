"""
SQLite run registry: logs each training run's params and holdout metrics per
model, tracks which run is "active" per model (the one the app reads), and
stores a series-stats snapshot at training time for staleness comparison.

Switching the active run for a model is atomic: clearing the previous
is_active=1 row and setting the new one happens inside a single transaction,
so it can never collide with the partial unique index (one active run per
model_name) or leave a model with zero/two active rows if interrupted.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name    TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    mae           REAL NOT NULL,
    rmse          REAL NOT NULL,
    smape         REAL NOT NULL,
    created_at    TEXT NOT NULL,
    data_as_of    TEXT NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 0,
    notes         TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_per_model
    ON runs(model_name) WHERE is_active = 1;

CREATE TABLE IF NOT EXISTS series_stats_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id),
    mean          REAL,
    std           REAL,
    recent_trend  REAL,
    created_at    TEXT NOT NULL,
    data_as_of    TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect(db_path: str = None):
    conn = sqlite3.connect(db_path or config.REGISTRY_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = None) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


def log_run(
    model_name: str,
    params: dict,
    mae: float,
    rmse: float,
    smape: float,
    data_as_of,
    series_stats: dict = None,
    notes: str = None,
    db_path: str = None,
) -> int:
    """Insert a new run row (inactive by default). Returns the new run's id."""
    created_at = _now_iso()
    data_as_of_iso = data_as_of.isoformat() if hasattr(data_as_of, "isoformat") else str(data_as_of)

    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO runs (model_name, params_json, mae, rmse, smape, created_at, data_as_of, is_active, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (model_name, json.dumps(params), mae, rmse, smape, created_at, data_as_of_iso, notes),
        )
        run_id = cur.lastrowid

        if series_stats:
            conn.execute(
                """INSERT INTO series_stats_snapshots (run_id, mean, std, recent_trend, created_at, data_as_of)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    series_stats.get("mean"),
                    series_stats.get("std"),
                    series_stats.get("recent_trend"),
                    created_at,
                    data_as_of_iso,
                ),
            )
    return run_id


def set_active(model_name: str, run_id: int, db_path: str = None) -> None:
    """
    Atomically make `run_id` the active run for `model_name`: clear any
    existing is_active=1 row for this model, then activate the requested
    run, both within one transaction (sqlite3's connection context manager
    commits once on success or rolls back entirely on error -- there is no
    window where two rows are active or zero rows are active as a result of
    this call).
    """
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET is_active = 0 WHERE model_name = ? AND is_active = 1",
            (model_name,),
        )
        cur = conn.execute(
            "UPDATE runs SET is_active = 1 WHERE id = ? AND model_name = ?",
            (run_id, model_name),
        )
        if cur.rowcount == 0:
            raise ValueError(f"No run with id={run_id} for model_name={model_name!r}")


def get_active_run(model_name: str, db_path: str = None) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE model_name = ? AND is_active = 1", (model_name,)
        ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def get_active_runs_all_models(db_path: str = None) -> dict:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM runs WHERE is_active = 1").fetchall()
    return {row["model_name"]: _row_to_dict(row) for row in rows}


def list_runs(model_name: str = None, limit: int = 50, db_path: str = None) -> list:
    with _connect(db_path) as conn:
        if model_name:
            rows = conn.execute(
                "SELECT * FROM runs WHERE model_name = ? ORDER BY created_at DESC LIMIT ?",
                (model_name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_series_stats_snapshot(run_id: int, db_path: str = None) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM series_stats_snapshots WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
    return dict(row) if row else None


def get_series_stats_at_last_training(model_name: str, db_path: str = None) -> dict | None:
    active = get_active_run(model_name, db_path=db_path)
    if active is None:
        return None
    return get_series_stats_snapshot(active["id"], db_path=db_path)


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json"))
    return d
