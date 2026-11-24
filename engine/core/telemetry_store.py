"""Non-blocking SQLite WAL telemetry & state store for AetherMix v5.1.

All writes are enqueued onto an in-memory queue and drained by a dedicated
background thread so the 50 Hz real-time control loop never blocks on disk I/O.
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "storage" / "telemetry.db"


@dataclass(slots=True)
class _WriteOp:
    kind: str
    args: tuple[Any, ...] = ()
    event: threading.Event | None = None


class TelemetryStore:
    """SQLite WAL-mode store with background thread/queue writer."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH, max_queue_size: int = 4096) -> None:
        self.db_path = Path(db_path)
        self._max_queue_size = max_queue_size
        self._queue: queue.Queue[_WriteOp] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._lock = threading.Lock()
        self._dropped_writes = 0

    @property
    def dropped_writes(self) -> int:
        return self._dropped_writes

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
            self._closed = False
            self._started = True
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="aethermix-telemetry-writer",
                daemon=True,
            )
            self._thread.start()

    def close(self, timeout: float = 3.0) -> None:
        with self._lock:
            if not self._started or self._closed:
                return
            self._closed = True
            thread = self._thread
            self._thread = None
            self._started = False

        done_evt = threading.Event()
        try:
            self._queue.put(_WriteOp(kind="shutdown", event=done_evt), timeout=timeout)
        except queue.Full:
            pass
        if thread is not None:
            thread.join(timeout=timeout)

    def flush(self, timeout: float = 3.0) -> bool:
        """Wait until all previously enqueued writes have been committed to SQLite."""
        if not self._started:
            return True
        evt = threading.Event()
        self._queue.put(_WriteOp(kind="flush", event=evt), timeout=timeout)
        return evt.wait(timeout=timeout)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS telemetry_1hz (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts_ms INTEGER NOT NULL,
                        active_phase TEXT NOT NULL,
                        autonomy_mode TEXT NOT NULL,
                        master_lufs REAL,
                        loop_p95_ms REAL,
                        loop_p99_ms REAL,
                        summary_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_telemetry_1hz_ts
                        ON telemetry_1hz (ts_ms);

                    CREATE TABLE IF NOT EXISTS discrete_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts_ms INTEGER NOT NULL,
                        category TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_discrete_events_ts
                        ON discrete_events (ts_ms);

                    CREATE TABLE IF NOT EXISTS last_known_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        updated_at_ms INTEGER NOT NULL,
                        armed_ids_json TEXT NOT NULL,
                        channels_cc_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS policy_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version INTEGER NOT NULL,
                        created_at_ms INTEGER NOT NULL,
                        state_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_policy_snapshots_ver
                        ON policy_snapshots (version DESC, id DESC);

                    CREATE TABLE IF NOT EXISTS pending_review_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sample_id TEXT NOT NULL DEFAULT '',
                        created_at_ms INTEGER NOT NULL,
                        sample_json TEXT NOT NULL
                    );
                    """
                )
                cols = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(pending_review_samples)").fetchall()
                }
                if "sample_id" not in cols:
                    conn.execute(
                        "ALTER TABLE pending_review_samples ADD COLUMN sample_id TEXT NOT NULL DEFAULT ''"
                    )
        finally:
            conn.close()

    def _enqueue(self, op: _WriteOp) -> bool:
        if not self._started:
            self.start()
        try:
            self._queue.put_nowait(op)
            return True
        except queue.Full:
            self._dropped_writes += 1
            logger.warning("TelemetryStore queue full; dropped %s write", op.kind)
            return False

    def _writer_loop(self) -> None:
        conn = self._connect()
        try:
            while True:
                op = self._queue.get()
                try:
                    if op.kind == "shutdown":
                        conn.commit()
                        if op.event is not None:
                            op.event.set()
                        break
                    if op.kind == "flush":
                        conn.commit()
                        if op.event is not None:
                            op.event.set()
                        continue
                    self._apply_op(conn, op)
                    # Batch-drain any immediately queued items before committing
                    drained = 0
                    shutdown_evt: threading.Event | None = None
                    flush_evts: list[threading.Event] = []
                    while drained < 64:
                        try:
                            nxt = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if nxt.kind == "shutdown":
                            shutdown_evt = nxt.event
                            break
                        if nxt.kind == "flush":
                            if nxt.event is not None:
                                flush_evts.append(nxt.event)
                            break
                        self._apply_op(conn, nxt)
                        drained += 1
                    conn.commit()
                    for ev in flush_evts:
                        ev.set()
                    if shutdown_evt is not None:
                        shutdown_evt.set()
                        break
                except sqlite3.Error:
                    logger.exception("TelemetryStore writer error on op %s", op.kind)
                    try:
                        conn.rollback()
                    except sqlite3.Error as rb_exc:
                        logger.debug("TelemetryStore rollback failed: %s", rb_exc)
                    if op.event is not None:
                        op.event.set()
        finally:
            conn.close()

    def _apply_op(self, conn: sqlite3.Connection, op: _WriteOp) -> None:
        if op.kind == "telemetry_1hz":
            ts_ms, active_phase, autonomy_mode, master_lufs, loop_p95_ms, loop_p99_ms, summary_json = op.args
            conn.execute(
                """
                INSERT INTO telemetry_1hz
                    (ts_ms, active_phase, autonomy_mode, master_lufs, loop_p95_ms, loop_p99_ms, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts_ms, active_phase, autonomy_mode, master_lufs, loop_p95_ms, loop_p99_ms, summary_json),
            )
        elif op.kind == "discrete_event":
            ts_ms, category, payload_json = op.args
            conn.execute(
                "INSERT INTO discrete_events (ts_ms, category, payload_json) VALUES (?, ?, ?)",
                (ts_ms, category, payload_json),
            )
        elif op.kind == "last_known_state":
            updated_at_ms, armed_ids_json, channels_cc_json = op.args
            conn.execute(
                """
                INSERT INTO last_known_state (id, updated_at_ms, armed_ids_json, channels_cc_json)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at_ms = excluded.updated_at_ms,
                    armed_ids_json = excluded.armed_ids_json,
                    channels_cc_json = excluded.channels_cc_json
                """,
                (updated_at_ms, armed_ids_json, channels_cc_json),
            )
        elif op.kind == "policy_snapshot":
            version, created_at_ms, state_json = op.args
            conn.execute(
                "INSERT INTO policy_snapshots (version, created_at_ms, state_json) VALUES (?, ?, ?)",
                (version, created_at_ms, state_json),
            )
        elif op.kind == "queue_review":
            sample_id, created_at_ms, sample_json = op.args
            conn.execute(
                "INSERT INTO pending_review_samples (sample_id, created_at_ms, sample_json) VALUES (?, ?, ?)",
                (sample_id, created_at_ms, sample_json),
            )
        elif op.kind == "clear_reviews":
            sample_ids = op.args[0] if op.args else None
            if sample_ids:
                placeholders = ",".join("?" for _ in sample_ids)
                conn.execute(
                    f"DELETE FROM pending_review_samples WHERE sample_id IN ({placeholders})",
                    tuple(sample_ids),
                )
            else:
                conn.execute("DELETE FROM pending_review_samples")

    # ------------------------------------------------------------------
    # Public non-blocking write APIs
    # ------------------------------------------------------------------

    def log_1hz_summary(
        self,
        active_phase: str = "SPOKEN_WORD",
        autonomy_mode: str = "SHADOW_OBSERVE",
        master_lufs: float = -60.0,
        loop_p95_ms: float = 0.0,
        loop_p99_ms: float = 0.0,
        summary: dict[str, Any] | None = None,
        ts_ms: int | None = None,
        **extra: Any,
    ) -> None:
        now_ms = int(time.time() * 1000) if ts_ms is None else int(ts_ms)
        payload = dict(summary or {})
        if extra:
            payload.update(extra)
        self._enqueue(
            _WriteOp(
                kind="telemetry_1hz",
                args=(
                    now_ms,
                    str(active_phase),
                    str(autonomy_mode),
                    float(master_lufs),
                    float(loop_p95_ms),
                    float(loop_p99_ms),
                    json.dumps(payload, separators=(",", ":")),
                ),
            )
        )

    def log_event(
        self,
        category: str,
        payload: dict[str, Any],
        ts_ms: int | None = None,
    ) -> None:
        now_ms = int(time.time() * 1000) if ts_ms is None else int(ts_ms)
        self._enqueue(
            _WriteOp(
                kind="discrete_event",
                args=(now_ms, str(category), json.dumps(payload, separators=(",", ":"))),
            )
        )

    def save_last_known_state(
        self,
        armed_ids: list[str],
        channels_cc: dict[str, int],
    ) -> None:
        now_ms = int(time.time() * 1000)
        self._enqueue(
            _WriteOp(
                kind="last_known_state",
                args=(
                    now_ms,
                    json.dumps(list(armed_ids), separators=(",", ":")),
                    json.dumps({str(k): int(v) for k, v in channels_cc.items()}, separators=(",", ":")),
                ),
            )
        )

    def save_policy_snapshot(self, version: int, state_dict: dict[str, Any]) -> None:
        now_ms = int(time.time() * 1000)
        self._enqueue(
            _WriteOp(
                kind="policy_snapshot",
                args=(int(version), now_ms, json.dumps(state_dict, separators=(",", ":"))),
            )
        )

    def queue_review_sample(self, sample: dict[str, Any]) -> str:
        now_ms = int(time.time() * 1000)
        enriched = dict(sample)
        sample_id = str(enriched.get("id") or f"rev_{uuid.uuid4().hex[:12]}")
        enriched["id"] = sample_id
        enriched.setdefault("ts_ms", now_ms)
        self._enqueue(
            _WriteOp(
                kind="queue_review",
                args=(sample_id, now_ms, json.dumps(enriched, separators=(",", ":"))),
            )
        )
        return sample_id

    def delete_queued_reviews(self, sample_ids: Sequence[str] | None = None) -> None:
        ids_list = [str(sid) for sid in sample_ids] if sample_ids else None
        self._enqueue(_WriteOp(kind="clear_reviews", args=(ids_list,)))

    def clear_queued_reviews(self) -> None:
        self.delete_queued_reviews(None)

    # ------------------------------------------------------------------
    # Public read APIs (flush pending writes first for strong consistency)
    # ------------------------------------------------------------------

    def load_last_known_state(self) -> dict[str, Any]:
        if not self._started:
            self.start()
        self.flush()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT updated_at_ms, armed_ids_json, channels_cc_json FROM last_known_state WHERE id = 1"
            ).fetchone()
            if row is None:
                return {"armed_ids": [], "channels_cc": {}, "updated_at_ms": None}
            return {
                "updated_at_ms": int(row[0]),
                "armed_ids": list(json.loads(row[1])),
                "channels_cc": {str(k): int(v) for k, v in json.loads(row[2]).items()},
            }
        finally:
            conn.close()

    def load_latest_policy_snapshot(self) -> tuple[int, dict[str, Any]] | None:
        if not self._started:
            self.start()
        self.flush()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT version, state_json FROM policy_snapshots ORDER BY version DESC, id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return int(row[0]), dict(json.loads(row[1]))
        finally:
            conn.close()

    def list_queued_reviews(self) -> list[dict[str, Any]]:
        if not self._started:
            self.start()
        self.flush()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, sample_id, sample_json FROM pending_review_samples ORDER BY id ASC"
            ).fetchall()
            out: list[dict[str, Any]] = []
            for row_id, sid, raw in rows:
                item = dict(json.loads(raw))
                if not item.get("id"):
                    item["id"] = str(sid or f"rev_{row_id}")
                out.append(item)
            return out
        finally:
            conn.close()

    def list_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self._started:
            self.start()
        self.flush()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT ts_ms, category, payload_json FROM discrete_events ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [
                {"ts_ms": int(r[0]), "category": str(r[1]), "payload": json.loads(r[2])}
                for r in reversed(rows)
            ]
        finally:
            conn.close()

    def list_recent_telemetry(self, limit: int = 60) -> list[dict[str, Any]]:
        if not self._started:
            self.start()
        self.flush()
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT ts_ms, active_phase, autonomy_mode, master_lufs, loop_p95_ms, loop_p99_ms, summary_json
                FROM telemetry_1hz ORDER BY id DESC LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [
                {
                    "ts_ms": int(r[0]),
                    "active_phase": str(r[1]),
                    "autonomy_mode": str(r[2]),
                    "master_lufs": float(r[3]) if r[3] is not None else -60.0,
                    "loop_p95_ms": float(r[4]) if r[4] is not None else 0.0,
                    "loop_p99_ms": float(r[5]) if r[5] is not None else 0.0,
                    "summary": json.loads(r[6]),
                }
                for r in reversed(rows)
            ]
        finally:
            conn.close()
