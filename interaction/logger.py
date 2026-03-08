"""
Interaction Logger

Thu log mỗi lần user tương tác với hệ thống tìm kiếm sản phẩm.

Các event được log:
  - impression  : query + list sản phẩm đã hiện (shown_items)
  - click       : user click vào sản phẩm
  - no_click    : user bỏ qua (dwell time quá ngắn hoặc không click)
  - next_page   : user chuyển trang (implicit: kết quả trang 1 không đủ tốt)
  - add_to_cart : user thêm vào giỏ (strong positive signal)
  - purchase    : user mua (strongest positive)

Schema SQLite:
  sessions       (session_id, user_id, created_at)
  impressions    (impression_id, session_id, query, shown_items_json, timestamp)
  interactions   (id, impression_id, item_id, event_type, position, timestamp)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    IMPRESSION  = "impression"   # items shown
    CLICK       = "click"
    NO_CLICK    = "no_click"
    NEXT_PAGE   = "next_page"
    ADD_TO_CART = "add_to_cart"
    PURCHASE    = "purchase"


# Reward weights for converting to preference pairs
EVENT_WEIGHT = {
    EventType.NO_CLICK:    0.0,
    EventType.NEXT_PAGE:   0.1,   # implicit negative for page 1 items
    EventType.CLICK:       0.3,
    EventType.ADD_TO_CART: 0.7,
    EventType.PURCHASE:    1.0,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ImpressionRecord:
    session_id: str
    query: str
    shown_items: List[str]    # item_ids in display order
    scores: List[float]       # reranker scores (same order)
    page: int = 1
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class InteractionRecord:
    impression_id: str
    item_id: str
    event_type: EventType
    position: int             # 0-indexed position in shown_items
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


# ---------------------------------------------------------------------------
# SQLite-backed Logger
# ---------------------------------------------------------------------------

class InteractionLogger:
    """
    Thread-safe SQLite interaction logger.

    Usage:
      logger = InteractionLogger("logs/interactions.db")

      # When showing results to user
      imp_id = logger.log_impression(session_id, query, shown_items, scores)

      # When user clicks
      logger.log_interaction(imp_id, item_id="B001", event=EventType.CLICK, position=2)

      # When user adds to cart
      logger.log_interaction(imp_id, item_id="B001", event=EventType.ADD_TO_CART, position=2)
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id  TEXT PRIMARY KEY,
        user_id     TEXT,
        created_at  REAL
    );

    CREATE TABLE IF NOT EXISTS impressions (
        impression_id   TEXT PRIMARY KEY,
        session_id      TEXT,
        query           TEXT,
        shown_items     TEXT,   -- JSON array of item_ids
        scores          TEXT,   -- JSON array of float scores
        page            INTEGER DEFAULT 1,
        timestamp       REAL,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );

    CREATE TABLE IF NOT EXISTS interactions (
        id              TEXT PRIMARY KEY,
        impression_id   TEXT,
        item_id         TEXT,
        event_type      TEXT,
        position        INTEGER,
        timestamp       REAL,
        FOREIGN KEY (impression_id) REFERENCES impressions(impression_id)
    );

    CREATE INDEX IF NOT EXISTS idx_interactions_impression
        ON interactions(impression_id);
    CREATE INDEX IF NOT EXISTS idx_impressions_query
        ON impressions(query);
    """

    def __init__(self, db_path: str = "interactions.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    # ------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _init_db(self) -> None:
        with self._cursor() as cur:
            cur.executescript(self._DDL)

    # ------------------------------------------------------------------
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new session and return session_id."""
        session_id = str(uuid.uuid4())
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO sessions VALUES (?, ?, ?)",
                (session_id, user_id or "anonymous", time.time()),
            )
        return session_id

    # ------------------------------------------------------------------
    def log_impression(
        self,
        session_id: str,
        query: str,
        shown_items: List[str],
        scores: Optional[List[float]] = None,
        page: int = 1,
    ) -> str:
        """
        Log that a list of items was shown to the user.
        Returns impression_id (use this for subsequent interaction logs).
        """
        impression_id = str(uuid.uuid4())
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO impressions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    impression_id,
                    session_id,
                    query,
                    json.dumps(shown_items),
                    json.dumps(scores or []),
                    page,
                    time.time(),
                ),
            )
        return impression_id

    # ------------------------------------------------------------------
    def log_interaction(
        self,
        impression_id: str,
        item_id: str,
        event: EventType,
        position: int,
    ) -> None:
        """Log a single user interaction."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), impression_id, item_id, event.value, position, time.time()),
            )

    def log_no_click(self, impression_id: str, shown_items: List[str]) -> None:
        """
        Convenience: mark all shown items as no_click.
        Call this when user navigates away without clicking anything.
        """
        for pos, item_id in enumerate(shown_items):
            self.log_interaction(impression_id, item_id, EventType.NO_CLICK, pos)

    def log_next_page(self, impression_id: str, shown_items: List[str]) -> None:
        """
        Mark all items on current page as 'next_page' signal
        (user was not satisfied enough to stay on page 1).
        """
        for pos, item_id in enumerate(shown_items):
            self.log_interaction(impression_id, item_id, EventType.NEXT_PAGE, pos)

    # ------------------------------------------------------------------
    def get_impression(self, impression_id: str) -> Optional[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM impressions WHERE impression_id = ?", (impression_id,))
            row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["shown_items"] = json.loads(d["shown_items"])
        d["scores"] = json.loads(d["scores"])
        return d

    def get_interactions(self, impression_id: str) -> List[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM interactions WHERE impression_id = ? ORDER BY timestamp",
                (impression_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_all_impressions(self, limit: int = 100_000) -> List[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM impressions ORDER BY timestamp LIMIT ?", (limit,))
            rows = cur.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["shown_items"] = json.loads(d["shown_items"])
            d["scores"] = json.loads(d["scores"])
            result.append(d)
        return result

    def stats(self) -> dict:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM impressions")
            n_imp = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM interactions")
            n_inter = cur.fetchone()[0]
            cur.execute("SELECT event_type, COUNT(*) FROM interactions GROUP BY event_type")
            event_counts = {r[0]: r[1] for r in cur.fetchall()}
        return {
            "impressions": n_imp,
            "interactions": n_inter,
            "event_counts": event_counts,
        }
