"""SQLite connection manager."""

import sqlite3
from pathlib import Path


class Database:

    def __init__(self, db_path: str = "storage/mlops.sqlite") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        return self.connection.execute(sql, params)

    def executescript(self, sql: str) -> None:
        self.connection.executescript(sql)

    def commit(self) -> None:
        self.connection.commit()

    def fetchall(self, sql: str, params: tuple | list = ()) -> list[dict]:
        rows = self.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
