"""Schema migration runner.

Migrations are plain .sql files in the migrations/ directory,
named with a numeric prefix that determines execution order:

    001_create_raw_batches.sql
    002_create_data_quality.sql
    ...

The migrator tracks applied migrations in a _migrations meta-table
and only runs new ones.
"""

from pathlib import Path

from src.database.connection import Database

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class Migrator:

    def __init__(self, db: Database, migrations_dir: Path | None = None) -> None:
        self._db = db
        self._dir = migrations_dir or _MIGRATIONS_DIR

    def _ensure_meta_table(self) -> None:
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS _migrations (
                   id        INTEGER PRIMARY KEY AUTOINCREMENT,
                   name      TEXT    NOT NULL UNIQUE,
                   applied_at TEXT   DEFAULT (datetime('now'))
               )"""
        )
        self._db.commit()

    def _applied(self) -> set[str]:
        rows = self._db.fetchall("SELECT name FROM _migrations")
        return {r["name"] for r in rows}

    def _pending(self) -> list[Path]:
        applied = self._applied()
        files = sorted(self._dir.glob("*.sql"))
        return [f for f in files if f.name not in applied]

    def migrate(self) -> list[str]:
        self._ensure_meta_table()
        pending = self._pending()
        applied: list[str] = []

        for migration_file in pending:
            sql = migration_file.read_text(encoding="utf-8")
            self._db.executescript(sql)
            self._db.execute(
                "INSERT INTO _migrations (name) VALUES (?)",
                (migration_file.name,),
            )
            self._db.commit()
            applied.append(migration_file.name)

        return applied

    def status(self) -> dict[str, bool]:
        self._ensure_meta_table()
        applied = self._applied()
        files = sorted(self._dir.glob("*.sql"))
        return {f.name: f.name in applied for f in files}
