"""
Database engine and session factory for BirdFrame.

Usage anywhere in the backend:

    from backend.database.engine import get_session

    with get_session() as session:
        session.add(some_object)
        session.commit()
"""

import sys
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

# Allow `import config` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.database.models import Base

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_engine(
    config.DATABASE_URL,
    # echo=True would print every SQL statement — useful for deep debugging,
    # leave off for normal use
    echo=False,
    connect_args={
        # Required for SQLite when used across threads (e.g. in the audio pipeline)
        "check_same_thread": False,
    },
)

# Enable WAL mode for SQLite: allows concurrent reads during a write,
# which matters once the audio pipeline and the API run simultaneously.
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,   # keep objects usable after session.commit()
)


def get_session() -> Session:
    """
    Return a new SQLAlchemy Session.

    Intended for use as a context manager:

        with get_session() as session:
            ...

    The session is committed automatically on clean exit and rolled
    back on exception.
    """
    return SessionLocal()


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create all tables that do not yet exist.

    Safe to call on every startup — existing tables are never dropped.
    """
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    print(f"Initialising database at: {config.DATABASE_URL}")
    init_db()
    print("Done.")