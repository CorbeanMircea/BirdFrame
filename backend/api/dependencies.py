"""
FastAPI dependencies shared across route modules.

Import these with FastAPI's Depends() mechanism:

    from backend.api.dependencies import get_db_session
    from fastapi import Depends
    from sqlalchemy.orm import Session

    @router.get("/example")
    def example(session: Session = Depends(get_db_session)):
        ...
"""

import sys
from pathlib import Path

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database.engine import get_session
from backend.database.repository import DetectionRepository


def get_db_session():
    """
    Yield a SQLAlchemy session for the duration of a request.
    Commits on clean exit, rolls back on exception, always closes.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_repository() -> DetectionRepository:
    """Return a DetectionRepository instance."""
    return DetectionRepository()