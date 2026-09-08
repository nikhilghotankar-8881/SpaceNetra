"""
Database Connection & Session Manager for SpaceNetra.

Supports SQLite local storage and PostgreSQL / PostGIS database connections.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database.models import Base

DEFAULT_DB_URI = "sqlite:///data/spacenetra.db"


class DatabaseManager:
    """
    Unified Session Manager for Relational & Spatial Database Operations.
    """

    def __init__(self, db_uri: str = DEFAULT_DB_URI):
        if db_uri.startswith("sqlite"):
            # Ensure SQLite database directory exists
            db_path = db_uri.replace("sqlite:///", "")
            if db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            db_uri,
            connect_args={"check_same_thread": False} if db_uri.startswith("sqlite") else {},
            echo=False,
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """
        Creates all database tables defined in Base metadata.
        """
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """
        Drops all database tables.
        """
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager providing transactional SQLAlchemy database session.
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
