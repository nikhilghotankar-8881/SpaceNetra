"""
SQLAlchemy ORM Models for SpaceNetra Relational & Spatial Database.

Defines database schemas for satellite scenes, image tiles, change events,
analyst feedback, and prediction provenance tracking.
"""

from datetime import datetime, timezone
from typing import Any, Dict
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class SceneModel(Base):
    """Satellite Scene Metadata Table."""
    __tablename__ = "scenes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(String(100), unique=True, nullable=False, index=True)
    sensor = Column(String(50), default="Sentinel-2")
    acquisition_date = Column(String(50), nullable=True)
    cloud_cover = Column(Float, default=0.0)
    filepath = Column(String(255), nullable=True)
    bounds = Column(String(255), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "scene_id": self.scene_id,
            "sensor": self.sensor,
            "acquisition_date": self.acquisition_date,
            "cloud_cover": self.cloud_cover,
            "filepath": self.filepath,
            "bounds": self.bounds,
        }


class TileModel(Base):
    """Satellite Image Tile Table."""
    __tablename__ = "tiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tile_id = Column(String(100), unique=True, nullable=False, index=True)
    scene_id = Column(String(100), ForeignKey("scenes.scene_id"), nullable=True)
    center_lat = Column(Float, nullable=True)
    center_lon = Column(Float, nullable=True)
    embedding_id = Column(String(100), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tile_id": self.tile_id,
            "scene_id": self.scene_id,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "embedding_id": self.embedding_id,
        }


class ChangeEventModel(Base):
    """Spatial Change Events Table."""
    __tablename__ = "change_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(100), unique=True, nullable=False, index=True)
    location = Column(String(255), nullable=True)
    area_m2 = Column(Float, default=0.0)
    first_observed = Column(String(50), nullable=True)
    confidence_score = Column(Float, default=1.0)
    status = Column(String(50), default="DETECTED")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "location": self.location,
            "area_m2": self.area_m2,
            "first_observed": self.first_observed,
            "confidence_score": self.confidence_score,
            "status": self.status,
        }


class AnalystFeedbackModel(Base):
    """Human Analyst Feedback Reviews Table."""
    __tablename__ = "analyst_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feedback_id = Column(String(100), unique=True, nullable=False, index=True)
    event_id = Column(String(100), ForeignKey("change_events.event_id"), nullable=False)
    analyst_id = Column(String(100), default="analyst_001")
    decision = Column(String(50), nullable=False)  # ACCEPT, REJECT, REVIEW
    notes = Column(Text, nullable=True)
    created_at = Column(String(50), default=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "feedback_id": self.feedback_id,
            "event_id": self.event_id,
            "analyst_id": self.analyst_id,
            "decision": self.decision,
            "notes": self.notes,
            "created_at": self.created_at,
        }


class ProvenanceModel(Base):
    """Prediction Provenance Traceability Table."""
    __tablename__ = "provenance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provenance_id = Column(String(100), unique=True, nullable=False, index=True)
    event_id = Column(String(100), ForeignKey("change_events.event_id"), nullable=False)
    model_name = Column(String(100), default="siamese_unet")
    checkpoint_hash = Column(String(100), nullable=True)
    timestamp = Column(String(50), default=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "provenance_id": self.provenance_id,
            "event_id": self.event_id,
            "model_name": self.model_name,
            "checkpoint_hash": self.checkpoint_hash,
            "timestamp": self.timestamp,
        }


class UserModel(Base):
    """User Credentials & Role-Based Access Control Table."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="ANALYST")
    is_active = Column(Integer, default=1)
    created_at = Column(String(50), default=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": bool(self.is_active),
            "created_at": self.created_at,
        }

