"""
AI Prediction Provenance Lineage Service for SpaceNetra.

Generates and retrieves end-to-end model execution trace records for audit compliance.
"""

import uuid
from typing import Any, Dict, Optional
from backend.database.connection import DatabaseManager
from backend.database.models import ProvenanceModel


class ProvenanceService:
    """
    Service layer for tracking model execution lineage and audit provenance.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self.db_manager.create_tables()

    def create_provenance_record(
        self,
        event_id: str,
        model_name: str = "siamese_unet",
        checkpoint_hash: str = "sha256_default_weights",
    ) -> Dict[str, Any]:
        """
        Records model provenance trace for a spatial change detection event.

        Args:
            event_id: Target event ID.
            model_name: Neural network architecture name.
            checkpoint_hash: Model checkpoint SHA-256 hash string.

        Returns:
            Dict containing created provenance trace record.
        """
        provenance_id = f"PROV_{uuid.uuid4().hex[:8].upper()}"

        with self.db_manager.get_session() as session:
            prov = ProvenanceModel(
                provenance_id=provenance_id,
                event_id=event_id,
                model_name=model_name,
                checkpoint_hash=checkpoint_hash,
            )
            session.add(prov)
            result = prov.to_dict()

        return result

    def get_provenance_trace(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves provenance trace record for a specific change event.

        Args:
            event_id: Target event ID.

        Returns:
            Dict containing provenance trace or None if not found.
        """
        with self.db_manager.get_session() as session:
            prov = session.query(ProvenanceModel).filter_by(event_id=event_id).first()
            if prov is not None:
                return prov.to_dict()
            return None
