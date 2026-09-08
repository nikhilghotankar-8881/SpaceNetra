"""
Analyst Feedback Review Service for SpaceNetra.

Handles analyst review submissions (ACCEPT, REJECT, REVIEW), database persistence,
and feedback analytics statistics computation.
"""

import uuid
from typing import Any, Dict, Optional
from backend.database.connection import DatabaseManager
from backend.database.models import AnalystFeedbackModel


class FeedbackService:
    """
    Service layer for analyst feedback workflow and active learning statistics.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self.db_manager.create_tables()

    def submit_feedback(
        self,
        event_id: str,
        decision: str,
        analyst_id: str = "analyst_001",
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Submits and stores analyst decision on a spatial change event.

        Args:
            event_id: Target event ID (e.g. 'EVENT_0001').
            decision: Review decision ('ACCEPT', 'REJECT', or 'REVIEW').
            analyst_id: User/analyst identifier.
            notes: Optional review commentary.

        Returns:
            Dict containing created feedback record.
        """
        valid_decisions = ["ACCEPT", "REJECT", "REVIEW"]
        decision_upper = decision.upper()
        if decision_upper not in valid_decisions:
            raise ValueError(f"Invalid decision '{decision}'. Must be one of {valid_decisions}")

        feedback_id = f"FB_{uuid.uuid4().hex[:8].upper()}"

        with self.db_manager.get_session() as session:
            feedback = AnalystFeedbackModel(
                feedback_id=feedback_id,
                event_id=event_id,
                analyst_id=analyst_id,
                decision=decision_upper,
                notes=notes,
            )
            session.add(feedback)
            result = feedback.to_dict()

        return result

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Computes aggregate analyst feedback metrics and acceptance rate.

        Returns:
            Dict containing total reviews, decision counts, and acceptance rate %.
        """
        with self.db_manager.get_session() as session:
            all_feedback = session.query(AnalystFeedbackModel).all()

            total = len(all_feedback)
            if total == 0:
                return {
                    "total_reviews": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "review_count": 0,
                    "acceptance_rate_pct": 0.0,
                }

            accepted = sum(1 for f in all_feedback if f.decision == "ACCEPT")
            rejected = sum(1 for f in all_feedback if f.decision == "REJECT")
            reviews = sum(1 for f in all_feedback if f.decision == "REVIEW")

            acceptance_rate = float((accepted / total) * 100.0)

            return {
                "total_reviews": total,
                "accepted_count": accepted,
                "rejected_count": rejected,
                "review_count": reviews,
                "acceptance_rate_pct": round(acceptance_rate, 2),
            }
