"""
Real-Time Event Bus & Alert Dispatcher for SpaceNetra.

Provides pub/sub topic routing for live satellite change detection alerts and stream events.
"""

from typing import Any, Callable, Dict, List
import logging

logger = logging.getLogger(__name__)


class EventBus:
    """In-memory publish-subscribe event dispatcher for real-time change events."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe a handler callback to a specific event topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        if handler not in self._subscribers[topic]:
            self._subscribers[topic].append(handler)
            logger.info(f"Subscribed handler {handler.__name__ if hasattr(handler, '__name__') else handler} to topic '{topic}'")

    def unsubscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Unsubscribe a handler callback from a topic."""
        if topic in self._subscribers and handler in self._subscribers[topic]:
            self._subscribers[topic].remove(handler)

    def publish(self, topic: str, event_data: Dict[str, Any]) -> int:
        """Publish an event payload to all topic subscribers."""
        handlers = self._subscribers.get(topic, [])
        dispatched_count = 0
        for handler in handlers:
            try:
                handler(event_data)
                dispatched_count += 1
            except Exception as e:
                logger.error(f"Error executing event handler on topic '{topic}': {e}")
        return dispatched_count

    def clear(self) -> None:
        """Clear all active subscriptions."""
        self._subscribers.clear()


# Global singleton instance for app-wide event routing
default_event_bus = EventBus()
