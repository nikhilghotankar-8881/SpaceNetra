"""
Unit tests for SpaceNetra EventBus, StreamConsumer, and WebSocket Alert Stream (Phase 28).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.pipeline.event_bus import EventBus
from src.pipeline.stream_consumer import StreamConsumer


@pytest.fixture
def client():
    return TestClient(app)


def test_event_bus_subscribe_and_publish():
    bus = EventBus()
    received_events = []

    def handle_event(data):
        received_events.append(data)

    bus.subscribe("TEST_TOPIC", handle_event)
    count = bus.publish("TEST_TOPIC", {"key": "value"})

    assert count == 1
    assert len(received_events) == 1
    assert received_events[0]["key"] == "value"

    bus.unsubscribe("TEST_TOPIC", handle_event)
    bus.publish("TEST_TOPIC", {"key": "value2"})
    assert len(received_events) == 1


def test_stream_consumer_processing():
    bus = EventBus()
    alerts = []

    bus.subscribe("CHANGE_ALERT", lambda data: alerts.append(data))
    consumer = StreamConsumer(event_bus=bus)

    # Low change pair
    t1_low = np.zeros((100, 100, 3), dtype=np.uint8)
    t2_low = np.zeros((100, 100, 3), dtype=np.uint8)
    res_low = consumer.process_stream_event(t1_low, t2_low, confidence_threshold=0.5)

    assert res_low["alert_triggered"] is False
    assert len(alerts) == 0

    # High change pair
    t1_high = np.zeros((100, 100, 3), dtype=np.uint8)
    t2_high = np.ones((100, 100, 3), dtype=np.uint8) * 255
    res_high = consumer.process_stream_event(t1_high, t2_high, confidence_threshold=0.5)

    assert res_high["alert_triggered"] is True
    assert len(alerts) == 1
    assert alerts[0]["change_probability"] > 0.5


def test_websocket_alerts_connection(client):
    with client.websocket_connect("/api/ws/alerts") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "CONNECTION_ESTABLISHED"

        websocket.send_text("ping")
        resp = websocket.receive_json()
        assert resp["type"] == "pong"
