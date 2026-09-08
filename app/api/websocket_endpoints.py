"""
FastAPI WebSocket Endpoints for Real-Time Satellite Change Alerts.
"""

from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.pipeline.event_bus import default_event_bus

router = APIRouter(prefix="/ws", tags=["Real-Time WebSockets"])


class ConnectionManager:
    """Manages active WebSocket connections for live alert broadcasting."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


@router.websocket("/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """Real-time WebSocket endpoint streaming satellite change detection alerts."""
    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "CONNECTION_ESTABLISHED",
            "message": "Connected to SpaceNetra Real-Time Alert Event Stream",
        })
        while True:
            # Keep connection alive listening for ping / client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
