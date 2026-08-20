"""
bridge/main.py
Versalis — Connection bridge between Unreal Engine and the clinician dashboard.

Two features:
  1. VR Mirror   — Unreal posts JPEG frames here every 100ms.
                   Dashboard fetches the latest frame every second.
  2. Encouragement — Dashboard posts a command here.
                     FastAPI forwards it to Unreal via WebSocket.

Run with:
    python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload

Port 8002 keeps this separate from the LLM service on 8001.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="Versalis Bridge Service")

# Allow the Streamlit dashboard (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# SECTION 1 — In-memory frame store for the VR mirror
# =============================================================================

class FrameStore:
    """Holds the latest JPEG frame sent by Unreal. Thread-safe via asyncio."""

    def __init__(self) -> None:
        self._frame: Optional[bytes] = None
        self._received_at: float = 0.0
        self._frame_count: int = 0

    def update(self, jpeg_bytes: bytes) -> None:
        self._frame = jpeg_bytes
        self._received_at = time.time()
        self._frame_count += 1

    def get(self) -> Optional[bytes]:
        return self._frame

    def age_seconds(self) -> float:
        if self._frame is None:
            return 999.0
        return time.time() - self._received_at

    def frame_count(self) -> int:
        return self._frame_count


frame_store = FrameStore()


# =============================================================================
# SECTION 2 — Connected Unreal WebSocket clients
# =============================================================================

connected_unreal_clients: List[WebSocket] = []


async def send_to_unreal(event: dict) -> int:
    """
    Broadcast a command dict to all connected Unreal instances.
    Returns the number of clients successfully reached.
    """
    disconnected = []
    sent_count = 0

    for client in connected_unreal_clients:
        try:
            await client.send_json(event)
            sent_count += 1
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        connected_unreal_clients.remove(client)

    return sent_count


# =============================================================================
# SECTION 3 — Request / response models
# =============================================================================

class FramePayload(BaseModel):
    """What Unreal sends when posting a mirror frame."""
    frame: str        # base64-encoded JPEG
    width: int = 0
    height: int = 0

class CommandPayload(BaseModel):
    """
    What the dashboard sends to control the Unreal session.

    Supported types:
      show_encouragement  — shows a popup on the headset
      set_intensity       — changes the VR scenario intensity band
      stop_session        — triggers safe session termination
    """
    type: str                   # "show_encouragement" | "set_intensity" | "stop_session"
    message: Optional[str] = None   # used by show_encouragement
    band: Optional[str] = None      # used by set_intensity: very_low|low|medium|high|very_high


# =============================================================================
# SECTION 4 — VR Mirror endpoints
# =============================================================================

@app.post("/mirror/frame")
async def receive_frame(payload: FramePayload):
    """
    Unreal posts a base64-encoded JPEG frame here.
    Called by MirrorSenderComponent every 100ms (10fps).
    """
    try:
        jpeg_bytes = base64.b64decode(payload.frame)
        frame_store.update(jpeg_bytes)
        return {
            "status": "ok",
            "frame_count": frame_store.frame_count()
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/mirror/latest")
async def get_latest_frame():
    """
    Dashboard fetches the latest JPEG frame here every second.
    Returns 204 No Content if no frame has arrived yet.
    Returns 503 if the last frame is more than 5 seconds old (Unreal likely stopped).
    """
    frame = frame_store.get()

    if frame is None:
        return Response(status_code=204)  # no frame yet

    age = frame_store.age_seconds()
    if age > 5.0:
        # Frame is stale — Unreal may have stopped sending
        return Response(
            status_code=503,
            headers={"X-Frame-Age": str(round(age, 1))}
        )

    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={
            "X-Frame-Age":   str(round(age, 3)),
            "X-Frame-Count": str(frame_store.frame_count()),
        }
    )


@app.get("/mirror/status")
async def mirror_status():
    """Dashboard can poll this to show connection state."""
    return {
        "frames_received": frame_store.frame_count(),
        "last_frame_age_seconds": round(frame_store.age_seconds(), 2),
        "unreal_connected": len(connected_unreal_clients) > 0,
        "unreal_client_count": len(connected_unreal_clients),
        "stream_active": frame_store.age_seconds() < 5.0,
    }


# =============================================================================
# SECTION 5 — Unreal WebSocket connection
# =============================================================================

@app.websocket("/unreal/ws")
async def unreal_websocket(websocket: WebSocket):
    """
    Unreal Engine connects here on session start via the WebSockets plugin.
    The connection stays open for the duration of the session.
    Commands from the dashboard are pushed to Unreal through this connection.
    """
    await websocket.accept()
    connected_unreal_clients.append(websocket)
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[bridge] Unreal connected from {client_host}. "
          f"Total clients: {len(connected_unreal_clients)}")

    try:
        while True:
            # Keep the connection alive.
            # Unreal can also send messages here if needed (e.g. safe word fired).
            data = await websocket.receive_text()
            print(f"[bridge] Message from Unreal: {data}")

    except WebSocketDisconnect:
        connected_unreal_clients.remove(websocket)
        print(f"[bridge] Unreal disconnected. "
              f"Remaining clients: {len(connected_unreal_clients)}")


# =============================================================================
# SECTION 6 — Dashboard → Unreal command endpoint
# =============================================================================

@app.post("/unreal/command")
async def send_unreal_command(payload: CommandPayload):
    """
    Dashboard posts a command here.
    FastAPI forwards it to Unreal via WebSocket.

    Examples:

    Encouragement button clicked:
        POST /unreal/command
        {"type": "show_encouragement", "message": "Breathe — you're doing great."}

    Intensity lowered:
        POST /unreal/command
        {"type": "set_intensity", "band": "low"}

    Session stopped:
        POST /unreal/command
        {"type": "stop_session"}
    """
    if not connected_unreal_clients:
        return {
            "status": "warning",
            "detail": "No Unreal client connected — command not delivered",
            "command": payload.dict()
        }

    event = payload.dict(exclude_none=True)
    sent_to = await send_to_unreal(event)

    return {
        "status": "sent",
        "delivered_to": sent_to,
        "command": event
    }


# =============================================================================
# SECTION 7 — Health check
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "unreal_connected": len(connected_unreal_clients) > 0,
        "mirror_active": frame_store.age_seconds() < 5.0,
        "frames_received": frame_store.frame_count(),
    }