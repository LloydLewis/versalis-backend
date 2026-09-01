from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import sys
import os
import httpx
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'llmsegment'))

from aisegment import ask_question_async

app = FastAPI(title="Versalis LLM Service")


# ── Request / Response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

def make_openai_response(reply: str, was_flagged: bool) -> dict:
    return {
        "id": f"versalis-{int(time.time())}",
        "object": "chat.completion",
        "model": "versalis-llm",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        },
        "was_flagged": was_flagged
    }


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat/completions")
@app.post("/v1/chat/completions")  # alias — some clients (e.g. NVIDIA ACE's
                                    # BP_ASR_Debug) default to the OpenAI-standard
                                    # /v1/ prefixed path
async def chat_completions(request: OpenAIRequest):
    # Extract the last user message
    patient_text = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            patient_text = msg.content
            break

    if not patient_text.strip():
        raise HTTPException(status_code=400, detail="No user message found")

    start_time = time.time()
    reply = await ask_question_async(patient_text)
    latency_ms = int((time.time() - start_time) * 1000)

    was_flagged = reply in [
        "I'm right here with you, and I'm getting your care team connected now.",
        "Let's pause for a moment — I'm here whenever you're ready."
    ]

    # Forward interaction to bridge for dashboard display and Realm storage
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(
                "http://127.0.0.1:8002/llm/interaction",
                json={
                    "sessionId":       "session-001",
                    "patientText":     patient_text,
                    "therapistReply":  reply,
                    "wasGuardFlagged": was_flagged,
                    "latencyMs":       latency_ms,
                    "modelUsed":       "llama3.2:1b"
                }
            )
    except Exception as e:
        print(f"[LLM] Failed to forward to bridge: {e}")

    return make_openai_response(reply, was_flagged)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}