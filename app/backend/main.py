from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.backend.adapters import list_adapters
from app.backend.agent import run_agent
from app.backend.prompts import SYSTEM_PROMPT

app = FastAPI(title="Agent-Tuning Playground")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    adapter_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    max_steps: int = 6


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/adapters")
def adapters():
    return {"adapters": list_adapters()}


@app.get("/api/system-prompt")
def system_prompt():
    return {"system_prompt": SYSTEM_PROMPT}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.adapter_id:
        raise HTTPException(400, "adapter_id required")
    user_history = [m.model_dump(exclude_none=True) for m in req.messages]
    try:
        result = run_agent(
            req.adapter_id,
            user_history,
            max_steps=req.max_steps,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}") from exc
    return result
