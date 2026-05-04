from typing import Literal, Optional, List, Any, Dict

from pydantic import BaseModel


# ── Response models (kept for reference / future use) ──────────

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class Choice(BaseModel):
    index: int = 0
    message: Message = Message(role="assistant", content="")
    delta: ChoiceDelta = ChoiceDelta()
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str = "local"
    choices: List[Choice]
    usage: Optional[Usage] = None
    created: Optional[int] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    model: str = "local"
    choices: List[Choice]
    created: Optional[int] = None
