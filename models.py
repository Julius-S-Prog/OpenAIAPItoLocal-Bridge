from typing import Literal, Optional, List, Any, Dict

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


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


class LlamaCppRequest(BaseModel):
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.7
    n_predict: Optional[int] = 2048
    top_p: Optional[float] = 0.95
    stream: bool = False
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
