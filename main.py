import uuid
import time
import json
import httpx

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    Message,
    Usage,
)
from converter import build_llama_request

app = FastAPI(title="OpenAI to llama.cpp Bridge")


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": settings.default_model,
                "object": "model",
                "owned_by": "bridge",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    openai_req = ChatCompletionRequest(**body)
    llama_req = build_llama_request(openai_req)
    stream = llama_req.stream or settings.stream

    if stream:
        return StreamingResponse(
            stream_response(llama_req, openai_req.model),
            media_type="text/event-stream",
        )
    else:
        return await non_stream_response(llama_req, openai_req.model)


async def non_stream_response(llama_req, model_name: str):
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            settings.completions_url,
            json=llama_req.model_dump(exclude_none=True),
            headers={"Content-Type": "application/json"},
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=resp.text,
        )

    data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="No choices in llama.cpp response")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    finish_reason = choices[0].get("finish_reason", "stop")

    return ChatCompletionResponse(
        id=data.get("id", f"chatcmpl-{uuid.uuid4()}"),
        model=model_name,
        choices=[
            Choice(
                index=choices[0].get("index", 0),
                message=Message(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", prompt_tokens + completion_tokens),
        ),
        created=data.get("created", int(time.time())),
    )


async def stream_response(llama_req, model_name: str):
    llama_req.stream = True
    chunk_id = f"chatcmpl-{uuid.uuid4()}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            settings.completions_url,
            json=llama_req.model_dump(exclude_none=True),
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status_code != 200:
                error_text = await resp.aread()
                yield f"data: {json.dumps({'error': error_text.decode()})}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]

                try:
                    chunk_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                choices = chunk_data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    chunk = ChatCompletionChunk(
                        id=chunk_id,
                        model=model_name,
                        created=int(time.time()),
                        choices=[
                            Choice(
                                index=0,
                                delta=ChoiceDelta(role="assistant", content=content),
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

    done_chunk = ChatCompletionChunk(
        id=chunk_id,
        model=model_name,
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {done_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
