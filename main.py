import uuid
import time
import json
import httpx
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

from config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("bridge")

app = FastAPI(title="OpenAI → llama.cpp Bridge")


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": settings.default_model, "object": "model", "owned_by": "bridge"}
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON")

    messages = body.get("messages", [])
    model_name = body.get("model", settings.default_model)

    # ── log incoming request ──────────────────────────────────
    logger.info("=" * 70)
    logger.info("📥 CLIENT → BRIDGE")
    logger.info(f"   model    : {model_name}")
    logger.info(f"   messages : {len(messages)}")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        text = msg.get("content", "")
        logger.info(f"   msg[{i}] : role={role}, len={len(text)}")
        preview = text[:300] if len(text) > 300 else text
        logger.info(f"            preview: {repr(preview)}")
    logger.info("=" * 70)

    # ── build llama.cpp payload (pass messages through verbatim) ─
    llama_body: dict = {"messages": messages}

    for src, dst in [
        ("temperature", "temperature"),
        ("max_tokens", "n_predict"),
        ("top_p", "top_p"),
        ("stop", "stop"),
        ("frequency_penalty", "frequency_penalty"),
        ("presence_penalty", "presence_penalty"),
    ]:
        if src in body:
            llama_body[dst] = body[src]

    llama_body.setdefault("temperature", settings.temperature)
    llama_body.setdefault("n_predict", settings.max_tokens)
    llama_body.setdefault("top_p", settings.top_p)
    llama_body.setdefault("stream", settings.stream)

    # ── log outgoing request ──────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"📤 BRIDGE → LLAMA.CPP ({settings.completions_url})")
    logger.info(f"   stream: {llama_body['stream']}")
    logger.info(f"   payload:\n{json.dumps(llama_body, ensure_ascii=False, indent=2)}")
    logger.info("=" * 70)

    if llama_body["stream"]:
        return StreamingResponse(
            _stream(llama_body, model_name),
            media_type="text/event-stream",
        )
    return await _non_stream(llama_body, model_name)


# ── non-streaming ──────────────────────────────────────────────
async def _non_stream(llama_body: dict, model_name: str):
    async with httpx.AsyncClient(timeout=240.0) as client:
        resp = await client.post(
            settings.completions_url,
            json=llama_body,
            headers={"Content-Type": "application/json"},
        )

    if resp.status_code != 200:
        logger.error(f"llama.cpp HTTP {resp.status_code}: {resp.text[:1000]}")
        raise HTTPException(resp.status_code, resp.text)

    data = resp.json()

    # ── log llama.cpp raw response ────────────────────────────
    logger.info("=" * 70)
    logger.info("📥 LLAMA.CPP → BRIDGE")
    raw_out = json.dumps(data, ensure_ascii=False)
    if len(raw_out) > 2000:
        logger.info(f"   raw ({len(raw_out)} chars): {raw_out[:2000]}...")
    else:
        logger.info(f"   raw: {raw_out}")
    logger.info("=" * 70)

    choices = data.get("choices", [])
    if not choices:
        raise HTTPException(502, "No choices in llama.cpp response")

    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    usage = data.get("usage", {})

    response = {
        "id": data.get("id", f"chatcmpl-{uuid.uuid4()}"),
        "object": "chat.completion",
        "model": model_name,
        "choices": [{
            "index": choices[0].get("index", 0),
            "message": {"role": msg.get("role", "assistant"), "content": content},
            "finish_reason": choices[0].get("finish_reason", "stop"),
        }],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "created": data.get("created", int(time.time())),
    }

    # ── log outgoing response ─────────────────────────────────
    logger.info("=" * 70)
    logger.info("📤 BRIDGE → CLIENT")
    logger.info(f"   content len : {len(content)}")
    preview = content[:300] if len(content) > 300 else content
    logger.info(f"   preview     : {repr(preview)}")
    logger.info("=" * 70)

    return response


# ── streaming ──────────────────────────────────────────────────
async def _stream(llama_body: dict, model_name: str):
    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    llama_body["stream"] = True
    full = ""

    async with httpx.AsyncClient(timeout=240.0) as client:
        async with client.stream(
            "POST", settings.completions_url, json=llama_body,
        ) as resp:
            if resp.status_code != 200:
                err = (await resp.aread()).decode()
                logger.error(f"stream HTTP {resp.status_code}: {err[:1000]}")
                yield f"data: {json.dumps({'error': err})}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                deltas = chunk.get("choices", [{}])[0].get("delta", {})
                text = deltas.get("content", "")
                full += text

                if text:
                    chunk_json = json.dumps({
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "created": int(time.time()),
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": text},
                        }],
                    })
                    yield "data: " + chunk_json + "\n\n"

    done_json = json.dumps({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model_name,
        "created": int(time.time()),
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    yield "data: " + done_json + "\n\n"
    yield "data: [DONE]\n\n"

    logger.info("[stream] done - accumulated " + str(len(full)) + " chars")
