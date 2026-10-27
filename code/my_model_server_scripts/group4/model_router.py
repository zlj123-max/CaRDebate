import json
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

app = FastAPI()

LOCAL_MODELS = json.loads(
    os.environ["LOCAL_MODEL_ROUTES_JSON"]
)

REMOTE_BASE_URL = os.environ["REMOTE_OPENAI_BASE_URL"].rstrip("/")
REMOTE_API_KEY = os.environ["REMOTE_OPENAI_API_KEY"]


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model"}
            for name in LOCAL_MODELS
        ],
    }


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy(path: str, request: Request):
    body = await request.body()

    payload = {}
    if body:
        try:
            payload = json.loads(body)
        except Exception:
            pass

    model = payload.get("model")

    if model in LOCAL_MODELS:
        base_url = LOCAL_MODELS[model]
        api_key = "local"
        destination = base_url
    else:
        base_url = REMOTE_BASE_URL
        api_key = REMOTE_API_KEY
        destination = "REMOTE"

    print(
        f"[ROUTER] model={model} path=/{path} -> {destination}",
        flush=True,
    )

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower()
        not in {"host", "content-length", "authorization"}
    }

    headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url}/{path}"

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=30,
            read=None,
            write=60,
            pool=60,
        )
    )

    upstream_request = client.build_request(
        method=request.method,
        url=url,
        headers=headers,
        content=body,
    )

    upstream = await client.send(
        upstream_request,
        stream=True,
    )

    if payload.get("stream") is True:

        async def generate():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            generate(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get(
                "content-type",
                "text/event-stream",
            ).split(";")[0],
        )

    content = await upstream.aread()

    status_code = upstream.status_code

    content_type = upstream.headers.get(
        "content-type",
        "application/json",
    ).split(";")[0]

    await upstream.aclose()
    await client.aclose()

    return Response(
        content=content,
        status_code=status_code,
        media_type=content_type,
    )
