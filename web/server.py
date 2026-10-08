"""Tiny web server for the browser demo: serves index.html and hands out LiveKit tokens.

    uv run uvicorn web.server:app --port 8000
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from livekit import api

load_dotenv(".env")

app = FastAPI()
INDEX = Path(__file__).parent / "index.html"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX)


@app.get("/token")
def token() -> dict[str, str]:
    # a fresh room per visit; the agent auto-joins every new room
    room = f"web-{uuid.uuid4().hex[:8]}"
    jwt = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(f"user-{uuid.uuid4().hex[:6]}")
        .with_name("Web user")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return {"url": os.environ["LIVEKIT_URL"], "token": jwt, "room": room}
