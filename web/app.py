from __future__ import annotations

import asyncio
import base64
import io
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from dotenv import load_dotenv

from .service import WebPipelineService, WebServiceError, default_service

WEB_DIR = Path(__file__).resolve().parent
load_dotenv(WEB_DIR.parent / ".env")


class RunRequest(BaseModel):
    source_dir: str
    output_dir: str | None = None
    use_local_llm: bool = False
    use_llm: bool = False
    strict_order_ids: bool = False
    strict: bool = False


class ApproveRequest(BaseModel):
    names: dict[str, str] = Field(default_factory=dict)


def create_app(service: WebPipelineService | None = None) -> FastAPI:
    app = FastAPI(title="UnpackModel Control", version="0.1.0")
    app.state.service = service or default_service()
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        password = os.getenv("UNPACK_WEB_PASSWORD") or os.getenv("ADMIN_PASSWORD")
        username = os.getenv("UNPACK_WEB_USERNAME") or os.getenv("LOGIN_USER") or "admin"
        if password and request.url.path != "/api/health":
            supplied_user = supplied_password = ""
            authorization = request.headers.get("authorization", "")
            if authorization.startswith("Basic "):
                try:
                    decoded = base64.b64decode(authorization[6:]).decode("utf-8")
                    supplied_user, supplied_password = decoded.split(":", 1)
                except (ValueError, UnicodeDecodeError):
                    pass
            if not (
                secrets.compare_digest(supplied_user, username)
                and secrets.compare_digest(supplied_password, password)
            ):
                return JSONResponse(
                    {"detail": "Требуется вход"},
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="UnpackModel"'},
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    def call(method, *args):
        try:
            return method(*args)
        except (WebServiceError, FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB_DIR / "templates" / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/config")
    def config():
        return {
            "source_dir": os.getenv("UNPACK_SOURCE_DIR", ""),
            "output_dir": os.getenv("UNPACK_OUTPUT_DIR", ""),
        }

    @app.get("/api/runs")
    def runs():
        return {"items": app.state.service.list_runs()}

    @app.get("/api/runs/current")
    def current():
        items = app.state.service.list_runs()
        return items[0] if items else None

    @app.post("/api/runs", status_code=202)
    def start(payload: RunRequest):
        return call(app.state.service.start, payload.model_dump())

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return call(app.state.service.public_run, call(app.state.service.get, run_id))

    @app.get("/api/runs/{run_id}/orders")
    def orders(run_id: str):
        return {"items": call(app.state.service.orders, run_id)}

    @app.get("/api/runs/{run_id}/orders/{order_id}")
    def order(run_id: str, order_id: str):
        _run, value = call(app.state.service.find_order, run_id, order_id)
        return app.state.service.public_order(run_id, value)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        return call(app.state.service.cancel, run_id)

    @app.post("/api/runs/{run_id}/orders/{order_id}/approve")
    def approve(run_id: str, order_id: str, payload: ApproveRequest):
        return call(app.state.service.approve, run_id, order_id, payload.names)

    @app.post("/api/runs/{run_id}/orders/{order_id}/postpone")
    def postpone(run_id: str, order_id: str):
        return call(app.state.service.postpone, run_id, order_id)

    @app.post("/api/runs/{run_id}/orders/{order_id}/manual-review")
    def manual(run_id: str, order_id: str):
        return call(app.state.service.manual_review, run_id, order_id)

    @app.get("/api/runs/{run_id}/files/{file_id}/source")
    def source(run_id: str, file_id: str):
        return FileResponse(call(app.state.service.find_file, run_id, file_id))

    @app.get("/api/runs/{run_id}/files/{file_id}/preview")
    def preview(run_id: str, file_id: str):
        path = call(app.state.service.find_file, run_id, file_id)
        if path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf"}:
            return FileResponse(path)
        if path.suffix.casefold() not in {".tif", ".tiff"}:
            raise HTTPException(415, "Предпросмотр этого формата недоступен")
        try:
            with Image.open(path) as image:
                image.thumbnail((1400, 1400))
                converted = image.convert("RGB")
                data = io.BytesIO()
                converted.save(data, "JPEG", quality=82)
            return StreamingResponse(io.BytesIO(data.getvalue()), media_type="image/jpeg")
        except Exception as exc:
            raise HTTPException(415, "Не удалось создать превью") from exc

    @app.get("/api/runs/{run_id}/events")
    async def events(request: Request, run_id: str):
        call(app.state.service.get, run_id)
        async def stream():
            cursor = 0
            while not await request.is_disconnected():
                run = app.state.service.get(run_id)
                for event in run["events"]:
                    if event["id"] > cursor:
                        cursor = event["id"]
                        import json
                        yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                if run["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                    break
                yield ": keep-alive\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("UNPACK_WEB_HOST", "127.0.0.1")
    password = os.getenv("UNPACK_WEB_PASSWORD") or os.getenv("ADMIN_PASSWORD")
    if host not in {"127.0.0.1", "localhost", "::1"} and not password:
        raise SystemExit("Для сетевого запуска задайте UNPACK_WEB_PASSWORD или ADMIN_PASSWORD")
    uvicorn.run(app, host=host, port=int(os.getenv("UNPACK_WEB_PORT", "8016")))
