import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from . import config
from .transcriber import transcriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    transcriber.load()
    yield


app = FastAPI(title="whisper-turbo", version="1.0.0", lifespan=lifespan)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if config.API_KEY is None:
        return
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if transcriber.ready else "loading",
        "model_path": config.MODEL_PATH,
        "device": transcriber.device,
        "compute_type": transcriber.compute_type,
        "auth_enabled": config.API_KEY is not None,
    }


@app.post("/transcribe", dependencies=[Depends(require_api_key)])
async def transcribe(
    file: UploadFile = File(..., description="Audio/video file; anything ffmpeg can decode"),
    language: str | None = Form(default=None, description="ISO 639-1 code, e.g. en. Auto-detect if omitted"),
    word_timestamps: bool = Form(default=False),
) -> dict:
    if not transcriber.ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    language = language.strip().lower() if language else None
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        written = 0
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status_code=413, detail=f"File exceeds MAX_UPLOAD_MB={config.MAX_UPLOAD_MB}"
                )
            tmp.write(chunk)
        tmp.close()
        if written == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            result = await run_in_threadpool(
                transcriber.transcribe, tmp.name, language, word_timestamps
            )
        except ValueError as exc:  # bad language code etc.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — decode failures surface here
            log.exception("Transcription failed for %s", file.filename)
            raise HTTPException(status_code=422, detail=f"Could not transcribe: {exc}") from exc

        result["filename"] = file.filename
        return result
    finally:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
