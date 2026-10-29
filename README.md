# whisper-turbo API + Gradio UI

FastAPI service that serves `whisper-large-v3-turbo` via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2),
plus a small Gradio frontend. Runs on an NVIDIA GPU when one is visible and falls back to CPU otherwise.

## Layout

```
app/            FastAPI backend (main.py, transcriber.py, config.py)
frontend/       Gradio UI (app.py, Dockerfile, requirements.txt)
assets/         Sample audio for testing
models/         Put model weights here (bind-mounted to /models)
Dockerfile
docker-compose.yml
.env            Runtime config
download_model.sh
```

## 1. Get the weights

```bash
./download_model.sh
```

This fetches `deepdml/faster-whisper-large-v3-turbo-ct2` into `models/whisper-large-v3-turbo-ct2/`.
Any CTranslate2 Whisper folder works. It must contain `model.bin`, `config.json`, `tokenizer.json`,
`preprocessor_config.json` and `vocabulary.json`. Point `MODEL_PATH` in `.env` at it.

## 2. Build and run

```bash
docker compose up --build -d
docker compose logs -f
```

This starts two containers: `whisper-turbo` (API on port 8000) and `whisper-frontend` (UI on port 7860).
The frontend waits for the backend health check before starting. First start loads the model and takes ~10-30 s.

No GPU? Delete the `deploy:` block in `docker-compose.yml` or set `DEVICE=cpu` in `.env`.

## 3. Use the UI

Open http://localhost:7860

- Drop in an audio or video file, or click the `test.wav` example.
- Pick a language or leave it on auto-detect. Tick **Word timestamps** for per-word timings.
- The API key field is only needed when the backend has `API_KEY` set. The frontend pre-fills it from `.env`.
- Results: transcript textbox, detected language and timings, a segment table, a word table when enabled,
  and **Download .txt** / **Download .srt** buttons.

The frontend talks to the backend over the compose network at `http://whisper:8000`. To point it elsewhere,
set `BACKEND_URL` on the `frontend` service.

## 4. Use the API directly

Health:

```bash
curl http://localhost:8000/health
```

Transcribe (auto-detect language):

```bash
curl -F "file=@assets/test.wav" http://localhost:8000/transcribe
```

With a fixed language and per-word timestamps:

```bash
curl -F "file=@assets/test.wav" -F "language=en" -F "word_timestamps=true" \
     http://localhost:8000/transcribe
```

With auth enabled (`API_KEY=secret` in `.env`):

```bash
curl -H "X-API-Key: secret" -F "file=@assets/test.wav" http://localhost:8000/transcribe
```

Response:

```json
{
  "text": "The detective tracked down the cat killer quickly.",
  "language": "en",
  "language_probability": 0.99,
  "duration": 4.252,
  "processing_time": 0.45,
  "segments": [
    {"id": 1, "start": 0.0, "end": 1.8, "text": "The detective tracked down the cat killer quickly.",
     "words": [{"word": " The", "start": 0.0, "end": 0.7, "probability": 0.95}, ...]}
  ],
  "filename": "test.wav"
}
```

Interactive docs: http://localhost:8000/docs

## Config (`.env`)

| Var | Default | Meaning |
|---|---|---|
| `PORT` | 8000 | Host port for the API |
| `FRONTEND_PORT` | 7860 | Host port for the Gradio UI |
| `MODEL_PATH` | /models/whisper-large-v3-turbo-ct2 | CT2 model folder inside the container |
| `DEVICE` | auto | auto / cuda / cpu |
| `COMPUTE_TYPE` | auto | float16 on GPU, int8 on CPU; or set explicitly |
| `CPU_THREADS` | 0 | CPU decode threads (0 = auto) |
| `MAX_UPLOAD_MB` | 200 | Upload size limit |
| `BEAM_SIZE` | 5 | 1 for fastest greedy decoding |
| `API_KEY` | empty | If set, requests need `X-API-Key`. Also passed to the frontend |

## Notes

- Requests are processed one at a time behind a lock so GPU memory stays bounded. Concurrent calls queue.
- Input is anything ffmpeg can decode (wav, mp3, m4a, flac, ogg, webm, mp4, ...).
- Errors: 400 empty file, bad language code or undecodable audio; 401 bad key; 413 too large; 422 missing file field; 503 model still loading.
- `assets/test.wav` is a 4 s English sample you can use to smoke-test the service.
