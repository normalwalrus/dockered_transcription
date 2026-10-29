"""Gradio frontend for the whisper-turbo FastAPI backend."""
import os
import tempfile
from pathlib import Path

import gradio as gr
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
DEFAULT_API_KEY = os.getenv("API_KEY", "")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))
EXAMPLE_DIR = Path(os.getenv("EXAMPLE_DIR", "/assets"))

LANGUAGES = [
    ("Auto-detect", ""),
    ("English (en)", "en"), ("Chinese (zh)", "zh"), ("Malay (ms)", "ms"),
    ("Tamil (ta)", "ta"), ("Indonesian (id)", "id"), ("Japanese (ja)", "ja"),
    ("Korean (ko)", "ko"), ("Hindi (hi)", "hi"), ("Thai (th)", "th"),
    ("Vietnamese (vi)", "vi"), ("French (fr)", "fr"), ("German (de)", "de"),
    ("Spanish (es)", "es"), ("Portuguese (pt)", "pt"), ("Italian (it)", "it"),
    ("Russian (ru)", "ru"), ("Arabic (ar)", "ar"),
]


def _fmt_srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}", seg["text"], ""]
    return "\n".join(lines)


def backend_status() -> str:
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=5)
        r.raise_for_status()
        h = r.json()
        icon = "🟢" if h.get("status") == "ok" else "🟡"
        auth = " · auth on" if h.get("auth_enabled") else ""
        return f"{icon} Backend `{h.get('status')}` · {h.get('device')} / {h.get('compute_type')}{auth}"
    except Exception as exc:  # noqa: BLE001
        return f"🔴 Backend unreachable at `{BACKEND_URL}`: {exc}"


def transcribe(file_path: str | None, language: str, word_timestamps: bool, api_key: str):
    if not file_path:
        raise gr.Error("Upload an audio or video file first.")

    headers = {"X-API-Key": api_key} if api_key else {}
    data = {"word_timestamps": "true" if word_timestamps else "false"}
    if language:
        data["language"] = language

    try:
        with open(file_path, "rb") as fh:
            r = httpx.post(
                f"{BACKEND_URL}/transcribe",
                headers=headers,
                data=data,
                files={"file": (os.path.basename(file_path), fh)},
                timeout=REQUEST_TIMEOUT,
            )
    except httpx.HTTPError as exc:
        raise gr.Error(f"Could not reach backend: {exc}") from exc

    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise gr.Error(f"Backend returned {r.status_code}: {detail}")

    res = r.json()
    segments = res.get("segments", [])

    info = (
        f"**Language:** {res.get('language')} ({res.get('language_probability', 0):.0%}) · "
        f"**Audio:** {res.get('duration', 0):.1f}s · "
        f"**Processed in:** {res.get('processing_time', 0):.2f}s"
    )
    seg_rows = [[s["id"], f"{s['start']:.2f}", f"{s['end']:.2f}", s["text"]] for s in segments]
    word_rows = [
        [f"{w['start']:.2f}", f"{w['end']:.2f}", w["word"].strip(), f"{w['probability']:.2f}"]
        for s in segments for w in s.get("words", [])
    ]

    stem = Path(file_path).stem or "transcript"
    out_dir = Path(tempfile.mkdtemp(prefix="whisper_"))
    txt_path = out_dir / f"{stem}.txt"
    srt_path = out_dir / f"{stem}.srt"
    txt_path.write_text(res.get("text", ""), encoding="utf-8")
    srt_path.write_text(_to_srt(segments), encoding="utf-8")

    return (
        res.get("text", ""),
        info,
        seg_rows,
        gr.Dataframe(value=word_rows, visible=bool(word_rows)),
        gr.DownloadButton(value=str(txt_path), visible=True),
        gr.DownloadButton(value=str(srt_path), visible=True),
    )


with gr.Blocks(title="Whisper Turbo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Whisper Turbo transcription")
    status = gr.Markdown(backend_status)

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.File(
                label="Audio / video file",
                file_types=["audio", "video"],
                type="filepath",
            )
            language = gr.Dropdown(LANGUAGES, value="", label="Language")
            word_ts = gr.Checkbox(False, label="Word timestamps")
            api_key = gr.Textbox(
                DEFAULT_API_KEY, label="API key (only if backend requires it)", type="password"
            )
            run_btn = gr.Button("Transcribe", variant="primary")

            example_file = EXAMPLE_DIR / "test.wav"
            if example_file.exists():
                gr.Examples(
                    examples=[[str(example_file)]], inputs=[audio_in], label="Example"
                )

        with gr.Column(scale=2):
            info_md = gr.Markdown()
            text_out = gr.Textbox(label="Transcript", lines=8, show_copy_button=True)
            with gr.Row():
                dl_txt = gr.DownloadButton("Download .txt", visible=False)
                dl_srt = gr.DownloadButton("Download .srt", visible=False)
            seg_table = gr.Dataframe(
                headers=["#", "Start", "End", "Text"],
                datatype=["number", "str", "str", "str"],
                label="Segments",
                wrap=True,
                interactive=False,
            )
            word_table = gr.Dataframe(
                headers=["Start", "End", "Word", "Prob"],
                datatype=["str", "str", "str", "str"],
                label="Words",
                visible=False,
                interactive=False,
            )

    run_btn.click(
        transcribe,
        inputs=[audio_in, language, word_ts, api_key],
        outputs=[text_out, info_md, seg_table, word_table, dl_txt, dl_srt],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        allowed_paths=[str(EXAMPLE_DIR)],
    )
