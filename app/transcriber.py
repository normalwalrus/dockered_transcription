"""Single shared faster-whisper model guarded by a lock."""
import logging
import os
import threading
import time

import ctranslate2
from faster_whisper import WhisperModel

from . import config

log = logging.getLogger("transcriber")


def _resolve_device() -> str:
    if config.DEVICE in ("cuda", "cpu"):
        return config.DEVICE
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception as exc:  # noqa: BLE001
        log.warning("CUDA probe failed (%s); falling back to CPU", exc)
    return "cpu"


def _resolve_compute_type(device: str) -> str:
    if config.COMPUTE_TYPE != "auto":
        return config.COMPUTE_TYPE
    return "float16" if device == "cuda" else "int8"


class Transcriber:
    def __init__(self) -> None:
        self.model: WhisperModel | None = None
        self.device = "unloaded"
        self.compute_type = "unloaded"
        self._lock = threading.Lock()

    def load(self) -> None:
        if not os.path.isdir(config.MODEL_PATH) or not os.path.exists(
            os.path.join(config.MODEL_PATH, "model.bin")
        ):
            raise RuntimeError(
                f"No CTranslate2 model found at {config.MODEL_PATH!r}. "
                "Expected a folder containing model.bin, config.json, tokenizer.json, "
                "preprocessor_config.json, vocabulary.json. Run ./download_model.sh "
                "or set MODEL_PATH."
            )
        self.device = _resolve_device()
        self.compute_type = _resolve_compute_type(self.device)
        log.info(
            "Loading model from %s on %s (%s)", config.MODEL_PATH, self.device, self.compute_type
        )
        t0 = time.perf_counter()
        self.model = WhisperModel(
            config.MODEL_PATH,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=config.CPU_THREADS,
            local_files_only=True,
        )
        log.info("Model loaded in %.1fs", time.perf_counter() - t0)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def transcribe(
        self,
        audio_path: str,
        language: str | None,
        word_timestamps: bool,
    ) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Blocking call; one transcription at a time so VRAM stays bounded.
        with self._lock:
            t0 = time.perf_counter()
            segments_iter, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=config.BEAM_SIZE,
                word_timestamps=word_timestamps,
            )
            segments = []
            for seg in segments_iter:  # generator; decoding happens here
                item = {
                    "id": seg.id,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                }
                if word_timestamps and seg.words:
                    item["words"] = [
                        {
                            "word": w.word,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(w.probability, 4),
                        }
                        for w in seg.words
                    ]
                segments.append(item)
            elapsed = time.perf_counter() - t0

        return {
            "text": " ".join(s["text"] for s in segments).strip(),
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 3),
            "processing_time": round(elapsed, 3),
            "segments": segments,
        }


transcriber = Transcriber()
