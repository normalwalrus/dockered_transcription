import os


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


MODEL_PATH: str = os.getenv("MODEL_PATH", "/models/whisper-large-v3-turbo-ct2")
DEVICE: str = os.getenv("DEVICE", "auto")              # auto | cuda | cpu
COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "auto")  # auto | float16 | int8 | int8_float16 | float32
CPU_THREADS: int = int(os.getenv("CPU_THREADS", "0"))  # 0 = let ctranslate2 decide
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "200"))
API_KEY: str | None = os.getenv("API_KEY") or None
BEAM_SIZE: int = int(os.getenv("BEAM_SIZE", "5"))
