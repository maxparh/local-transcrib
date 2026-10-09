"""Desktop/core settings. Override defaults with TRANSCRIBER_* environment variables."""

import os


def _env(name, default):
    return os.environ.get(f"TRANSCRIBER_{name}", default)


def _integer(name, default, minimum=0):
    value = int(_env(name, str(default)))
    if value < minimum:
        raise ValueError(f"TRANSCRIBER_{name} must be >= {minimum}")
    return value


def _boolean(name, default):
    value = _env(name, str(default)).lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"TRANSCRIBER_{name} must be true/false or 1/0")
    return value in {"true", "1"}


MODEL_NAME = _env("MODEL_NAME", "large-v3")
MODEL_SOURCE = _env("MODEL_SOURCE", MODEL_NAME)  # Identifier or local directory.
DEVICE = _env("DEVICE", "auto")
if DEVICE not in {"auto", "cuda", "cpu"}:
    raise ValueError("TRANSCRIBER_DEVICE must be auto, cuda or cpu")
COMPUTE_TYPE = _env("COMPUTE_TYPE", "int8")
CPU_COMPUTE_TYPE = _env("CPU_COMPUTE_TYPE", "int8")
GPU_COMPUTE_TYPE = _env("GPU_COMPUTE_TYPE", _env("COMPUTE_TYPE", "float16"))
CPU_THREADS = _integer("CPU_THREADS", 8, 1)
WORKERS = _integer("WORKERS", 1, 1)
BATCHED_TRANSCRIPTION = _boolean("BATCHED_TRANSCRIPTION", True)
GPU_BATCH_SIZE = _integer("GPU_BATCH_SIZE", 4, 1)
CPU_BATCH_SIZE = _integer("CPU_BATCH_SIZE", 1, 1)
CHUNK_LENGTH = _integer("CHUNK_LENGTH", 15, 1)
if CHUNK_LENGTH > 30:
    raise ValueError("TRANSCRIBER_CHUNK_LENGTH must be <= 30")

TRANSCRIPTION_OPTIONS = {
    "language": _env("LANGUAGE", "ru") or None,
    "beam_size": _integer("BEAM_SIZE", 5, 1),
    "vad_filter": _boolean("VAD_FILTER", True),
    "vad_parameters": {
        "threshold": float(_env("VAD_THRESHOLD", "0.35")),
        "min_silence_duration_ms": _integer("VAD_MIN_SILENCE_MS", 300),
        "speech_pad_ms": _integer("VAD_SPEECH_PAD_MS", 300),
    },
    "condition_on_previous_text": _boolean("CONDITION_ON_PREVIOUS_TEXT", False),
    "compression_ratio_threshold": float(_env("COMPRESSION_RATIO_THRESHOLD", "2.4")),
    "temperature": float(_env("TEMPERATURE", "0.0")),
    "best_of": _integer("BEST_OF", 1, 1),
    "hotwords": _env("HOTWORDS", "") or None,
}

TEMP_DIR = _env("TEMP_DIR", "") or None  # None uses the system temporary directory.
