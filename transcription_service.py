"""Synchronous core. Import/construction never loads or downloads a model."""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import gc
import logging
import os
from threading import RLock
import time

import av

import config
from preprocessing import MediaError, prepare_media

logger = logging.getLogger(__name__)


class DeviceError(RuntimeError):
    """Requested device or compute type is unavailable to CTranslate2."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    duration_sec: float
    language: str
    segments_count: int
    audio_duration_sec: float


# Keep DLL-directory handles alive for the entire process lifetime.
_DLL_DIRECTORIES = []


@lru_cache(maxsize=1)
def _prepare_runtime():
    """Register installed libraries without importing Torch or renaming DLLs."""
    # Initialize VAD's native extension before CTranslate2/Torch and model
    # allocation. Late ONNX import from VAD caused a native access violation
    # in the reported desktop run. This still runs in the background worker.
    if config.TRANSCRIPTION_OPTIONS.get("vad_filter"):
        import onnxruntime
        logger.info("VAD runtime initialized: ONNX Runtime %s", onnxruntime.__version__)
    if os.name != "nt":
        return
    from importlib.metadata import distribution, PackageNotFoundError

    directories = []
    for package, relative in (
        ("nvidia-cublas-cu12", "nvidia/cublas/bin"),
        ("nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime/bin"),
        ("nvidia-cuda-nvrtc-cu12", "nvidia/cuda_nvrtc/bin"),
        ("ctranslate2", "ctranslate2"),
        ("torch", "torch/lib"),  # Existing cuDNN component DLLs, no torch import.
    ):
        try:
            directory = distribution(package).locate_file(relative).resolve()
        except PackageNotFoundError:
            continue
        if directory.is_dir():
            directories.append(str(directory))
            _DLL_DIRECTORIES.append(os.add_dll_directory(str(directory)))
    # CTranslate2 also loads libraries dynamically by name via the native loader.
    os.environ["PATH"] = os.pathsep.join(directories + [os.environ.get("PATH", "")])


def _resolve_device(requested, cpu_compute_type, gpu_compute_type):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    _prepare_runtime()
    import ctranslate2

    device = requested
    if requested in {"auto", "cuda"}:
        available = ctranslate2.get_cuda_device_count() > 0
        if requested == "cuda" and not available:
            raise DeviceError("CTranslate2 не обнаружил доступное CUDA-устройство")
        device = "cuda" if available else "cpu"
    compute_type = gpu_compute_type if device == "cuda" else cpu_compute_type
    if compute_type not in ctranslate2.get_supported_compute_types(device):
        raise DeviceError(f"CTranslate2 не поддерживает compute_type={compute_type} на {device}")
    return device, compute_type


def _create_model(source, **kwargs):
    if os.path.isabs(source) and not os.path.isdir(source):
        raise FileNotFoundError("Модель Whisper не установлена: " + source)
    from faster_whisper import WhisperModel
    # Prefer an already cached model: switching devices should not make an
    # HTTP request. Keep first-run development downloads available if missing.
    if not os.path.isdir(source):
        from faster_whisper.utils import download_model
        from huggingface_hub.errors import LocalEntryNotFoundError
        try:
            source = download_model(source, local_files_only=True)
        except LocalEntryNotFoundError:
            pass
    return WhisperModel(source, **kwargs)


def _create_batch_pipeline(model):
    from faster_whisper import BatchedInferencePipeline
    return BatchedInferencePipeline(model=model)


class TranscriptionService:
    def __init__(self, model_source=None, device=None):
        self.model_source = os.fspath(config.MODEL_SOURCE if model_source is None else model_source)
        self._requested_device = config.DEVICE if device is None else device
        if self._requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        self._cpu_compute_type = config.CPU_COMPUTE_TYPE
        self._gpu_compute_type = config.GPU_COMPUTE_TYPE
        self._options = deepcopy(config.TRANSCRIPTION_OPTIONS)
        self._lock = RLock()
        self._model = None
        self._device = None
        self._compute_type = None

    @property
    def device(self):
        """Actual loaded device, or None when unloaded."""
        with self._lock:
            return self._device

    @property
    def is_loaded(self):
        with self._lock:
            return self._model is not None

    def cuda_available(self):
        """Probe configured CUDA support without loading a Whisper model.

        This does not guarantee sufficient GPU memory or usable inference DLLs.
        CTranslate2 may import Torch internally through its optional converters.
        """
        try:
            _resolve_device("cuda", self._cpu_compute_type, self._gpu_compute_type)
            return True
        except Exception:
            logger.warning("CUDA capability check failed; CPU remains available", exc_info=True)
            return False

    def load_model(self, device=None):
        with self._lock:
            requested = self._requested_device if device is None else device
            if requested not in {"auto", "cpu", "cuda"}:
                raise ValueError("device must be auto, cpu or cuda")
            selected, compute_type = _resolve_device(requested, self._cpu_compute_type, self._gpu_compute_type)
            if self._model is not None and (selected, compute_type) == (self._device, self._compute_type):
                self._requested_device = requested
                return
            # Drop the old instance before allocating another large model.
            # Failed creation leaves the service unloaded, ready for a retry.
            self.unload_model()
            logger.info("Loading faster-whisper %s (%s, %s)", self.model_source, selected, compute_type)
            self._model = _create_model(
                self.model_source, device=selected, compute_type=compute_type,
                cpu_threads=config.CPU_THREADS, num_workers=config.WORKERS,
            )
            self._device, self._compute_type = selected, compute_type
            self._requested_device = requested

    def set_model_source(self, source):
        """Switch weights through the same lifecycle; never keep two models."""
        source = os.fspath(source)
        with self._lock:
            if source != self.model_source:
                self.unload_model()
                self.model_source = source

    def switch_device(self, device):
        """Wait for active transcription, then load on the requested device."""
        self.load_model(device=device)

    def unload_model(self):
        with self._lock:
            had_model = self._model is not None
            self._model = None
            self._device = self._compute_type = None
            if had_model:
                # CTranslate2 owns its allocations, not the torch allocator.
                gc.collect()

    def transcribe(self, media_path, *, device=None, on_stage=None, on_progress=None):
        """Transcribe local media, optionally selecting a device and reporting stages.

        on_stage receives preparing/loading/switching/transcribing on the calling
        thread. UI adapters must forward notifications to their UI thread.
        """
        if device is not None and device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be auto, cpu or cuda")
        def notify(stage):
            logger.info("Stage: %s; requested device: %s", stage, device or self._requested_device)
            if on_stage is not None:
                on_stage(stage)
        with self._lock:
            logger.info("Input media: %s", os.path.abspath(os.fspath(media_path)))
            notify("preparing")
            with prepare_media(media_path) as audio_path:
                logger.info("Prepared audio: %s", audio_path)
                if self._model is None:
                    notify("loading")
                    self.load_model(device=device)
                elif device is not None and device != self._device:
                    notify("switching")
                    self.switch_device(device)
                notify("transcribing")
                start = time.perf_counter()
                try:
                    options = deepcopy(self._options)
                    batched = config.BATCHED_TRANSCRIPTION and options.get("vad_filter", False)
                    engine = _create_batch_pipeline(self._model) if batched else self._model
                    if batched:
                        options.update(batch_size=config.GPU_BATCH_SIZE if self._device == "cuda" else config.CPU_BATCH_SIZE,
                                       chunk_length=config.CHUNK_LENGTH)
                    logger.info("Inference: device=%s compute=%s batch=%s chunk_sec=%s",
                                self._device, self._compute_type, options.get("batch_size", 1), options.get("chunk_length", 30))
                    segments, info = engine.transcribe(audio_path, **options)
                    text_parts = []
                    count = 0
                    # Consume lazy inference before releasing model/audio.
                    for segment in segments:
                        text_parts.append(segment.text.strip())
                        count += 1
                        if on_progress is not None:
                            end = min(info.duration, max(0.0, segment.end))
                            on_progress(end, info.duration, " ".join(text_parts).strip())
                except av.FFmpegError as exc:
                    raise MediaError("Аудиофайл повреждён или его формат не поддерживается") from exc
                elapsed = round(time.perf_counter() - start, 2)
                logger.info("Completed: audio_sec=%.2f inference_sec=%.2f segments=%d", info.duration, elapsed, count)
                return TranscriptionResult(
                    text=" ".join(text_parts).strip(),
                    duration_sec=elapsed,
                    language=info.language,
                    segments_count=count,
                    audio_duration_sec=round(info.duration, 2),
                )
