"""Qt worker adapter. Core work and model cleanup happen off the UI thread."""

import logging

from PySide6.QtCore import QObject, Signal, Slot

from preprocessing import MediaError
from transcription_service import DeviceError

logger = logging.getLogger(__name__)


class TranscriptionWorker(QObject):
    devices_ready = Signal(bool)
    stage_changed = Signal(str)
    progress_changed = Signal(float, float, str)
    succeeded = Signal(object)
    failed = Signal(str)
    operation_finished = Signal()
    shutdown_finished = Signal()

    def __init__(self, service):
        super().__init__()
        self.service = service
        self._stage = "preparing"

    @Slot()
    def check_devices(self):
        try:
            available = self.service.cuda_available()
        except Exception:
            logger.exception("Device discovery failed")
            available = False
        self.devices_ready.emit(available)

    def _on_stage(self, stage):
        self._stage = stage
        self.stage_changed.emit(stage)

    @Slot(str, str)
    def transcribe(self, path, device):
        self._stage = "preparing"
        try:
            result = self.service.transcribe(path, device=device, on_stage=self._on_stage,
                                             on_progress=self.progress_changed.emit)
            self.succeeded.emit(result)
        except Exception as exc:
            logger.exception("Transcription failed during %s on %s", self._stage, device)
            if isinstance(exc, MediaError):
                message = str(exc)
            elif isinstance(exc, DeviceError):
                message = "Выбранное устройство недоступно. Попробуйте CPU или проверьте настройку CUDA."
            elif self._stage in {"loading", "switching"}:
                message = "Не удалось загрузить модель на выбранном устройстве. Проверьте источник модели и настройки устройства."
            elif self._stage == "preparing":
                message = "Не удалось подготовить файл. Проверьте формат, доступ к файлу и свободное место."
            else:
                message = "Не удалось выполнить транскрибацию. Попробуйте другой файл или CPU; подробности записаны в консоль."
            self.failed.emit(" ".join(message.split())[:500])
        finally:
            self.operation_finished.emit()

    @Slot()
    def shutdown(self):
        # Queued behind the active operation; no forced thread termination.
        try:
            self.service.unload_model()
        except Exception:
            logger.exception("Model cleanup failed")
        finally:
            self.shutdown_finished.emit()
