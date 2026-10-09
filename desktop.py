"""Native desktop entry point. Run with the project's Python environment."""

import logging
import faulthandler
from pathlib import Path
import sys

from PySide6.QtCore import QSettings, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

import config
from desktop_worker import TranscriptionWorker
from transcription_service import TranscriptionService

logger = logging.getLogger(__name__)
STAGES = {
    "preparing": "Подготовка файла…",
    "loading": "Загрузка модели…",
    "switching": "Переключение устройства и загрузка модели…",
    "transcribing": "Транскрибация…",
}


class MainWindow(QMainWindow):
    transcribe_requested = Signal(str, str)
    shutdown_requested = Signal()

    def __init__(self, service=None, settings=None):
        super().__init__()
        self._settings = settings if settings is not None else QSettings("LinerTranscriber", "Desktop")
        self._path = None
        self._result_text = ""
        self._ready = self._busy = self._closing = self._stopped = False
        self.setWindowTitle("Транскрибация аудио и видео")
        self.resize(900, 720)
        self.setMinimumSize(600, 480)
        self.setAcceptDrops(True)
        self._build_ui()

        # One service and one persistent worker. No HTTP or second model owner.
        self._thread = QThread(self)
        self._worker = TranscriptionWorker(service if service is not None else TranscriptionService())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.check_devices)
        self.transcribe_requested.connect(self._worker.transcribe)
        self.shutdown_requested.connect(self._worker.shutdown)
        self._worker.devices_ready.connect(self._devices_ready)
        self._worker.stage_changed.connect(self._stage_changed)
        self._worker.progress_changed.connect(self._progress_changed)
        self._worker.succeeded.connect(self._succeeded)
        self._worker.failed.connect(self._failed)
        self._worker.operation_finished.connect(self._operation_finished)
        self._worker.shutdown_finished.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_finished)
        self._update_controls()
        self._thread.start()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        title = QLabel("Аудио и видео → текст")
        font = title.font()
        font.setPointSize(22)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        subtitle = QLabel("Выберите файл и устройство для распознавания речи.")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        drop_area = QFrame()
        drop_area.setFrameShape(QFrame.Shape.StyledPanel)
        file_layout = QVBoxLayout(drop_area)
        file_layout.setContentsMargins(18, 18, 18, 18)
        self.file_label = QLabel("Перетащите сюда аудио или видео")
        self.file_label.setWordWrap(True)
        self.file_label.setTextFormat(Qt.TextFormat.PlainText)
        self.file_label.setMinimumHeight(38)
        file_layout.addWidget(self.file_label)
        self.choose_button = QPushButton("Выбрать файл")
        self.choose_button.clicked.connect(self.choose_file)
        file_layout.addWidget(self.choose_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(drop_area)

        options = QHBoxLayout()
        options.addWidget(QLabel("Устройство:"))
        self.device_selector = QComboBox()
        self.device_selector.addItem("CPU", "cpu")
        self.device_selector.addItem("NVIDIA GPU (CUDA)", "cuda")
        self.device_selector.model().item(1).setEnabled(False)
        self.device_selector.currentIndexChanged.connect(self._device_changed)
        options.addWidget(self.device_selector, 1)
        self.start_button = QPushButton("Транскрибировать")
        self.start_button.setMinimumHeight(36)
        self.start_button.clicked.connect(self.start_transcription)
        options.addWidget(self.start_button)
        layout.addLayout(options)

        self.status_label = QLabel("Проверка доступных устройств…")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        layout.addWidget(self.progress)
        self.result_edit = QPlainTextEdit()
        self.result_edit.setReadOnly(True)
        self.result_edit.setAcceptDrops(False)
        self.result_edit.viewport().setAcceptDrops(False)
        self.result_edit.setPlaceholderText("Здесь появится текст транскрипции")
        layout.addWidget(self.result_edit, 1)
        actions = QHBoxLayout()
        actions.addStretch()
        self.copy_button = QPushButton("Копировать")
        self.copy_button.clicked.connect(self.copy_text)
        self.save_button = QPushButton("Сохранить как…")
        self.save_button.clicked.connect(self.save_text)
        actions.addWidget(self.copy_button)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        self.setCentralWidget(root)

    @Slot(bool)
    def _devices_ready(self, cuda_available):
        if self._closing:
            return
        self.device_selector.model().item(1).setEnabled(cuda_available)
        self.device_selector.setItemData(1, "" if cuda_available else "CUDA недоступна. Используйте CPU.", Qt.ItemDataRole.ToolTipRole)
        preferred = self._settings.value("device", config.DEVICE)
        self.device_selector.setCurrentIndex(1 if cuda_available and preferred != "cpu" else 0)
        self._ready = True
        self.status_label.setText("Файл выбран" if self._path else "Выберите или перетащите файл")
        self._update_controls()

    @Slot(int)
    def _device_changed(self, index):
        if self._ready and not self._busy and not self._closing:
            self._settings.setValue("device", self.device_selector.currentData())
            self._settings.sync()
            # Actual switching/loading is deferred to the next worker operation.

    def _update_controls(self):
        enabled = self._ready and not self._busy and not self._closing
        self.choose_button.setEnabled(not self._busy and not self._closing)
        self.device_selector.setEnabled(enabled)
        self.start_button.setEnabled(enabled and self._path is not None)
        self.copy_button.setEnabled(bool(self._result_text) and not self._closing)
        self.save_button.setEnabled(bool(self._result_text) and not self._closing)
        self.progress.setVisible(self._busy or not self._ready or self._closing)

    @Slot()
    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите аудио или видео", "",
            "Аудио и видео (*.wav *.mp3 *.flac *.ogg *.opus *.m4a *.aac *.mp4 *.mkv *.mov *.avi *.webm *.wma *.wmv);;Все файлы (*)",
        )
        if path:
            self.select_file(path)

    def select_file(self, path):
        if self._busy or self._closing:
            return
        path = Path(path)
        if not path.is_file():
            self._failed("Выберите существующий файл, а не папку.")
            return
        # Real format detection belongs exclusively to preprocessing at Start.
        self._path = path
        self.file_label.setText(path.name)
        self.file_label.setToolTip(str(path))
        self._result_text = ""
        self.result_edit.clear()
        self.status_label.setText("Файл выбран")
        self._update_controls()

    @Slot()
    def start_transcription(self):
        if not self._ready or self._busy or self._closing or self._path is None:
            return
        self._busy = True
        self.progress.setRange(0, 0)
        self._result_text = ""
        self.result_edit.clear()
        self.status_label.setText("Подготовка файла…")
        self._update_controls()
        self.transcribe_requested.emit(str(self._path), self.device_selector.currentData())

    @Slot(str)
    def _stage_changed(self, stage):
        if not self._closing:
            self.status_label.setText(STAGES.get(stage, "Обработка…"))

    @Slot(object)
    def _succeeded(self, result):
        if not self._closing:
            self._result_text = result.text  # Exact source for copy/save.
            self.result_edit.setPlainText(result.text)
            self.status_label.setText("Готово" if result.text else "Готово — речь не обнаружена")

    @Slot(float, float, str)
    def _progress_changed(self, processed, total, text):
        if self._closing:
            return
        if total > 0:
            self.progress.setRange(0, 1000)
            # Audio coverage, not a prediction of remaining execution time.
            self.progress.setValue(min(1000, int(processed / total * 1000)))
            self.status_label.setText(f"Транскрибация: обработано {processed:.0f} из {total:.0f} с аудио")
        self.result_edit.setPlainText(text)

    @Slot(str)
    def _failed(self, message):
        if not self._closing:
            self.status_label.setText(f"Ошибка: {message}")

    @Slot()
    def _operation_finished(self):
        self._busy = False
        self._update_controls()

    @Slot()
    def copy_text(self):
        if self._result_text:
            QApplication.clipboard().setText(self._result_text)
            self.statusBar().showMessage("Текст скопирован", 3000)

    @Slot()
    def save_text(self):
        if not self._result_text:
            return
        suggested = f"{self._path.stem if self._path else 'transcript'}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить текст", suggested, "Текстовые файлы (*.txt)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".txt"
        try:
            # Bytes prevent Windows newline conversion.
            Path(path).write_bytes(self._result_text.encode("utf-8"))
            self.statusBar().showMessage("Текст сохранён", 3000)
        except OSError:
            logger.exception("Cannot save transcription to %s", path)
            self._failed("Не удалось сохранить текст. Проверьте права доступа и свободное место.")

    def _drop_path(self, mime):
        if self._busy or self._closing or not mime.hasUrls():
            return None
        urls = mime.urls()
        return urls[0].toLocalFile() if len(urls) == 1 and urls[0].isLocalFile() else None

    def dragEnterEvent(self, event):
        if self._drop_path(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self._drop_path(event.mimeData())
        if path:
            self.select_file(path)
            event.acceptProposedAction()
        else:
            event.ignore()

    def closeEvent(self, event):
        if self._stopped:
            event.accept()
            return
        event.ignore()
        if not self._closing:
            self._closing = True
            self.status_label.setText("Закрытие после завершения текущей операции…" if self._busy else "Освобождение модели…")
            self._update_controls()
            self.shutdown_requested.emit()

    @Slot()
    def _thread_finished(self):
        self._stopped = True
        QTimer.singleShot(0, self.close)


def main():
    log_directory = Path(__file__).resolve().parent / "logs"
    log_directory.mkdir(exist_ok=True)
    crash_log = (log_directory / "desktop-crash.log").open("a", encoding="utf-8")
    faulthandler.enable(file=crash_log, all_threads=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_directory / "desktop.log", encoding="utf-8")],
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Liner Transcriber")
    window = MainWindow()
    window.show()
    try:
        return app.exec()
    finally:
        faulthandler.disable()
        crash_log.close()


if __name__ == "__main__":
    sys.exit(main())
