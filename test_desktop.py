"""Quick headless Qt checks; no Whisper model, download or GPU inference."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PySide6.QtCore import QMimeData, QPoint, QPointF, QSettings, QTimer, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import desktop
from preprocessing import MediaError


class FakeService:
    def __init__(self):
        self.cuda = True
        self.error = None
        self.stage = "transcribing"
        self.gate = threading.Event()
        self.started = threading.Event()
        self.calls = []
        self.thread_ids = []
        self.unloaded = False
        self.text = "Привет, мир!\r\nВторая строка.\n"

    def set_model_source(self, source):
        self.model_source = source

    def cuda_available(self):
        self.thread_ids.append(threading.get_ident())
        if isinstance(self.cuda, Exception):
            raise self.cuda
        return self.cuda

    def transcribe(self, path, *, device, on_stage, on_progress=None):
        self.thread_ids.append(threading.get_ident())
        self.calls.append((path, device))
        on_stage("preparing")
        self.started.set()
        if not self.gate.wait(3):
            raise TimeoutError("Test did not release worker")
        on_stage("loading")
        on_stage(self.stage)
        if self.error:
            raise self.error
        if on_progress is not None:
            on_progress(15.0, 30.0, self.text)
        return SimpleNamespace(text=self.text)

    def unload_model(self):
        self.thread_ids.append(threading.get_ident())
        self.unloaded = True


class FakeFormatter:
    def __init__(self):
        self.calls = []
        self.thread_ids = []
        self.error = None
        self.closed = False
        self.started = threading.Event()
        self.gate = threading.Event()
        self.gate.set()

    def format(self, text, instruction, *, on_progress, cancel):
        self.calls.append((text, instruction))
        self.thread_ids.append(threading.get_ident())
        self.started.set()
        while not self.gate.wait(0.01):
            if cancel.is_set():
                from text_formatting import FormattingCancelled
                raise FormattingCancelled("Форматирование отменено")
        if self.error:
            raise self.error
        on_progress("Предварительный конспект")
        return "Готовый конспект"

    def close(self):
        self.closed = True


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.file = self.root / "input.unknown"
        self.file.write_bytes(b"test input")
        self.service = FakeService()
        self.formatter = FakeFormatter()
        self.settings = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
        self.window = None

    def pump_until(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertTrue(predicate(), "Qt event did not complete")

    def open_window(self):
        self.window = desktop.MainWindow(self.service, self.settings, formatter=self.formatter)
        self.window.show()
        self.pump_until(lambda: self.window._ready)
        return self.window

    def tearDown(self):
        self.service.gate.set()
        self.formatter.gate.set()
        if self.window is not None:
            self.window.close()
            self.pump_until(lambda: self.window._stopped)
            self.assertTrue(self.window._thread.wait(1000))
            self.app.processEvents()
            self.window.deleteLater()
            self.app.processEvents()
        self.temp.cleanup()

    def test_success_thread_responsiveness_no_double_start_and_copy_save(self):
        window = self.open_window()
        window.select_file(self.file)
        window.device_selector.setCurrentIndex(0)
        self.assertEqual(self.settings.value("device"), "cpu")
        window.start_transcription()
        self.pump_until(self.service.started.is_set)
        self.assertFalse(window.device_selector.isEnabled())
        self.assertFalse(window.choose_button.isEnabled())
        self.assertFalse(window.start_button.isEnabled())
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        self.pump_until(lambda: bool(heartbeat))
        window.start_transcription()  # Must not queue a second operation.
        self.assertEqual(self.service.calls, [(str(self.file), "cpu")])
        self.service.gate.set()
        self.pump_until(lambda: not window._busy)
        self.assertTrue(window.device_selector.isEnabled())
        self.assertTrue(window.start_button.isEnabled())
        self.assertEqual(window._result_text, self.service.text)
        self.assertEqual(window.status_label.text(), "Готово")
        self.assertEqual(window.progress.maximum(), 1000)
        self.assertEqual(window.progress.value(), 500)
        window.copy_text()
        self.assertEqual(self.app.clipboard().text(), self.service.text)
        target = self.root / "result.txt"
        with patch.object(desktop.QFileDialog, "getSaveFileName", return_value=(str(target), "")):
            window.save_text()
        self.assertEqual(target.read_bytes(), self.service.text.encode("utf-8"))
        window.device_selector.setCurrentIndex(1)
        window.start_transcription()
        self.pump_until(lambda: not window._busy)
        self.assertEqual(self.service.calls[-1][1], "cuda")
        self.assertTrue(all(i != threading.get_ident() for i in self.service.thread_ids))

    def test_errors_restore_controls_and_worker_remains_usable(self):
        window = self.open_window()
        window.select_file(self.file)
        self.service.gate.set()
        for stage, error in (("preparing", MediaError("Видео не содержит аудиодорожки")),
                             ("loading", RuntimeError("private runtime details")),
                             ("transcribing", RuntimeError("private runtime details"))):
            with self.subTest(stage=stage):
                self.service.stage, self.service.error = stage, error
                with self.assertLogs("desktop_worker", level="ERROR"):
                    window.start_transcription()
                    self.pump_until(lambda: not window._busy)
                self.assertTrue(window.status_label.text().startswith("Ошибка:"))
                self.assertNotIn("private runtime details", window.status_label.text())
                self.assertTrue(window.start_button.isEnabled())
                self.assertTrue(window.device_selector.isEnabled())
        self.service.error = None
        window.start_transcription()
        self.pump_until(lambda: not window._busy)
        self.assertEqual(window.status_label.text(), "Готово")

    def test_cuda_unavailable_keeps_cpu_and_disables_gpu(self):
        self.service.cuda = False
        window = self.open_window()
        self.assertEqual(window.device_selector.currentData(), "cpu")
        self.assertFalse(window.device_selector.model().item(1).isEnabled())
        self.assertTrue(window.device_selector.model().item(0).isEnabled())

    def test_cuda_discovery_exception_does_not_crash(self):
        self.service.cuda = RuntimeError("CUDA DLL unavailable")
        with self.assertLogs("desktop_worker", level="ERROR"):
            window = self.open_window()
        self.assertEqual(window.device_selector.currentData(), "cpu")
        self.assertFalse(window.device_selector.model().item(1).isEnabled())

    def test_saved_cpu_preference_overrides_gpu_default(self):
        self.settings.setValue("device", "cpu")
        window = self.open_window()
        self.assertEqual(window.device_selector.currentData(), "cpu")

    def test_drag_drop_and_dialog_share_selection_and_pipeline(self):
        window = self.open_window()
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(self.file))])
        enter = QDragEnterEvent(QPoint(20, 20), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(window, enter)
        self.assertTrue(enter.isAccepted())
        drop = QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(window, drop)
        self.assertEqual(window._path, self.file)
        self.assertEqual(window.file_label.text(), self.file.name)
        self.service.gate.set()
        window.start_transcription()
        self.pump_until(lambda: not window._busy)
        with patch.object(desktop.QFileDialog, "getOpenFileName", return_value=(str(self.file), "")):
            window.choose_file()
        window.start_transcription()
        self.pump_until(lambda: not window._busy)
        self.assertEqual(self.service.calls[0], self.service.calls[1])

    def test_close_during_work_waits_and_unloads_in_worker(self):
        window = self.open_window()
        window.select_file(self.file)
        window.start_transcription()
        self.pump_until(self.service.started.is_set)
        window.close()
        self.assertTrue(window._closing)
        self.assertFalse(window._stopped)
        self.assertTrue(window._thread.isRunning())
        self.assertFalse(self.service.unloaded)
        self.service.gate.set()
        self.pump_until(lambda: window._stopped)
        self.assertTrue(window._thread.wait(1000))
        self.assertTrue(self.service.unloaded)
        self.assertNotEqual(self.service.thread_ids[-1], threading.get_ident())


    def test_opt_in_formatting_keeps_original_and_serializes_work(self):
        window = self.open_window()
        window.select_file(self.file)
        window.format_checkbox.setChecked(True)
        self.service.gate.set()
        self.formatter.gate.clear()
        with patch.object(window, "configure_formatting", return_value=True):
            window.start_transcription()
        self.pump_until(self.formatter.started.is_set)
        self.assertTrue(self.service.unloaded)
        self.assertTrue(window._busy)
        self.assertEqual(window._original_text, self.service.text)
        self.assertFalse(window.model_selector.isEnabled())
        self.assertFalse(window.format_button.isEnabled())
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        self.pump_until(lambda: bool(heartbeat))
        window.start_transcription()
        self.assertEqual(len(self.service.calls), 1)
        self.formatter.gate.set()
        self.pump_until(lambda: not window._busy)
        self.assertEqual(window._result_text, "Готовый конспект")
        window.copy_text()
        self.assertEqual(self.app.clipboard().text(), "Готовый конспект")
        target = self.root / "note.txt"
        with patch.object(desktop.QFileDialog, "getSaveFileName", return_value=(str(target), "")):
            window.save_text()
        self.assertEqual(target.read_text(encoding="utf-8"), "Готовый конспект")
        window.restore_original()
        self.assertEqual(window._result_text, self.service.text)
        self.assertNotEqual(self.formatter.thread_ids[0], threading.get_ident())

    def test_formatting_failure_and_close_preserve_raw_and_stop_thread(self):
        from text_formatting import FormattingError
        window = self.open_window()
        window.select_file(self.file)
        self.service.gate.set()
        window.start_transcription()
        self.pump_until(lambda: not window._busy)
        self.assertEqual(self.formatter.calls, [])  # SLM disabled by default.
        self.assertEqual(self.service.model_source, window.model_selector.currentData())
        self.formatter.error = FormattingError("Модель не установлена")
        with self.assertLogs("desktop_worker", level="ERROR"), patch.object(window, "configure_formatting", return_value=True):
            window.start_formatting()
            self.pump_until(lambda: not window._busy)
        self.assertEqual(window._result_text, self.service.text)
        self.assertEqual(window.result_edit.toPlainText(), self.service.text.replace("\r\n", "\n"))
        self.formatter.error = None
        self.formatter.started.clear()
        self.formatter.gate.clear()
        with patch.object(window, "configure_formatting", return_value=True):
            window.start_formatting()
        self.pump_until(self.formatter.started.is_set)
        with self.assertLogs("desktop_worker", level="ERROR"):
            window.close()
            self.pump_until(lambda: window._stopped)
        self.assertTrue(window._thread.wait(1000))
        self.assertTrue(self.formatter.closed)


class DesktopImportTests(unittest.TestCase):
    def test_no_http_or_model_imports_and_syntax(self):
        for filename in ("desktop.py", "desktop_worker.py", "transcription_service.py"):
            compile(Path(__file__).with_name(filename).read_text(encoding="utf-8"), filename, "exec")
        code = """
import sys
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'fastapi','starlette','uvicorn','torch','faster_whisper'}:
            raise AssertionError('Forbidden desktop import: ' + name)
sys.meta_path.insert(0, Block())
import desktop
assert not desktop.TranscriptionService().is_loaded
"""
        subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=15, cwd=Path(__file__).parent)


if __name__ == "__main__":
    unittest.main()
