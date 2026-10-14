"""Formatter contract checks: no downloads, Ollama process or GPU required."""
import asyncio
import json
import threading
import time
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import httpx

import config
import app_paths
from text_formatting import FormattingCancelled, FormattingError, OllamaFormatter


class FormatterTests(unittest.TestCase):
    def make_formatter(self, *, lines=None, missing=False):
        self.requests = []
        def handle(request):
            self.requests.append(request)
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "test"})
            if request.url.path == "/api/show":
                return httpx.Response(404 if missing else 200, json={})
            if request.url.path == "/api/generate":
                return httpx.Response(200, json={"done": True})
            if request.url.path == "/api/chat":
                events = lines if lines is not None else [
                    {"message": {"content": "Конспект: "}},
                    {"message": {"content": "350 рублей."}, "done": True, "done_reason": "stop"},
                ]
                body = "\n".join(json.dumps(item, ensure_ascii=False) for item in events)
                return httpx.Response(200, text=body)
            raise AssertionError(request.url.path)
        return OllamaFormatter(transport=httpx.MockTransport(handle))

    def test_streaming_contract_preserves_instruction_and_disables_thinking(self):
        formatter = self.make_formatter()
        progress = []
        result = formatter.format("Вызов от 350 рублей, если неисправность внутри квартиры.", "Конспект", on_progress=progress.append)
        self.assertEqual(result, "Конспект: 350 рублей.")
        payload = json.loads(self.requests[-1].content)
        self.assertEqual(payload["model"], "qwen3:4b-instruct")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["keep_alive"], 0)
        self.assertIn("350 рублей", payload["messages"][1]["content"])
        self.assertTrue(progress)

    def test_missing_component_and_incomplete_outputs_are_errors(self):
        for formatter in (
            self.make_formatter(missing=True),
            self.make_formatter(lines=[{"message": {"content": "partial"}}]),
            self.make_formatter(lines=[{"message": {"content": "partial"}, "done": True, "done_reason": "length"}]),
            self.make_formatter(lines=[{"done": True, "message": {"content": ""}}]),
            self.make_formatter(lines=[{"error": "native backend error"}]),
        ):
            with self.assertRaises(FormattingError):
                formatter.format("Исходник", "Конспект")

    def test_cancel_during_stream_releases_model(self):
        formatter = self.make_formatter()
        cancel = Event()
        with self.assertRaises(FormattingCancelled):
            formatter.format("Исходник", "Конспект", cancel=cancel, on_progress=lambda text: cancel.set())
        self.assertEqual(self.requests[-1].url.path, "/api/generate")

    def test_limits_reject_before_runtime_and_do_not_silently_truncate(self):
        formatter = self.make_formatter()
        with patch.object(config, "FORMATTING_MAX_CHARS", 2):
            with self.assertRaises(FormattingError):
                formatter.format("123", "Конспект")
        self.assertEqual(self.requests, [])

    def test_cancel_before_first_token_does_not_wait_for_read_timeout(self):
        cancel = Event()
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.sleep(60)
                yield b"never reached"
        def handle(request):
            if request.url.path == "/api/chat":
                return httpx.Response(200, stream=SlowStream())
            return httpx.Response(200, json={})
        formatter = OllamaFormatter(transport=httpx.MockTransport(handle))
        timer = threading.Timer(0.1, cancel.set)
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaises(FormattingCancelled):
                formatter.format("Исходник", "Конспект", cancel=cancel)
            self.assertLess(time.monotonic() - started, 2)
        finally:
            timer.cancel()

    def test_remote_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            OllamaFormatter(url="https://example.org")


class InstallPathsTests(unittest.TestCase):
    def test_portable_marker_keeps_data_beside_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "portable.flag").write_text("portable")
            with patch.object(app_paths.sys, "frozen", True, create=True), patch.object(app_paths.sys, "executable", str(root / "LocalTranscriber.exe")), patch.dict(app_paths.os.environ, {}, clear=True):
                self.assertEqual(app_paths.data_directory(), root / "data")

    def test_frozen_assets_and_logs_do_not_use_extraction_or_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install = root / "Program Files" / "Transcriber"
            model = install / "models" / "whisper-russian"
            model.mkdir(parents=True)
            for name in ("model.bin", "config.json", "tokenizer.json"):
                (model / name).write_text("fixture")
            with patch.object(app_paths.sys, "frozen", True, create=True), patch.object(app_paths.sys, "executable", str(install / "desktop.exe")), patch.dict(app_paths.os.environ, {"TRANSCRIBER_DATA_DIR": str(root / "user-data")}):
                self.assertEqual(app_paths.data_directory(), root / "user-data")
                self.assertEqual(app_paths.russian_model_source("repository"), str(model))
                (model / "model.bin").unlink()
                self.assertEqual(app_paths.russian_model_source("repository"), str(root / "user-data" / "models" / "whisper-russian"))


if __name__ == "__main__":
    unittest.main()
