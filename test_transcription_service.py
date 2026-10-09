"""Fast checks: no model downloads, DLL workaround execution or GPU inference."""

from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave
import weakref

import ctranslate2

import config
import transcription_service as core


class FakeModel:
    def transcribe(self, path, **options):
        def segments():
            assert Path(path).exists()  # Lazy decoding must finish before cleanup.
            yield SimpleNamespace(text="hello", end=0.005)
            yield SimpleNamespace(text="world", end=0.01)
        return segments(), SimpleNamespace(language="ru", duration=0.01)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(config, "TEMP_DIR", str(self.root)))
        self.stack.enter_context(patch.object(config, "CPU_COMPUTE_TYPE", "int8"))
        self.stack.enter_context(patch.object(config, "GPU_COMPUTE_TYPE", "int8"))
        self.runtime = self.stack.enter_context(patch.object(core, "_prepare_runtime"))
        self.count = self.stack.enter_context(patch.object(ctranslate2, "get_cuda_device_count", return_value=1))
        self.stack.enter_context(patch.object(ctranslate2, "get_supported_compute_types", return_value={"int8", "float32"}))
        self.references = []

        def create(*args, **kwargs):
            self.assertTrue(all(ref() is None for ref in self.references))
            model = FakeModel()
            self.references.append(weakref.ref(model))
            return model

        self.factory = self.stack.enter_context(patch.object(core, "_create_model", side_effect=create))
        self.batch_factory = self.stack.enter_context(patch.object(core, "_create_batch_pipeline", side_effect=lambda model: model))
        self.service = core.TranscriptionService(device="cpu")
        self.addCleanup(self.service.unload_model)
        self.audio = self.root / "source.wav"
        with wave.open(str(self.audio), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(b"\0\0" * 160)

    def test_lifecycle_and_single_model(self):
        self.assertFalse(self.service.is_loaded)
        self.factory.assert_not_called()
        for device in ("cpu", "cuda", "cpu"):
            self.service.load_model(device)
            self.assertEqual(self.service.device, device)
            self.assertEqual(self.factory.call_args.kwargs["compute_type"], "int8")
        self.service.load_model("cpu")
        self.assertEqual(self.factory.call_count, 3)
        self.service.unload_model()
        self.assertFalse(self.service.is_loaded)
        self.assertIsNone(self.service.device)
        self.assertTrue(all(ref() is None for ref in self.references))

    def test_auto_and_explicit_device_selection(self):
        self.service.load_model("auto")
        self.assertEqual(self.service.device, "cuda")
        self.count.return_value = 0
        self.service.switch_device("auto")
        self.assertEqual(self.service.device, "cpu")
        with self.assertRaises(core.DeviceError):
            self.service.switch_device("cuda")
        self.assertEqual(self.service.device, "cpu")
        with self.assertRaises(ValueError):
            self.service.switch_device("invalid")

    def test_failed_reload_releases_old_instance_and_can_retry(self):
        self.service.load_model("cpu")
        factory = self.factory.side_effect
        self.factory.side_effect = RuntimeError("failed to load")
        with self.assertRaises(RuntimeError):
            self.service.switch_device("cuda")
        self.assertFalse(self.service.is_loaded)
        self.assertIsNone(self.service.device)
        self.assertIsNone(self.references[0]())
        self.factory.side_effect = factory
        self.service.switch_device("cuda")
        self.assertEqual(self.service.device, "cuda")

    def test_source_and_independent_compute_settings(self):
        with patch.object(config, "MODEL_SOURCE", "models/large-v3"), patch.object(config, "CPU_COMPUTE_TYPE", "float32"):
            service = core.TranscriptionService(device="cpu")
        try:
            service.load_model()
            self.assertEqual(self.factory.call_args.args[0], "models/large-v3")
            self.assertEqual(self.factory.call_args.kwargs["compute_type"], "float32")
            service.switch_device("cuda")
            self.assertEqual(self.factory.call_args.kwargs["compute_type"], "int8")
        finally:
            service.unload_model()

    def test_desktop_stages_and_switch_use_existing_lifecycle(self):
        stages = []
        self.service.transcribe(self.audio, device="cpu", on_stage=stages.append)
        self.assertEqual(stages, ["preparing", "loading", "transcribing"])
        stages.clear()
        # Mock call history must not keep an otherwise released model alive.
        self.batch_factory.reset_mock()
        with patch.object(self.service, "switch_device", wraps=self.service.switch_device) as switch:
            self.service.transcribe(self.audio, device="cuda", on_stage=stages.append)
            switch.assert_called_once_with("cuda")
        self.assertEqual(stages, ["preparing", "switching", "transcribing"])
        self.assertEqual(self.service.device, "cuda")
        self.assertEqual(self.factory.call_count, 2)

    def test_prepared_audio_lifetime_on_success_and_failure(self):
        import preprocessing
        def prepare(source, directory):
            target = directory / "prepared.wav"
            target.write_bytes(self.audio.read_bytes())
            return target
        with patch.object(preprocessing, "_prepare_audio", side_effect=prepare):
            result = self.service.transcribe(self.audio)
            self.assertEqual(result.text, "hello world")
            self.assertEqual(list(self.root.iterdir()), [self.audio])
            def failing_decode(path, **kwargs):
                def segments():
                    self.assertTrue(Path(path).exists())
                    raise RuntimeError("failure during lazy decoding")
                    yield
                return segments(), SimpleNamespace(language="ru", duration=0)
            with patch.object(FakeModel, "transcribe", side_effect=failing_decode):
                with self.assertRaises(RuntimeError):
                    self.service.transcribe(self.audio)
            self.assertEqual(list(self.root.iterdir()), [self.audio])

    def test_cuda_probe_registers_runtime_without_model(self):
        self.assertTrue(self.service.cuda_available())
        self.runtime.assert_called_once()
        self.factory.assert_not_called()
        self.count.return_value = 0
        with self.assertLogs(core.logger, level="WARNING"):
            self.assertFalse(self.service.cuda_available())

    def test_transcription_and_preprocessing_cleanup(self):
        result = self.service.transcribe(self.audio)
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.segments_count, 2)
        self.assertEqual(result.audio_duration_sec, 0.01)
        self.assertEqual(list(self.root.iterdir()), [self.audio])
        with patch.object(FakeModel, "transcribe", side_effect=RuntimeError("inference failed")):
            with self.assertRaises(RuntimeError):
                self.service.transcribe(self.audio)
        self.assertEqual(list(self.root.iterdir()), [self.audio])

    def test_batch_uses_same_model_and_reports_real_audio_coverage(self):
        progress = []
        result = self.service.transcribe(self.audio, on_progress=lambda *value: progress.append(value))
        self.assertEqual(self.batch_factory.call_count, 1)
        self.assertEqual(self.factory.call_count, 1)
        self.assertEqual(progress[-1], (0.01, 0.01, result.text))
        self.assertEqual(progress[0], (0.005, 0.01, "hello"))

    def test_disabled_vad_uses_sequential_pipeline(self):
        self.service._options["vad_filter"] = False
        self.service.transcribe(self.audio)
        self.batch_factory.assert_not_called()

class ImportTests(unittest.TestCase):
    def test_cached_model_is_used_without_online_lookup(self):
        factory = Mock()
        cached = Mock(return_value="cached/model")
        with patch.dict(sys.modules, {
            "faster_whisper": SimpleNamespace(WhisperModel=factory),
            "faster_whisper.utils": SimpleNamespace(download_model=cached),
        }):
            core._create_model("large-v3", device="cpu")
        cached.assert_called_once_with("large-v3", local_files_only=True)
        factory.assert_called_once_with("cached/model", device="cpu")

    def test_missing_cache_preserves_first_run_model_loading(self):
        from huggingface_hub.errors import LocalEntryNotFoundError
        factory = Mock()
        cached = Mock(side_effect=LocalEntryNotFoundError("not cached"))
        with patch.dict(sys.modules, {
            "faster_whisper": SimpleNamespace(WhisperModel=factory),
            "faster_whisper.utils": SimpleNamespace(download_model=cached),
        }):
            core._create_model("large-v3", device="cpu")
        factory.assert_called_once_with("large-v3", device="cpu")

    def test_import_without_web_or_model_runtime(self):
        code = """
import sys
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'fastapi', 'starlette', 'torch', 'faster_whisper', 'ctranslate2'}:
            raise AssertionError('Unexpected import: ' + name)
sys.meta_path.insert(0, Block())
from transcription_service import TranscriptionService
assert not TranscriptionService(device='cpu').is_loaded
assert not TranscriptionService(device='cuda').is_loaded
"""
        subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=15, cwd=Path(__file__).parent)

    def test_syntax(self):
        for name in ("config.py", "preprocessing.py", "transcription_service.py", "desktop.py", "desktop_worker.py"):
            compile(Path(__file__).with_name(name).read_text(encoding="utf-8-sig"), name, "exec")


if __name__ == "__main__":
    unittest.main()
