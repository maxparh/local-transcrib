"""Stable install/data paths; never write into a frozen bundle's extraction dir."""
import os
from pathlib import Path
import sys


def install_directory():
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def data_directory():
    override = os.environ.get("TRANSCRIBER_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False) and (install_directory() / "portable.flag").is_file():
        return install_directory() / "data"
    if not getattr(sys, "frozen", False):
        return install_directory()
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "LocalTranscriber"


def russian_model_source(repository):
    # Installer may provision shared read-only weights next to the executable,
    # or per-user weights under DATA_DIR. Both survive one-file extraction.
    for base in (data_directory(), install_directory()):
        candidate = base / "models" / "whisper-russian"
        if all((candidate / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")):
            return str(candidate)
    if getattr(sys, "frozen", False):
        return str(data_directory() / "models" / "whisper-russian")
    return repository  # Development first-run download, as before.
