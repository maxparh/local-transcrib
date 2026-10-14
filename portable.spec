# Build with the existing project Python and isolated build/tools PyInstaller.
from pathlib import Path
from importlib.metadata import distribution
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH)
site = root / ".venv" / "Lib" / "site-packages"
datas = collect_data_files("faster_whisper")
for name in ("PySide6", "PySide6_Essentials", "PySide6_Addons", "shiboken6", "faster-whisper", "ctranslate2", "av", "onnxruntime", "tokenizers", "huggingface_hub", "torch", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12"):
    datas += copy_metadata(name)
binaries = []
for relative, pattern in (
    ("nvidia/cublas/bin", "cublas*.dll"),
    ("nvidia/cuda_runtime/bin", "*.dll"),
    ("nvidia/cuda_nvrtc/bin", "*.dll"),
    ("torch/lib", "cudnn*.dll"),
    ("torch/lib", "cudart64_13.dll"),
):
    for source in (site / relative).glob(pattern):
        binaries.append((str(source), relative))
a = Analysis([str(root / "portable_entry.py")], pathex=[str(root)],
             binaries=binaries, datas=datas,
             hiddenimports=collect_submodules("faster_whisper") + ["onnxruntime.capi.onnxruntime_pybind11_state", "ctranslate2._ext", "tokenizers"],
             excludes=["torch", "torchvision", "transformers", "tensorflow", "pandas", "scipy", "matplotlib", "IPython", "pytest", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="LocalTranscriber", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="LocalTranscriber-portable")
