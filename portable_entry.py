"""Frozen entry point and explicit diagnostics for the distributed runtime."""
import argparse
import json
from pathlib import Path
import sys


def main():
    if "--check" in sys.argv or "--check-gui" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--check")
        parser.add_argument("--check-gui", action="store_true")
        parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        parser.add_argument("--report", required=True)
        args = parser.parse_args()
        report = {"ok": False}
        try:
            if args.check_gui:
                from PySide6.QtCore import QTimer
                from PySide6.QtWidgets import QApplication
                from desktop import MainWindow
                app = QApplication([])
                window = MainWindow()
                window.show()
                timer = QTimer()
                def close_ready():
                    if window._ready and not window._closing:
                        window.close()
                timer.timeout.connect(close_ready)
                timer.start(100)
                app.exec()
                report.update(ok=window._stopped and not window._thread.isRunning(), mode="gui")
            else:
                from dataclasses import asdict
                from transcription_service import TranscriptionService
                service = TranscriptionService(device=args.device)
                try:
                    result = service.transcribe(args.check, device=args.device)
                    report.update(ok=True, device=service.device, model=service.model_source, result=asdict(result))
                finally:
                    service.unload_model()
        except Exception:
            import traceback
            report["error"] = traceback.format_exc()
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if report["ok"] else 1
    from desktop import main as desktop_main
    return desktop_main()


if __name__ == "__main__":
    sys.exit(main())
