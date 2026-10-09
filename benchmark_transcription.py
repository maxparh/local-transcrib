"""Explicit local-only comparison on selected clips; never downloads models."""

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import time

import config
from transcription_service import _prepare_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--compute", default="int8")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--chunk-length", type=int)
    parser.add_argument("--vad-threshold", type=float)
    parser.add_argument("--vad-silence", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _prepare_runtime()
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    from faster_whisper.audio import decode_audio

    start = time.perf_counter()
    model = WhisperModel(config.MODEL_SOURCE, device=args.device,
                         compute_type=args.compute, num_workers=args.workers,
                         cpu_threads=config.CPU_THREADS, local_files_only=True)
    report = {"device": args.device, "requested_compute": args.compute,
              "actual_compute": model.model.compute_type,
              "workers": args.workers, "batch": args.batch,
              "load_sec": round(time.perf_counter() - start, 3), "clips": []}
    engine = BatchedInferencePipeline(model=model) if args.batch else model
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}), flush=True)
    for filename in args.files * args.repeat:
        audio = decode_audio(filename, sampling_rate=16000)
        start_index = int(args.offset * 16000)
        audio = audio[start_index:start_index + int(args.seconds * 16000)]
        options = deepcopy(config.TRANSCRIPTION_OPTIONS)
        if args.batch:
            options["batch_size"] = args.batch
        if args.chunk_length:
            options["chunk_length"] = args.chunk_length
        if args.vad_threshold is not None:
            options["vad_parameters"]["threshold"] = args.vad_threshold
        if args.vad_silence is not None:
            options["vad_parameters"]["min_silence_duration_ms"] = args.vad_silence
        started = time.perf_counter()
        segments, info = engine.transcribe(audio, **options)
        segments = list(segments)
        elapsed = time.perf_counter() - started
        clip = {"file": filename, "offset_sec": args.offset, "audio_sec": len(audio) / 16000,
                "elapsed_sec": round(elapsed, 3), "text": " ".join(s.text for s in segments).strip(),
                "segments": [{"start": s.start, "end": s.end, "text": s.text,
                              "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob} for s in segments]}
        report["clips"].append(clip)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k:v for k,v in clip.items() if k != "segments"}, ensure_ascii=False), flush=True)
    del engine, model
    gc.collect()


if __name__ == "__main__":
    main()
