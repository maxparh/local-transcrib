"""Prepare local media without depending on HTTP, Whisper or GPU libraries."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
import wave
import logging

import av
from av.stream import Disposition

import config

logger = logging.getLogger(__name__)


class MediaError(Exception):
    """Input media cannot be decoded into usable audio."""


def _inspect_media(path):
    with av.open(str(path)) as container:
        # Attached pictures (album covers) do not make an audio file a video.
        media_type = "video" if any(
            not stream.disposition & Disposition.attached_pic
            for stream in container.streams.video
        ) else "audio"
        audio_streams = list(container.streams.audio)
        if not audio_streams:
            message = "Видео не содержит аудиодорожки" if media_type == "video" else "Файл не содержит аудиодорожки"
            raise MediaError(message)
        first_index = audio_streams[0].index
        indices = [stream.index for stream in audio_streams]
        logger.info("Media probe: type=%s, container_duration_sec=%s, audio_streams=%s",
                    media_type, container.duration / av.time_base if container.duration is not None else None, indices)

    # Rule: use the first audio stream in container order that can decode a
    # nonempty frame. Do not prefer a language or the 'default' disposition.
    for index in indices:
        try:
            with av.open(str(path)) as container:
                stream = container.streams[index]
                if stream.codec_context is None or not stream.codec_context.is_decoder:
                    continue
                for frame in container.decode(stream):
                    if frame.samples:
                        return media_type, index, first_index
        except av.FFmpegError:
            continue
    raise MediaError("Аудиодорожки повреждены, пусты или используют неподдерживаемый кодек")


def _extract_audio(source, destination, stream_index):
    # PCM mono 16 kHz is the input format used by the transcription decoder.
    with av.open(str(source)) as container, wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

        def write_frames(frames):
            for frame in frames:
                # Plane buffers may include padding; write only actual samples.
                output.writeframesraw(bytes(frame.planes[0])[:frame.samples * 2])

        for frame in container.decode(container.streams[stream_index]):
            # Resample consecutive decoded samples even with discontinuous PTS.
            frame.pts = None
            write_frames(resampler.resample(frame))
        write_frames(resampler.resample(None))


def _prepare_audio(source, directory):
    try:
        media_type, stream_index, first_index = _inspect_media(source)
        if media_type == "audio" and stream_index == first_index:
            return source  # Keep ordinary audio unchanged; no re-encoding.
        # Also isolate a fallback audio stream if the first one was undecodable:
        # Whisper otherwise selects the first audio stream itself.
        destination = directory / "audio.wav"
        _extract_audio(source, destination, stream_index)
        return destination
    except av.FFmpegError as exc:
        raise MediaError("Медиафайл повреждён или его формат не поддерживается") from exc


@contextmanager
def prepare_media(media_path):
    """Yield prepared audio; remove only generated files, never the input."""
    source = Path(media_path)
    if not source.is_file():
        raise MediaError("Входной файл не найден")
    if not source.stat().st_size:
        raise MediaError("Входной файл пуст")
    with tempfile.TemporaryDirectory(prefix="transcriber-audio-", dir=config.TEMP_DIR) as temporary:
        yield str(_prepare_audio(source, Path(temporary)))
