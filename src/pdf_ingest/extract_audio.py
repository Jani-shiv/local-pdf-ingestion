"""Audio extraction using faster-whisper (local, fast)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


@dataclass
class AudioSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


class AudioExtractor:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def extract(
        self,
        audio_path: str | Path,
        language: str | None = None,
        vad_filter: bool = True,
    ) -> Generator[AudioSegment, None, None]:
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=vad_filter,
            word_timestamps=True,
        )
        for segment in segments:
            yield AudioSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
            )

    def extract_full_text(self, audio_path: str | Path, language: str | None = None) -> str:
        segments, _ = self.model.transcribe(str(audio_path), language=language)
        return " ".join(s.text.strip() for s in segments)


def extract_audio_text(
    audio_path: str | Path,
    model_size: str = "base",
    language: str | None = None,
) -> str:
    extractor = AudioExtractor(model_size=model_size)
    return extractor.extract_full_text(audio_path, language)


def extract_audio_segments(
    audio_path: str | Path,
    model_size: str = "base",
    language: str | None = None,
) -> list[AudioSegment]:
    extractor = AudioExtractor(model_size=model_size)
    return list(extractor.extract(audio_path, language))