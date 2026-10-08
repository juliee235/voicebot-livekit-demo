"""LiveKit STT adapter for PyThaiASR Typhoon Nemotron streaming ASR (ONNX INT4, CPU, 16 kHz).

Nemotron decodes 0.56 s chunks with a cache-aware encoder and returns new tokens as they
are recognized, but it has no endpointing. A small Silero VAD inside the stream marks each
utterance: audio is fed to Nemotron only while the user speaks (plus a short pre-roll),
interim transcripts are emitted as text arrives, and the end of speech flushes the model
into a FINAL_TRANSCRIPT. The LiveKit turn detector then decides if the turn is over.

Run `uv run python thai_stt.py some.wav` to transcribe a file and time it.
"""

from __future__ import annotations

import asyncio
from collections import deque

import numpy as np
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    stt,
    utils,
)
from livekit.agents.vad import VAD, VADEventType
from pythaiasr.typhoon_nemotron_asr import NemotronStreamASR, NemotronStreamingASR

SAMPLE_RATE = 16000
PREROLL_SECONDS = 0.6  # audio kept before VAD start-of-speech fires
LANGUAGE = "th"


def _to_float(frame: rtc.AudioFrame) -> np.ndarray:
    return np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0


class ThaiSTT(stt.STT):
    def __init__(self, *, vad: VAD, language: str = "th-TH", device: str = "cpu") -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True)
        )
        self._vad = vad
        self._language = language
        # downloads ~700 MB to ~/pythaiasr-data on first use
        self._model = NemotronStreamingASR(
            precision="int4", device=device, default_language=language
        )

    @property
    def model(self) -> str:
        return "typhoon-asr-streaming-nemotron-0.6b-int4"

    @property
    def provider(self) -> str:
        return "pythaiasr"

    def new_session(self) -> NemotronStreamASR:
        return NemotronStreamASR(self._model, sample_rate=SAMPLE_RATE, language=self._language)

    def transcribe(self, audio: np.ndarray) -> str:
        """Blocking: transcribe a whole float32 16 kHz utterance."""
        session = self.new_session()
        session.process_chunk(audio)
        session.flush()
        return session.get_full_transcript().strip()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        frame = utils.merge_frames(buffer)
        if frame.sample_rate != SAMPLE_RATE or frame.num_channels != 1:
            resampler = rtc.AudioResampler(frame.sample_rate, SAMPLE_RATE, num_channels=1)
            frame = utils.merge_frames([*resampler.push(frame), *resampler.flush()])
        text = await asyncio.to_thread(self.transcribe, _to_float(frame))
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=LANGUAGE, text=text)],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(stt=self, conn_options=conn_options)


class SpeechStream(stt.RecognizeStream):
    def __init__(self, *, stt: ThaiSTT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._thai: ThaiSTT = stt

    async def _run(self) -> None:
        vad_stream = self._thai._vad.stream()
        session = self._thai.new_session()
        asr_lock = asyncio.Lock()  # Nemotron session state is not thread-safe

        speaking = False
        last_text = ""
        preroll: deque[np.ndarray] = deque()
        preroll_samples = 0
        pending: list[np.ndarray] = []
        pending_samples = 0

        def emit(type: stt.SpeechEventType, text: str = "") -> None:
            alts = [stt.SpeechData(language=LANGUAGE, text=text)] if text else []
            self._event_ch.send_nowait(stt.SpeechEvent(type=type, alternatives=alts))

        async def decode(audio: np.ndarray, *, flush: bool = False) -> str:
            def work() -> str:
                if len(audio):
                    session.process_chunk(audio)
                if flush:
                    session.flush()
                return session.get_full_transcript().strip()

            async with asr_lock:
                return await asyncio.to_thread(work)

        def take_pending() -> np.ndarray:
            nonlocal pending_samples
            audio = np.concatenate(pending) if pending else np.zeros(0, dtype=np.float32)
            pending.clear()
            pending_samples = 0
            return audio

        async def forward_audio() -> None:
            nonlocal preroll_samples, pending_samples, last_text
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    vad_stream.flush()
                    continue
                vad_stream.push_frame(item)
                pcm = _to_float(item)

                if not speaking:
                    preroll.append(pcm)
                    preroll_samples += len(pcm)
                    while preroll_samples > PREROLL_SECONDS * SAMPLE_RATE:
                        preroll_samples -= len(preroll.popleft())
                    continue

                pending.append(pcm)
                pending_samples += len(pcm)
                if pending_samples >= session.chunk_samples:
                    text = await decode(take_pending())
                    if text and speaking and text != last_text:
                        last_text = text
                        emit(stt.SpeechEventType.INTERIM_TRANSCRIPT, text)
            vad_stream.end_input()

        async def handle_vad() -> None:
            nonlocal speaking, preroll_samples, pending_samples, last_text
            async for ev in vad_stream:
                if ev.type == VADEventType.START_OF_SPEECH:
                    async with asr_lock:
                        session.reset()
                    pending[:] = list(preroll)
                    pending_samples = preroll_samples
                    preroll.clear()
                    preroll_samples = 0
                    last_text = ""
                    speaking = True
                    emit(stt.SpeechEventType.START_OF_SPEECH)

                elif ev.type == VADEventType.END_OF_SPEECH:
                    speaking = False
                    text = await decode(take_pending(), flush=True)
                    if text:
                        emit(stt.SpeechEventType.FINAL_TRANSCRIPT, text)
                    emit(stt.SpeechEventType.END_OF_SPEECH)

        tasks = [
            asyncio.create_task(forward_audio(), name="thai_stt.forward_audio"),
            asyncio.create_task(handle_vad(), name="thai_stt.handle_vad"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await vad_stream.aclose()


if __name__ == "__main__":
    import sys
    import time

    import soundfile as sf
    from livekit.plugins import silero

    t0 = time.perf_counter()
    thai = ThaiSTT(vad=silero.VAD.load())
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    audio, sr = sf.read(sys.argv[1], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from pythaiasr.typhoon import resample_audio

        audio = resample_audio(audio, sr, SAMPLE_RATE)

    session = thai.new_session()
    step = session.chunk_samples
    t0 = time.perf_counter()
    for i in range(0, len(audio), step):
        t1 = time.perf_counter()
        if text := session.process_chunk(audio[i : i + step]):
            print(f"  +{(time.perf_counter() - t1) * 1000:4.0f} ms  {session.get_full_transcript()}")
    session.flush()
    elapsed = time.perf_counter() - t0
    print(f"final: {session.get_full_transcript()}")
    print(f"{len(audio) / SAMPLE_RATE:.1f}s audio in {elapsed:.2f}s (RTF {elapsed / (len(audio) / SAMPLE_RATE):.2f})")
