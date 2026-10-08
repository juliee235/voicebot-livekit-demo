"""LiveKit TTS adapter for PyThaiTTS (FastThaiG2P + Thai Kokoro-82M, CPU, 24 kHz mono).

Gemini tokens are grouped into whole sentences (ending in ค่ะ/คะ/ครับ or punctuation) and
each sentence is synthesized in one pass, with a short pause between sentences.
PyThaiTTS's own `stream()` splits at every space, which gives each phrase its own
intonation and clipped edges and makes the speech sound choppy.
The engine is synchronous, so it runs in a worker thread and pushes audio back to the
asyncio loop as each sentence is ready.

Run `uv run python thai_tts.py` once to download the model (~330 MB) and write test.wav.
"""

from __future__ import annotations

import asyncio
import queue
import re
import threading
from collections.abc import Iterable, Iterator

import numpy as np
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, tts, utils
from pythaitts.preprocess import preprocess_text
from pythaitts.pretrained.fastthaig2p import FastThaiG2P
from pythaitts.realtime import chunk_text

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
SENTENCE_GAP = np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.int16).tobytes()
SENTENCE_END = re.compile(r"(ค่ะ|คะ|ครับ|จ้ะ|จ้า|[.!?…])$")
MAX_SENTENCE_CHARS = 150  # cut a long run without an ending particle at a space
MAX_PHONEMES = 450  # Kokoro's hard limit is 510
_END = object()


def split_sentences(tokens: Iterable[str]) -> Iterator[str]:
    """Group streamed text into sentences, yielding each one as soon as it is complete."""
    pending = ""
    sentence: list[str] = []
    for token in tokens:
        pending += token
        *phrases, pending = re.split(r"\s+", pending)
        for phrase in phrases:
            if not phrase:
                continue
            sentence.append(phrase)
            if SENTENCE_END.search(phrase) or len(" ".join(sentence)) >= MAX_SENTENCE_CHARS:
                yield " ".join(sentence)
                sentence = []
    if pending.strip():
        sentence.append(pending.strip())
    if sentence:
        yield " ".join(sentence)


class ThaiTTS(tts.TTS):
    def __init__(self, *, speed: float = 1.0) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        # downloads the model to ~/.cache/fastthaig2p on first use
        self._engine = FastThaiG2P(speed=speed)

    @property
    def model(self) -> str:
        return "fastthaig2p"

    @property
    def provider(self) -> str:
        return "pythaitts"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)

    def pcm_chunks(self, text: str | Iterable[str]) -> Iterator[bytes]:
        """Blocking generator of 16-bit PCM chunks, one per synthesized sentence."""
        tokens = [text] if isinstance(text, str) else text
        for i, sentence in enumerate(split_sentences(tokens)):
            audio = self._synthesize_sentence(preprocess_text(sentence))
            if len(audio) == 0:
                continue
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            yield (SENTENCE_GAP + pcm) if i > 0 else pcm

    def _synthesize_sentence(self, text: str) -> np.ndarray:
        if len(self._engine._text_to_phonemes(text)) <= MAX_PHONEMES:
            return self._engine.generate(text)
        # rare: a very long sentence; let PyThaiTTS split it under the phoneme limit
        parts = chunk_text(
            text,
            max_phonemes=MAX_PHONEMES,
            preprocess=False,
            g2p_converter=self._engine._text_to_phonemes,
        )
        return np.concatenate([self._engine.generate(p) for p in parts])

    async def _synthesize_into(
        self, text: str | Iterable[str], output_emitter: tts.AudioEmitter
    ) -> None:
        loop = asyncio.get_running_loop()
        stop = threading.Event()

        def worker() -> None:
            for pcm in self.pcm_chunks(text):
                if stop.is_set():
                    return
                try:
                    loop.call_soon_threadsafe(output_emitter.push, pcm)
                except RuntimeError:  # event loop already closed
                    return

        try:
            await asyncio.to_thread(worker)
        finally:
            stop.set()


class ChunkedStream(tts.ChunkedStream):
    """Whole text in, audio out (used by `session.say(...)`)."""

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        assert isinstance(self._tts, ThaiTTS)
        await self._tts._synthesize_into(self._input_text, output_emitter)
        output_emitter.flush()


class SynthesizeStream(tts.SynthesizeStream):
    """LLM tokens in, audio out. One segment per stream (one agent reply)."""

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        tokens: queue.Queue[object] = queue.Queue()

        def token_iter() -> Iterator[str]:
            while (token := tokens.get()) is not _END:
                yield token  # type: ignore[misc]

        async def forward_input() -> None:
            try:
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        break
                    self._mark_started()
                    tokens.put(data)
            finally:
                tokens.put(_END)  # also unblocks the worker thread on interruption

        output_emitter.start_segment(segment_id=utils.shortuuid())
        input_task = asyncio.create_task(forward_input())
        try:
            assert isinstance(self._tts, ThaiTTS)
            await self._tts._synthesize_into(token_iter(), output_emitter)
        finally:
            await utils.aio.cancel_and_wait(input_task)
        output_emitter.end_segment()


if __name__ == "__main__":
    import time
    import wave

    t0 = time.perf_counter()
    thai = ThaiTTS()
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    def fake_llm() -> Iterator[str]:
        yield from ["สวัสดี", "ครับ ", "ยินดี", "ต้อนรับ", "สู่", "ระบบ", "เสียง", "ภาษาไทย ",
                    "วันนี้", "มี", "อะไร", "ให้", "ช่วย", "ไหม", "ครับ"]

    t0 = time.perf_counter()
    audio = b""
    for i, pcm in enumerate(thai.pcm_chunks(fake_llm())):
        if i == 0:
            print(f"first audio chunk after {time.perf_counter() - t0:.2f}s")
        audio += pcm
    seconds = len(audio) / 2 / SAMPLE_RATE
    print(f"{seconds:.1f}s of audio in {time.perf_counter() - t0:.2f}s")

    with wave.open("test.wav", "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio)
    print("wrote test.wav")
