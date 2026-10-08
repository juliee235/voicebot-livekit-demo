"""Thai voicebot: STT -> Gemini (LiveKit Inference) -> TTS, with Soniox or local models.

    STT_PROVIDER = "soniox" | "local"   # local: PyThaiASR Typhoon Nemotron on CPU
    TTS_PROVIDER = "soniox" | "local"   # local: PyThaiTTS FastThaiG2P on CPU

End of turn is decided by LiveKit's audio turn detector (v1-mini, local). Serves both the
web demo (WebRTC) and phone calls (SIP): the agent has no agent_name, so LiveKit
auto-dispatches it into every new room, including `call-*` rooms.

    uv run agent.py console   # talk from the terminal
    uv run agent.py dev       # connect to LiveKit Cloud
"""

import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    JobProcess,
    cli,
    inference,
    stt,
    tts,
)
from livekit.plugins import silero, soniox

from systemprompt import GREETING_INSTRUCTIONS, SYSTEM_PROMPT
from thai_stt import ThaiSTT
from thai_tts import ThaiTTS
from tools import build_tools

load_dotenv(".env")


STT_PROVIDER = "soniox"
TTS_PROVIDER = "soniox"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT, tools=build_tools())


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()
    # local models take a few seconds to load, so load them once per process
    if STT_PROVIDER == "local":
        # the STT's own VAD only cuts utterances for Nemotron, so it can react faster
        proc.userdata["stt"] = ThaiSTT(vad=silero.VAD.load(min_silence_duration=0.35))
    if TTS_PROVIDER == "local":
        proc.userdata["tts"] = ThaiTTS()


def build_stt(proc: JobProcess) -> stt.STT:
    if STT_PROVIDER == "local":
        return proc.userdata["stt"]
    return soniox.STT(params=soniox.STTOptions(model="stt-rt-v5", language_hints=["th", "en"]))


def build_tts(proc: JobProcess) -> tts.TTS:
    if TTS_PROVIDER == "local":
        return proc.userdata["tts"]
    return soniox.TTS(
        model="tts-rt-v2",  # the plugin still defaults to the deprecated tts-rt-v1-preview
        language="th",
        voice=os.getenv("SONIOX_TTS_VOICE", "Emma"),
    )


server = AgentServer(
    setup_fnc=prewarm,
    num_idle_processes=1,  # keep one process warm (with local models loaded, if used)
    initialize_process_timeout=60.0,  # loading the local models takes a few seconds
    job_memory_warn_mb=2500,  # local STT + TTS models use ~1.5 GB per process
)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=build_stt(ctx.proc),
        # Gemini via LiveKit Inference: billed to LiveKit Cloud credits, no Google key needed
        llm=inference.LLM(
            model=os.getenv("LLM_MODEL", "google/gemini-3-flash-preview"),
            extra_kwargs={"reasoning_effort": "minimal"},  # least thinking = fastest first token
        ),
        tts=build_tts(ctx.proc),
        vad=ctx.proc.userdata["vad"],
        turn_handling={
            # audio-based end-of-turn model, runs locally on CPU.
            # Thai isn't in its trained language list, so it uses the English threshold.
            "turn_detection": inference.TurnDetector(version="v1-mini"),
        },
    )
    await session.start(agent=Assistant(), room=ctx.room)

    # shop ambience under the call, plus keyboard typing while the agent is thinking.
    # Not played in `console` mode; test with the web demo or a phone call.
    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.CROWDED_ROOM, volume=0.3),
        thinking_sound=AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.5),
    )
    await background_audio.start(room=ctx.room, agent_session=session)

    await session.generate_reply(instructions=GREETING_INSTRUCTIONS)


if __name__ == "__main__":
    cli.run_app(server)
