"""Minimal LiveKit voice agent (console mode)."""

import logging
import os
import time
import asyncio
from types import MethodType
import urllib.error
import urllib.request
from urllib.parse import urlencode

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.voice.turn import InterruptionOptions, TurnHandlingOptions
from livekit.plugins import azure, openai, silero

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("4o-transcribe-agent")

AZURE_SCOPE = "https://cognitiveservices.azure.com/.default"
_GREETING_SENT = False

# Empty env keys should be treated as unset when using Entra auth.
if (os.getenv("AZURE_OPENAI_API_KEY") or "").strip() == "":
    os.environ.pop("AZURE_OPENAI_API_KEY", None)


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _require(name: str) -> str:
    value = _env(name)
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _normalize_stt_language(lang: str) -> str:
    value = (lang or "").strip()
    if value.lower() == "zh":
        return "zh-CN"
    return value or "zh-CN"


class EntraTokenManager:
    def __init__(self, scope: str = AZURE_SCOPE):
        self._credential = DefaultAzureCredential()
        self._scope = scope
        self._token = None

    def get_token(self) -> str:
        if self._token is None or self._token.expires_on - time.time() < 300:
            self._token = self._credential.get_token(self._scope)
            logger.info("Entra token refreshed, expires at %s", self._token.expires_on)
        return self._token.token


def _speech_resource_endpoint() -> str:
    endpoint = _env("AZURE_SPEECH_RESOURCE_ENDPOINT") or _env("AZURE_SPEECH_ENDPOINT")
    if endpoint is None:
        raise ValueError("AZURE_SPEECH_RESOURCE_ENDPOINT is required for Entra Speech auth")
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/cognitiveservices/v1"):
        endpoint = endpoint[: -len("/cognitiveservices/v1")]
    return endpoint


class SpeechTokenManager:
    def __init__(self, entra: EntraTokenManager):
        self._entra = entra
        self._token: str | None = None
        self._expires_on: float = 0.0

    def get_token(self) -> str:
        if self._token is None or self._expires_on - time.time() < 120:
            self._token = self._refresh_token()
            self._expires_on = time.time() + 8 * 60
            logger.info("Speech STS token refreshed")
        return self._token

    def _refresh_token(self) -> str:
        sts_url = f"{_speech_resource_endpoint()}/sts/v1.0/issueToken"
        req = urllib.request.Request(
            sts_url,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._entra.get_token()}",
                "Content-Length": "0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                token = resp.read().decode("utf-8").strip()
                if not token:
                    raise RuntimeError("Speech STS returned empty token")
                return token
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Speech STS token exchange failed: {e.code} {body}") from e


entra_token_manager = EntraTokenManager()
speech_token_manager = SpeechTokenManager(entra_token_manager)


def _use_key_auth() -> bool:
    return _env("AZURE_OPENAI_API_KEY") is not None


def _patch_openai_azure_realtime_ws(stt_instance: openai.STT) -> openai.STT:
    async def _connect_ws_patched(self, timeout: float):
        prompt = self._opts.prompt if getattr(self._opts, "prompt", None) else ""
        deployment = os.getenv("AZURE_OPENAI_STT_DEPLOYMENT", "gpt-4o-transcribe")
        transcription_config = {"model": deployment}
        if prompt:
            transcription_config["prompt"] = prompt

        language = getattr(self._opts, "language", None)
        if language and getattr(language, "language", None):
            transcription_config["language"] = language.language

        # Azure realtime transcription expects conversation-style session fields.
        session_payload = {
            "input_audio_format": "pcm16",
            "input_audio_transcription": transcription_config,
            "turn_detection": self._opts.turn_detection,
        }

        realtime_config = {
            "type": "session.update",
            "session": session_payload,
        }

        endpoint = _require("AZURE_OPENAI_ENDPOINT")
        query = urlencode({"model": deployment, "intent": "transcription"})
        url = f"{endpoint.rstrip('/')}/openai/v1/realtime?{query}"
        if url.startswith("http"):
            url = url.replace("http", "ws", 1)

        headers = {"User-Agent": "LiveKit Agents"}
        api_key = _env("AZURE_OPENAI_API_KEY")
        if api_key:
            headers["api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {entra_token_manager.get_token()}"

        session = self._ensure_session()
        ws = await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)
        await ws.send_json(realtime_config)
        logger.info("Patched Azure realtime WS connected: %s", url)
        return ws

    patched_connect = MethodType(_connect_ws_patched, stt_instance)
    stt_instance._connect_ws = patched_connect
    stt_instance._pool._connect_cb = patched_connect
    return stt_instance


def build_stt() -> openai.STT:
    stt_language = _normalize_stt_language(os.getenv("STT_LANGUAGE", "zh"))
    speech_region = _require("AZURE_SPEECH_REGION")
    speech_key = _env("AZURE_SPEECH_KEY")
    if speech_key:
        return azure.STT(
            speech_key=speech_key,
            speech_region=speech_region,
            language=stt_language,
        )
    return azure.STT(
        speech_auth_token=speech_token_manager.get_token(),
        speech_region=speech_region,
        language=stt_language,
    )


def build_llm() -> openai.LLM:
    common = {
        "model": os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4o-mini"),
        "azure_deployment": os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4o-mini"),
        "azure_endpoint": _require("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    }
    if _use_key_auth():
        return openai.LLM.with_azure(api_key=_require("AZURE_OPENAI_API_KEY"), **common)
    return openai.LLM.with_azure(azure_ad_token=entra_token_manager.get_token(), **common)


def build_tts() -> azure.TTS:
    speech_region = _require("AZURE_SPEECH_REGION")
    voice = os.getenv("AZURE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    speech_key = _env("AZURE_SPEECH_KEY")
    if speech_key:
        return azure.TTS(speech_key=speech_key, speech_region=speech_region, voice=voice)
    return azure.TTS(
        speech_auth_token=speech_token_manager.get_token(),
        speech_region=speech_region,
        voice=voice,
    )


class Assistant(Agent):
    def __init__(self) -> None:
        self._greeted = False
        super().__init__(
            instructions=(
                "你是一个专业的智能客服助手。"
                "请保持礼貌、简洁、专业。"
                "用户说什么语言，你就用什么语言回复。"
            )
        )

    async def on_enter(self) -> None:
        global _GREETING_SENT
        auto_greet = (_env("AUTO_GREETING") or "1").lower() not in {"0", "false", "no", "off"}
        if not auto_greet or self._greeted or _GREETING_SENT:
            return
        self._greeted = True
        _GREETING_SENT = True
        logger.info("Greeting triggered via generate_reply")
        self.session.generate_reply(
            instructions="用中文简短打招呼并询问用户需要什么帮助。"
        )


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    turn_handling = TurnHandlingOptions(
        interruption=InterruptionOptions(
            enabled=True,
            mode="vad",
            min_duration=0.25,
        ),
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=build_stt(),
        llm=build_llm(),
        tts=build_tts(),
        turn_handling=turn_handling,
        aec_warmup_duration=0.0,
    )

    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
            api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
            api_secret=os.getenv("LIVEKIT_API_SECRET", "devsecret_at_least_32_chars_long_xx"),
        )
    )
