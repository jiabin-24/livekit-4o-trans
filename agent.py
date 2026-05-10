"""
LiveKit Voice Agent: Cascade Architecture (本地 console 模式)

Pipeline:
  STT: Azure OpenAI gpt-4o-transcribe
  LLM: Azure OpenAI (chat completions, e.g. gpt-4o / gpt-4o-mini)
  TTS: Azure Speech Service
  VAD: Silero

本地运行:
  python agent.py console

通过麦克风输入，扬声器输出，无需 LiveKit 服务器。
"""

import logging
import os
import time
import asyncio
import json
import urllib.error
import urllib.request
from types import MethodType
from urllib.parse import urlencode

import aiohttp

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, azure, silero

load_dotenv(override=True)
logger = logging.getLogger("4o-transcribe-agent")
logging.basicConfig(level=logging.INFO)

# OpenAI SDK treats an empty AZURE_OPENAI_API_KEY as an explicit api_key and may
# reject azure_ad_token auth. Remove blank keys from process env.
if (os.getenv("AZURE_OPENAI_API_KEY") or "").strip() == "":
    os.environ.pop("AZURE_OPENAI_API_KEY", None)

AZURE_CREDENTIAL_SCOPE = "https://cognitiveservices.azure.com/.default"


class EntraTokenManager:
    """Manages Entra ID tokens with automatic refresh."""

    def __init__(self, scope: str = AZURE_CREDENTIAL_SCOPE):
        self._credential = DefaultAzureCredential()
        self._scope = scope
        self._token = None

    def get_token(self) -> str:
        if self._token is None or self._token.expires_on - time.time() < 300:
            self._token = self._credential.get_token(self._scope)
            logger.info("Entra ID token refreshed, expires at %s", self._token.expires_on)
        return self._token.token


token_manager = EntraTokenManager()


class SpeechTokenManager:
    """Exchange Entra token for Azure Speech STS token (short-lived)."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_on: float = 0

    def get_token(self) -> str:
        # Speech STS token is short-lived; refresh early.
        if self._token is None or self._expires_on - time.time() < 120:
            self._token = self._refresh_token()
            self._expires_on = time.time() + 8 * 60
        return self._token

    def _refresh_token(self) -> str:
        resource_endpoint = _speech_resource_endpoint()
        if resource_endpoint is None:
            raise ValueError(
                "AZURE_SPEECH_RESOURCE_ENDPOINT is required for Entra-based Speech auth "
                "(example: https://<resource-name>.cognitiveservices.azure.com)"
            )

        aad_token = token_manager.get_token()
        sts_url = f"{resource_endpoint.rstrip('/')}/sts/v1.0/issueToken"
        req = urllib.request.Request(
            sts_url,
            method="POST",
            headers={
                "Authorization": f"Bearer {aad_token}",
                "Content-Length": "0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                token = resp.read().decode("utf-8").strip()
                if not token:
                    raise RuntimeError("Speech STS returned empty token")
                logger.info("Speech STS token refreshed")
                return token
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Speech STS token exchange failed: {e.code} {body}") from e


speech_token_manager = SpeechTokenManager()


def _env(name: str) -> str | None:
    """Return stripped env value, treating empty strings as unset."""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _use_key_auth() -> bool:
    return _env("AZURE_OPENAI_API_KEY") is not None


def _speech_resource_endpoint() -> str | None:
    """Return Speech custom subdomain endpoint used for STS token exchange."""
    endpoint = _env("AZURE_SPEECH_RESOURCE_ENDPOINT") or _env("AZURE_SPEECH_ENDPOINT")
    if endpoint is None:
        return None
    # If a full TTS path is provided, normalize back to resource root.
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/cognitiveservices/v1"):
        endpoint = endpoint[: -len("/cognitiveservices/v1")]
    return endpoint


def _patch_openai_azure_realtime_ws(stt_instance: openai.STT) -> openai.STT:
    """Patch LiveKit OpenAI STT realtime WS URL/header for Azure compatibility."""

    async def _connect_ws_patched(self, timeout: float):
        prompt = self._opts.prompt if self._opts.prompt is not None else ""
        transcription_config: dict[str, object] = {
            "model": self._opts.model,
        }
        if prompt:
            transcription_config["prompt"] = prompt
        if self._opts.language:
            transcription_config["language"] = self._opts.language.language

        input_config: dict[str, object] = {
            "format": {
                "type": "audio/pcm",
                "rate": 24000,
            },
            "transcription": transcription_config,
            "turn_detection": self._opts.turn_detection,
        }
        if self._opts.noise_reduction_type:
            input_config["noise_reduction"] = {"type": self._opts.noise_reduction_type}

        turn_detection = dict(self._opts.turn_detection or {})
        turn_detection.setdefault("type", "server_vad")
        # STT pipeline only needs user transcript events, not model auto-response events.
        turn_detection["create_response"] = False
        turn_detection["interrupt_response"] = False

        input_config["turn_detection"] = turn_detection

        realtime_config: dict[str, object] = {
            "type": "session.update",
            "session": {
                # GA-style fields
                "input_audio_format": "pcm16",
                "input_audio_transcription": transcription_config,
                "turn_detection": turn_detection,
                # Compatibility fields used by some transcription flows
                "type": "transcription",
                "audio": {"input": input_config},
            },
        }

        endpoint = _env("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_STT_DEPLOYMENT", "gpt-4o-transcribe")
        if endpoint is None:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required")

        # GA realtime route: /openai/v1/realtime?model=<deployment>
        qs = urlencode({"model": deployment})
        url = f"{endpoint.rstrip('/')}/openai/v1/realtime?{qs}"
        if url.startswith("http"):
            url = url.replace("http", "ws", 1)
        logger.info("Patched STT realtime WS URL: %s", url)

        headers = {"User-Agent": "LiveKit Agents"}
        api_key = _env("AZURE_OPENAI_API_KEY")
        if api_key:
            headers["api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {token_manager.get_token()}"

        session = self._ensure_session()
        ws = await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)
        logger.info("Patched STT realtime WS connected")
        await ws.send_json(realtime_config)
        logger.info("Patched STT realtime session.update sent")

        # Hook receive() to expose realtime server events and force commit on speech stop.
        original_receive = ws.receive

        async def _receive_with_commit(*args, **kwargs):
            msg = await original_receive(*args, **kwargs)
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    event = json.loads(msg.data)
                    event_type = event.get("type", "")
                    if event_type:
                        logger.debug("STT WS event: %s", event_type)
                    if event_type == "error":
                        logger.error("STT WS error event: %s", event)
                    if event_type == "input_audio_buffer.speech_stopped":
                        await ws.send_json({"type": "input_audio_buffer.commit"})
                        logger.info("Patched STT sent input_audio_buffer.commit")

                    # GA realtime emits transcript events under response.output_audio_transcript.*
                    # but livekit.plugins.openai.STT expects conversation.item.input_audio_transcription.*.
                    if event_type == "response.output_audio_transcript.delta":
                        mapped = {
                            "type": "conversation.item.input_audio_transcription.delta",
                            "item_id": event.get("item_id", ""),
                            "delta": event.get("delta", ""),
                        }
                        return aiohttp.WSMessage(msg.type, json.dumps(mapped), msg.extra)

                    if event_type == "response.output_audio_transcript.done":
                        mapped = {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": event.get("item_id", ""),
                            "transcript": event.get("transcript", ""),
                        }
                        return aiohttp.WSMessage(msg.type, json.dumps(mapped), msg.extra)
                except Exception:
                    logger.exception("Failed to parse STT WS event")
            return msg

        ws.receive = _receive_with_commit  # type: ignore[assignment]
        return ws

    patched_connect = MethodType(_connect_ws_patched, stt_instance)
    stt_instance._connect_ws = patched_connect
    # STT constructor already captured original callback into pool; replace it too.
    stt_instance._pool._connect_cb = patched_connect
    logger.info(
        "STT capabilities: streaming=%s interim_results=%s",
        stt_instance.capabilities.streaming,
        stt_instance.capabilities.interim_results,
    )
    logger.info("Applied local Azure realtime WS patch for openai.STT.with_azure")
    return stt_instance


def build_stt():
    """Azure OpenAI gpt-4o-transcribe STT (forced realtime/WebSocket)."""
    logger.info("build_stt called")
    use_realtime = True
    endpoint = _env("AZURE_OPENAI_ENDPOINT")

    common = dict(
        model=os.getenv("AZURE_OPENAI_STT_DEPLOYMENT", "gpt-4o-transcribe"),
        azure_deployment=os.getenv("AZURE_OPENAI_STT_DEPLOYMENT", "gpt-4o-transcribe"),
        azure_endpoint=endpoint,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        language=os.getenv("STT_LANGUAGE", "zh"),
    )
    if _use_key_auth():
        stt_instance = openai.STT.with_azure(
            api_key=_env("AZURE_OPENAI_API_KEY"),
            use_realtime=use_realtime,
            **common,
        )
        return _patch_openai_azure_realtime_ws(stt_instance)
    stt_instance = openai.STT.with_azure(
        azure_ad_token=token_manager.get_token(),
        use_realtime=use_realtime,
        **common,
    )
    return _patch_openai_azure_realtime_ws(stt_instance)


def build_speech_stt():
    """Azure Speech 流式 STT (真正的实时 partial)."""
    return azure.STT(
        speech_key=os.getenv("AZURE_SPEECH_KEY"),
        speech_region=os.getenv("AZURE_SPEECH_REGION"),
        language=os.getenv("STT_LANGUAGE", "zh-CN"),
    )


def build_llm():
    """Azure OpenAI Chat LLM."""
    common = dict(
        model=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4o-mini"),
        azure_deployment=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4o-mini"),
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )
    if _use_key_auth():
        return openai.LLM.with_azure(api_key=_env("AZURE_OPENAI_API_KEY"), **common)
    return openai.LLM.with_azure(azure_ad_token=token_manager.get_token(), **common)


def build_tts():
    """Azure Speech TTS."""
    speech_region = _env("AZURE_SPEECH_REGION")
    voice = os.getenv("AZURE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")

    # speech_key = _env("AZURE_SPEECH_KEY")
    # if speech_key:
    #     return azure.TTS(
    #         speech_key=speech_key,
    #         speech_region=speech_region,
    #         voice=voice,
    #     )
    # Entra ID auth
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
                "请保持礼貌、简洁、专业。客户说什么语言，客服就要用什么语言回复"
                "如果用户的问题你无法回答，请礼貌地告知并建议转接人工客服。"
            ),
        )

    async def on_enter(self) -> None:
        if self._greeted:
            return
        self._greeted = True
        self.session.generate_reply(
            instructions="用中文跟用户打招呼，简短地介绍自己是智能客服助手，询问有什么可以帮助的。"
        )


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=build_stt(),
        llm=build_llm(),
        tts=build_tts(),
    )

    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # console 模式下不会真正连接 LiveKit server, 但 WorkerOptions 仍需要一个占位 url
            ws_url=os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
            api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
            api_secret=os.getenv("LIVEKIT_API_SECRET", "devsecret_at_least_32_chars_long_xx"),
        )
    )
