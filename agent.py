"""LiveKit console voice agent with Azure OpenAI realtime STT."""

import logging
import os
import time

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, azure, silero

from azure_speech_sts_from_entra import AzureSpeechStsFromEntraTokenManager

load_dotenv(override=True)
logger = logging.getLogger("4o-transcribe-agent")
logging.basicConfig(level=logging.INFO)
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

AZURE_CREDENTIAL_SCOPE = "https://cognitiveservices.azure.com/.default"


class EntraTokenManager:
    """Manages Entra ID tokens with automatic refresh."""

    def __init__(self, scope: str = AZURE_CREDENTIAL_SCOPE):
        # Prefer fast local auth chain (Azure CLI) to avoid long IMDS/shared-cache probing delays.
        self._credential = DefaultAzureCredential(
            exclude_managed_identity_credential=True,
            exclude_shared_token_cache_credential=True,
            exclude_visual_studio_code_credential=True,
        )
        self._scope = scope
        self._token = None

    def get_token(self) -> str:
        if self._token is None or self._token.expires_on - time.time() < 300:
            self._token = self._credential.get_token(self._scope)
            logger.info("Entra ID token refreshed, expires at %s", self._token.expires_on)
        return self._token.token


token_manager = EntraTokenManager()


speech_sts_token_manager = AzureSpeechStsFromEntraTokenManager(token_manager.get_token)


class RefreshingAzureSpeechTTS(azure.TTS):
    """Refreshes Speech STS token before each synthesize call."""

    def __init__(self, token_provider, **kwargs):
        self._token_provider = token_provider
        super().__init__(speech_auth_token=self._token_provider(), **kwargs)

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        self._opts.auth_token = self._token_provider()
        return super().synthesize(text, conn_options=conn_options)


def _use_key_auth() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY"))


def _openai_v1_base_url() -> str:
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is required")
    return f"{endpoint}/openai/v1"


def build_stt():
    """Realtime STT using OpenAI v1 endpoint style, compatible with Foundry speech resources."""
    model = os.getenv("AZURE_OPENAI_STT_DEPLOYMENT", "gpt-realtime-whisper")
    base_url = _openai_v1_base_url()
    api_token = os.getenv("AZURE_OPENAI_API_KEY") if _use_key_auth() else token_manager.get_token()

    logger.info("STT mode: realtime, model=%s, base_url=%s", model, base_url)
    return openai.STT(
        model=model,
        base_url=base_url,
        api_key=api_token,
        language=os.getenv("STT_LANGUAGE", "zh"),
        use_realtime=True,
        turn_detection={
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
        },
    )


def build_llm():
    """LLM on OpenAI v1 endpoint style to match the configured Foundry resource."""
    model = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4o-mini")
    base_url = _openai_v1_base_url()
    api_token = os.getenv("AZURE_OPENAI_API_KEY") if _use_key_auth() else token_manager.get_token()

    logger.info("LLM mode: v1, model=%s, base_url=%s", model, base_url)
    return openai.LLM(
        model=model,
        base_url=base_url,
        api_key=api_token,
    )


def build_tts():
    """Azure Speech TTS."""
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    voice = os.getenv("AZURE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")

    # Exchange Entra ID token for Speech STS token, then use STS token for Speech auth.
    return RefreshingAzureSpeechTTS(
        token_provider=speech_sts_token_manager.get_token,
        speech_region=speech_region,
        voice=voice,
    )


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "你是智能客服助手。"
                "回复简洁、礼貌、专业；与用户使用同一语言。"
                "无法回答时明确说明，并建议转人工。"
            ),
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="用中文简短问候并询问需要什么帮助。"
        )


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=build_stt(),
        llm=build_llm(),
        tts=build_tts(),
    )

    @session.on("error")
    def _on_session_error(event):
        logger.exception("[SESSION_ERROR] %s", event.error)

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
