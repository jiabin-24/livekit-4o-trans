"""LiveKit console voice agent with Azure OpenAI realtime STT."""

import logging
import os
import time
from typing import Any

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


def _fmt_ms(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds * 1000:.0f}ms"


def _emit_turn_metrics(user_metrics: dict[str, Any] | None, assistant_metrics: dict[str, Any]) -> None:
    end_of_turn = user_metrics.get("end_of_turn_delay") if user_metrics else None
    turn_cb = user_metrics.get("on_user_turn_completed_delay") if user_metrics else None
    llm_ttft = assistant_metrics.get("llm_node_ttft")
    tts_ttfb = assistant_metrics.get("tts_node_ttfb")
    e2e = assistant_metrics.get("e2e_latency")

    e2e_text = _fmt_ms(e2e)
    if e2e is None and llm_ttft is not None and tts_ttfb is not None:
        # Fallback approximation when e2e is absent (usually missing speech stop/start anchors).
        estimated_e2e = (end_of_turn or 0.0) + llm_ttft + tts_ttfb
        e2e_text = f"{_fmt_ms(estimated_e2e)}~"

    logger.info(
        "turn_metrics end_of_turn=%s turn_completed_cb=%s llm_ttft=%s tts_ttfb=%s e2e=%s",
        _fmt_ms(end_of_turn),
        _fmt_ms(turn_cb),
        _fmt_ms(llm_ttft),
        _fmt_ms(tts_ttfb),
        e2e_text,
    )


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
    log_turn_metrics = os.getenv("LOG_TURN_METRICS", "1").strip().lower() in {"1", "true", "yes", "on"}

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=build_stt(),
        llm=build_llm(),
        tts=build_tts(),
    )

    last_user_metrics: dict[str, Any] | None = None

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event):
        nonlocal last_user_metrics
        item = event.item
        if getattr(item, "role", None) == "user":
            last_user_metrics = item.metrics if getattr(item, "metrics", None) else None
            return

        if not log_turn_metrics:
            return

        if getattr(item, "role", None) == "assistant" and getattr(item, "metrics", None):
            _emit_turn_metrics(last_user_metrics, item.metrics)
            last_user_metrics = None

    @session.on("agent_state_changed")
    def _on_agent_state_changed(_event):
        if not log_turn_metrics:
            return

        early_metrics = getattr(session, "_early_assistant_metrics", None)
        if early_metrics:
            _emit_turn_metrics(last_user_metrics, early_metrics)
            session._early_assistant_metrics = None

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
