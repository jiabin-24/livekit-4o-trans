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

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, azure, silero

from azure_speech_sts_from_entra import AzureSpeechStsFromEntraTokenManager
from gpt4o_transcribe_stream_stt import GPT4oTranscribeStreamSTT

load_dotenv(override=True)
logger = logging.getLogger("4o-transcribe-agent")
logging.basicConfig(level=logging.INFO)

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


speech_sts_token_manager = AzureSpeechStsFromEntraTokenManager(token_manager.get_token)


def _use_key_auth() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY"))


def build_stt():
    """Azure OpenAI gpt-4o-transcribe STT (Realtime/流式)."""
    common = dict(
        model=os.getenv("AZURE_OPENAI_STT_DEPLOYMENT", "gpt-4o-transcribe"),
        azure_deployment=os.getenv("AZURE_OPENAI_STT_DEPLOYMENT", "gpt-4o-transcribe"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        language=os.getenv("STT_LANGUAGE", "zh"),
    )
    if _use_key_auth():
        return GPT4oTranscribeStreamSTT.with_azure(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            **common,
        )
    return GPT4oTranscribeStreamSTT.with_azure(
        azure_ad_token_provider=token_manager.get_token,
        **common,
    )


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
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )
    if _use_key_auth():
        return openai.LLM.with_azure(api_key=os.getenv("AZURE_OPENAI_API_KEY"), **common)
    return openai.LLM.with_azure(
        api_key="aad-token-auth-placeholder",
        azure_ad_token_provider=token_manager.get_token,
        **common,
    )


def build_tts():
    """Azure Speech TTS."""
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    voice = os.getenv("AZURE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")

    # Exchange Entra ID token for Speech STS token, then use STS token for Speech auth.
    return azure.TTS(
        speech_auth_token=speech_sts_token_manager.get_token(),
        speech_region=speech_region,
        voice=voice,
    )


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "你是一个专业的智能客服助手。"
                "请保持礼貌、简洁、专业。客户说什么语言，客服就要用什么语言回复"
                "如果用户的问题你无法回答，请礼貌地告知并建议转接人工客服。"
            ),
        )

    async def on_enter(self) -> None:
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
