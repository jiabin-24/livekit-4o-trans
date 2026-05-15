from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlencode

import aiohttp
import openai
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins import openai as livekit_openai
from openai.types.beta.realtime.transcription_session_update_param import SessionTurnDetection

AzureADTokenProvider = Callable[[], str | Awaitable[str]]


class GPT4oTranscribeStreamSTT(livekit_openai.STT):
    """Azure OpenAI gpt-4o-transcribe realtime STT with Azure-specific WS wiring."""

    def __init__(
        self,
        *,
        language: str = "en",
        detect_language: bool = False,
        model: str = "gpt-4o-transcribe",
        prompt: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[str] = NOT_GIVEN,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AzureADTokenProvider | None = None,
    ) -> None:
        # AsyncAzureOpenAI enforces that one credential field is present at init time.
        # For AAD-only flows, we pass a harmless placeholder key and use bearer token at WS connect.
        client_api_key = api_key
        if client_api_key is None and (azure_ad_token is not None or azure_ad_token_provider is not None):
            client_api_key = "aad-token-auth-placeholder"

        azure_client = openai.AsyncAzureOpenAI(
            max_retries=0,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=client_api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
        )  # type: ignore[arg-type]

        super().__init__(
            language=language,
            detect_language=detect_language,
            model=model,
            prompt=prompt,
            turn_detection=turn_detection,
            noise_reduction_type=noise_reduction_type,
            client=azure_client,
            use_realtime=True,
        )

        self._azure_endpoint = azure_endpoint.rstrip("/")
        self._azure_deployment = azure_deployment
        self._api_version = api_version
        self._api_key = api_key
        self._azure_ad_token = azure_ad_token
        self._azure_ad_token_provider = azure_ad_token_provider

    @staticmethod
    def with_azure(
        *,
        language: str = "en",
        detect_language: bool = False,
        model: str = "gpt-4o-transcribe",
        prompt: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[str] = NOT_GIVEN,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AzureADTokenProvider | None = None,
    ) -> "GPT4oTranscribeStreamSTT":
        azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment = azure_deployment or os.environ.get(
            "AZURE_OPENAI_STT_DEPLOYMENT", "gpt-4o-transcribe"
        )
        api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required")
        if not azure_deployment:
            raise ValueError("AZURE_OPENAI_STT_DEPLOYMENT is required")

        return GPT4oTranscribeStreamSTT(
            language=language,
            detect_language=detect_language,
            model=model,
            prompt=prompt,
            turn_detection=turn_detection,
            noise_reduction_type=noise_reduction_type,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
        )

    async def _resolve_azure_ad_token(self) -> str | None:
        if self._azure_ad_token_provider is not None:
            token_or_awaitable = self._azure_ad_token_provider()
            if inspect.isawaitable(token_or_awaitable):
                return str(await token_or_awaitable)
            return str(token_or_awaitable)

        if self._azure_ad_token:
            return self._azure_ad_token

        if is_given(getattr(self._client, "_azure_ad_token_provider", NOT_GIVEN)):
            maybe_token_provider = getattr(self._client, "_azure_ad_token_provider")
            token_or_awaitable = maybe_token_provider()
            if inspect.isawaitable(token_or_awaitable):
                return str(await token_or_awaitable)
            return str(token_or_awaitable)

        if is_given(getattr(self._client, "_get_azure_ad_token", NOT_GIVEN)):
            token_or_awaitable = self._client._get_azure_ad_token()  # type: ignore[attr-defined]
            if inspect.isawaitable(token_or_awaitable):
                return str(await token_or_awaitable)
            return str(token_or_awaitable)

        return None

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        prompt = self._opts.prompt if is_given(self._opts.prompt) else ""
        transcription_config: dict[str, Any] = {
            "model": self._opts.model,
        }
        if prompt:
            transcription_config["prompt"] = prompt
        if self._opts.language:
            transcription_config["language"] = self._opts.language.language

        input_config: dict[str, Any] = {
            "format": {
                "type": "audio/pcm",
                "rate": 24000,
            },
            "transcription": transcription_config,
            "turn_detection": self._opts.turn_detection,
        }

        if self._opts.noise_reduction_type:
            input_config["noise_reduction"] = {"type": self._opts.noise_reduction_type}

        realtime_config: dict[str, Any] = {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": input_config,
                },
            },
        }

        query_params: dict[str, str] = {
            "intent": "transcription",
            "api-version": self._api_version,
        }
        if self._azure_deployment:
            query_params["deployment"] = self._azure_deployment

        ws_url = f"{self._azure_endpoint}/openai/realtime?{urlencode(query_params)}"
        if ws_url.startswith("https://"):
            ws_url = "wss://" + ws_url[len("https://") :]
        elif ws_url.startswith("http://"):
            ws_url = "ws://" + ws_url[len("http://") :]

        headers = {
            "User-Agent": "LiveKit Agents",
        }

        aad_token = await self._resolve_azure_ad_token()
        if aad_token:
            headers["Authorization"] = f"Bearer {aad_token}"
        elif self._api_key:
            headers["api-key"] = self._api_key
        elif self._client.api_key:
            headers["api-key"] = str(self._client.api_key)

        session = self._ensure_session()
        ws = await asyncio.wait_for(session.ws_connect(ws_url, headers=headers), timeout)
        await ws.send_json(realtime_config)
        return ws
