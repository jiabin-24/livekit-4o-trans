import logging
import os
import time
from collections.abc import Callable

import httpx

logger = logging.getLogger("4o-transcribe-agent")


class AzureSpeechStsFromEntraTokenManager:
    """Exchanges Entra ID token for Azure Speech STS token with automatic refresh."""

    def __init__(self, entra_token_provider: Callable[[], str]) -> None:
        self._entra_token_provider = entra_token_provider
        self._token: str | None = None
        self._expires_at: float = 0.0

    def get_token(self) -> str:
        speech_resource_endpoint = os.getenv("AZURE_SPEECH_RESOURCE_ENDPOINT")
        if not speech_resource_endpoint:
            raise ValueError("AZURE_SPEECH_RESOURCE_ENDPOINT is required for Speech STS token exchange")

        now = time.time()
        if self._token is None or now > self._expires_at - 60:
            entra_token = self._entra_token_provider()
            sts_url = f"{speech_resource_endpoint.rstrip('/')}/sts/v1.0/issueToken"
            resp = httpx.post(
                sts_url,
                headers={"Authorization": f"Bearer {entra_token}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            self._token = resp.text
            # STS token is typically valid for 10 minutes.
            self._expires_at = now + 10 * 60
            logger.info("Azure Speech STS token refreshed via Entra, expires at %s", int(self._expires_at))

        return self._token
