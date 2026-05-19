import asyncio
from urllib.parse import urlencode

from livekit.plugins import openai


class RealtimeApiKeyAzureSTT(openai.STT):
    """最小补丁：Azure Realtime 场景下改用 api-key 头进行 WS 鉴权。"""

    async def _connect_ws(self, timeout: float):
        prompt = self._opts.prompt if self._opts.prompt else ""
        transcription_config = {"model": self._opts.model}
        if prompt:
            transcription_config["prompt"] = prompt
        if self._opts.language:
            transcription_config["language"] = self._opts.language.language

        input_config = {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": transcription_config,
            "turn_detection": self._opts.turn_detection,
        }
        if self._opts.noise_reduction_type:
            input_config["noise_reduction"] = {"type": self._opts.noise_reduction_type}

        realtime_config = {
            "type": "session.update",
            "session": {"type": "transcription", "audio": {"input": input_config}},
        }

        # 告知服务端当前会话是实时转写意图。
        query_params = {"intent": "transcription"}
        # 由 HTTP 地址组装出 Realtime WS 地址。
        url = f"{str(self._client.base_url).rstrip('/')}/realtime?{urlencode(query_params)}"
        if url.startswith("http"):
            url = url.replace("http", "ws", 1)

        # 关键差异：Azure key 认证需要 api-key 头，而非 Bearer token。
        headers = {
            "User-Agent": "LiveKit Agents",
            "api-key": self._client.api_key,
        }

        session = self._ensure_session()
        ws = await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)
        await ws.send_json(realtime_config)
        return ws