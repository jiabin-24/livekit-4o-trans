"""Reproduce the exact OpenAI SDK call livekit makes for STT."""
import os, asyncio, wave, struct, math
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncAzureOpenAI

# build a 1-second 16kHz silent wav in memory
import io
buf = io.BytesIO()
with wave.open(buf, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"\x00\x00" * 16000)
data = buf.getvalue()

async def main():
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4o-transcribe",
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    print("endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))
    print("api_version:", os.getenv("AZURE_OPENAI_API_VERSION"))
    print("key starts:", (os.getenv("AZURE_OPENAI_API_KEY") or "")[:6])
    try:
        resp = await client.audio.transcriptions.create(
            file=("file.wav", data, "audio/wav"),
            model="gpt-4o-transcribe",
            language="zh",
            response_format="json",
        )
        print("OK:", resp)
    except Exception as e:
        print("ERR:", type(e).__name__, e)

asyncio.run(main())
