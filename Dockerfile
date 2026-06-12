FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime libraries for audio-related deps (sounddevice/portaudio) and TLS certs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libportaudio2 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Pre-download VAD model at build time to reduce first-run latency in containers.
RUN python agent.py download-files

# Agent-only container. Override with: docker run ... python agent.py console
CMD ["python", "agent.py", "dev"]
