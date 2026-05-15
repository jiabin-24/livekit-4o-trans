from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

CHUNK_MS = 200

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal mic->realtime transcription sample (delta + final)"
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("OPENAI_ENDPOINT", ""),
        help="Azure endpoint, e.g. https://xxx.cognitiveservices.azure.com",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="Azure API key",
    )
    parser.add_argument(
        "--deployment",
        default=os.getenv("REALTIME_DEPLOYMENT", ""),
        help="Realtime deployment name",
    )
    parser.add_argument(
        "--transcription-model",
        default=os.getenv("REALTIME_TRANSCRIPTION_MODEL", "") or os.getenv("REALTIME_DEPLOYMENT", ""),
        help="Transcription model/deployment used by session.audio.input.transcription.model",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("STT_LANGUAGE", ""),
        help="Optional language hint, e.g. zh, en, fil",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.getenv("MIC_SAMPLE_RATE", "24000") or 24000),
        help="Microphone sample rate",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=int(os.getenv("MIC_MAX_SECONDS", "0") or 0),
        help="0 means run until Ctrl+C",
    )
    parser.add_argument(
        "--no-deltas",
        action="store_true",
        help="Do not print partial delta text",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory for generated latency reports",
    )
    parser.add_argument(
        "--report-prefix",
        default="asr_latency",
        help="Prefix of report files",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=int(os.getenv("REALTIME_SILENCE_MS", "500") or 500),
        help="Server VAD silence_duration_ms (how long of silence triggers speech_stopped). Default 500.",
    )
    args = parser.parse_args()

    if not args.endpoint:
        parser.error("--endpoint required or set OPENAI_ENDPOINT in .env")
    if not args.api_key:
        parser.error("--api-key required or set OPENAI_API_KEY in .env")
    if not args.deployment:
        parser.error("--deployment required or set REALTIME_DEPLOYMENT in .env")
    if not args.transcription_model:
        parser.error(
            "--transcription-model required or set REALTIME_TRANSCRIPTION_MODEL/REALTIME_DEPLOYMENT in .env"
        )
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be > 0")
    if args.max_seconds < 0:
        parser.error("--max-seconds must be >= 0")
    if args.silence_ms < 0:
        parser.error("--silence-ms must be >= 0")

    return args


def build_ws_url(endpoint: str, deployment: str) -> str:
    base = endpoint.rstrip("/")
    url = f"{base}/openai/v1/realtime?intent=transcription&deployment={deployment}"
    return url.replace("https://", "wss://").replace("http://", "ws://")


def build_session_update(transcription_model: str, language: str, silence_ms: int) -> dict[str, Any]:
    input_config: dict[str, Any] = {
        "transcription": {"model": transcription_model},
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": silence_ms,
        }
    }
    if language:
        input_config["transcription"]["language"] = language

    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": input_config
            },
        },
    }


async def send_mic_audio(
    ws: websockets.WebSocketClientProtocol,
    sample_rate: int,
    max_seconds: int,
    stop_event: asyncio.Event,
    timing: dict[str, Any],
    use_server_vad: bool,
) -> None:
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    loop = asyncio.get_running_loop()
    chunk_frames = int(sample_rate * CHUNK_MS / 1000)

    def push_chunk(chunk: bytes) -> None:
        try:
            queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            return
        pcm16 = np.clip(indata[:, 0], -1.0, 1.0)
        chunk = (pcm16 * 32767.0).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(push_chunk, chunk)

    started_at = time.time()
    run_forever = max_seconds == 0
    if run_forever:
        print(f"[MIC] Recording at {sample_rate} Hz until Ctrl+C...")
    else:
        print(f"[MIC] Recording for up to {max_seconds}s at {sample_rate} Hz...")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                if not run_forever and time.time() - started_at >= max_seconds:
                    break
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
                now = time.perf_counter()
                if timing.get("first_audio_sent_at") is None:
                    timing["first_audio_sent_at"] = now
                timing["last_audio_sent_at"] = now
                timing["audio_chunks_sent"] = timing.get("audio_chunks_sent", 0) + 1
                timing["pending_audio_since_commit"] = True
    finally:
        # When using server VAD, server auto-commits on speech_stopped.
        # Manual commit on shutdown almost always triggers commit_empty; only commit in non-VAD mode.
        if (not use_server_vad) and timing.get("pending_audio_since_commit"):
            try:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                print("[MIC] Audio committed")
                timing["pending_audio_since_commit"] = False
            except Exception:
                pass


async def receive_events(
    ws: websockets.WebSocketClientProtocol,
    show_deltas: bool,
    stop_event: asyncio.Event,
    timing: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "delta_count": 0,
        "final_count": 0,
        "last_final": "",
        "all_finals": [],
        "event_types": {},
        # Per-utterance records, each: {started, stopped, first_delta_at, completed_at, delta_from_started_ms, delta_from_stopped_ms, final_latency_ms, speech_duration_ms, text}
        "utterances": [],
        "errors": [],
    }
    # Index of the current utterance being captured for first-delta timing.
    current_utt_idx: list[int] = [-1]

    while not stop_event.is_set():
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except websockets.exceptions.ConnectionClosed:
            break

        event = json.loads(raw)
        event_type = event.get("type", "unknown")
        result["event_types"][event_type] = result["event_types"].get(event_type, 0) + 1

        if event_type == "conversation.item.input_audio_transcription.delta":
            delta = event.get("delta", "")
            if delta:
                result["delta_count"] += 1
                now = time.perf_counter()
                if timing.get("first_delta_at") is None:
                    timing["first_delta_at"] = now
                # Record first delta of the currently-active utterance (if any).
                if result["utterances"]:
                    utt = result["utterances"][-1]
                    if utt.get("first_delta_at") is None and utt.get("completed_at") is None:
                        utt["first_delta_at"] = now
                if show_deltas:
                    print(delta, end="", flush=True)

        elif event_type == "conversation.item.input_audio_transcription.completed":
            final_text = event.get("transcript") or event.get("text") or ""
            result["final_count"] += 1
            result["last_final"] = final_text
            result["all_finals"].append(final_text)
            now = time.perf_counter()
            if timing.get("first_final_at") is None:
                timing["first_final_at"] = now
            # Match this final with the oldest open utterance (FIFO).
            target_utt = None
            for utt in result["utterances"]:
                if utt.get("completed_at") is None:
                    target_utt = utt
                    break
            if target_utt is not None:
                target_utt["completed_at"] = now
                target_utt["text"] = final_text
                started = target_utt.get("started_at")
                stopped = target_utt.get("stopped_at")
                first_d = target_utt.get("first_delta_at")
                if started is not None and stopped is not None:
                    target_utt["speech_duration_ms"] = round((stopped - started) * 1000, 2)
                if stopped is not None:
                    target_utt["final_latency_ms"] = round((now - stopped) * 1000, 2)
                if first_d is not None and started is not None:
                    target_utt["delta_from_started_ms"] = round((first_d - started) * 1000, 2)
                if first_d is not None and stopped is not None:
                    target_utt["delta_from_stopped_ms"] = round((first_d - stopped) * 1000, 2)
            if show_deltas:
                print()
            print("\n[FINAL]", final_text)

        elif event_type == "error":
            err = event.get("error", {})
            result["errors"].append(
                {
                    "code": err.get("code"),
                    "message": err.get("message"),
                }
            )
            print("\n[ERROR]", json.dumps(event, ensure_ascii=False))

        elif event_type == "conversation.item.input_audio_transcription.failed":
            failure = event.get("error") or event.get("failure") or {}
            result["errors"].append(
                {
                    "code": failure.get("code") or "transcription_failed",
                    "message": failure.get("message") or json.dumps(event, ensure_ascii=False),
                }
            )
            print("\n[TRANSCRIPTION FAILED]", json.dumps(event, ensure_ascii=False))

        elif event_type == "input_audio_buffer.speech_started":
            now = time.perf_counter()
            timing.setdefault("speech_started_times", []).append(now)
            result["utterances"].append({
                "started_at": now,
                "stopped_at": None,
                "first_delta_at": None,
                "completed_at": None,
                "text": "",
            })

        elif event_type == "input_audio_buffer.speech_stopped":
            now = time.perf_counter()
            timing.setdefault("speech_stopped_times", []).append(now)
            # Attach to the latest open utterance.
            for utt in reversed(result["utterances"]):
                if utt.get("stopped_at") is None:
                    utt["stopped_at"] = now
                    break

        elif event_type == "input_audio_buffer.committed":
            timing["pending_audio_since_commit"] = False
            print(f"[EVENT] {event_type}")

        elif event_type in {"session.created", "session.updated"}:
            print(f"[EVENT] {event_type}")

    return result


def write_report_files(args: argparse.Namespace, result: dict[str, Any], timing: dict[str, Any]) -> dict[str, str]:
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.report_prefix}_{run_id}"

    utterances = [u for u in result.get("utterances", []) if u.get("completed_at") is not None]

    def _avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 2) if values else None

    final_lats = [u["final_latency_ms"] for u in utterances if "final_latency_ms" in u]
    delta_started_lats = [u["delta_from_started_ms"] for u in utterances if "delta_from_started_ms" in u]
    delta_stopped_lats = [u["delta_from_stopped_ms"] for u in utterances if "delta_from_stopped_ms" in u]
    speech_durs = [u["speech_duration_ms"] for u in utterances if "speech_duration_ms" in u]

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "deployment": args.deployment,
        "language": args.language or "",
        "sample_rate": args.sample_rate,
        "max_seconds": args.max_seconds,
        "silence_ms": args.silence_ms,
        "audio_chunks_sent": timing.get("audio_chunks_sent", 0),
        "ws_handshake_ms": timing.get("ws_handshake_ms"),
        "delta_count": result.get("delta_count", 0),
        "final_count": result.get("final_count", 0),
        "utterance_count": len(utterances),
        "avg_speech_duration_ms": _avg(speech_durs),
        "avg_delta_from_started_ms": _avg(delta_started_lats),
        "avg_delta_from_stopped_ms": _avg(delta_stopped_lats),
        "avg_final_latency_ms": _avg(final_lats),
        "last_final": result.get("last_final", ""),
        "all_finals": result.get("all_finals", []),
        "event_types": result.get("event_types", {}),
        "utterances": utterances,
        "errors": result.get("errors", []),
    }

    json_path = report_dir / f"{base_name}.json"
    csv_path = report_dir / f"{base_name}.csv"
    md_path = report_dir / f"{base_name}.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "speech_duration_ms",
                "delta_from_started_ms",
                "delta_from_stopped_ms",
                "final_latency_ms",
                "text",
            ],
        )
        writer.writeheader()
        for idx, u in enumerate(utterances, start=1):
            writer.writerow({
                "idx": idx,
                "speech_duration_ms": u.get("speech_duration_ms"),
                "delta_from_started_ms": u.get("delta_from_started_ms"),
                "delta_from_stopped_ms": u.get("delta_from_stopped_ms"),
                "final_latency_ms": u.get("final_latency_ms"),
                "text": u.get("text", ""),
            })

    md_lines = [
        "# ASR Latency Report",
        "",
        f"- run_id: {summary['run_id']}",
        f"- timestamp: {summary['timestamp']}",
        f"- deployment: {summary['deployment']}",
        f"- language: {summary['language'] or '(none)'}",
        f"- sample_rate: {summary['sample_rate']}",
        f"- max_seconds: {summary['max_seconds']}",
        f"- silence_ms (server_vad.silence_duration_ms): {summary['silence_ms']}",
        f"- audio_chunks_sent: {summary['audio_chunks_sent']}",
        f"- ws_handshake_ms: {summary['ws_handshake_ms']}",
        f"- delta_count: {summary['delta_count']}",
        f"- final_count: {summary['final_count']}",
        f"- utterance_count: {summary['utterance_count']}",
        "",
        "## Averages (per utterance)",
        f"- avg_speech_duration_ms: {summary['avg_speech_duration_ms']}",
        f"- avg_delta_from_started_ms (speech_started -> first delta, depends on sentence length): {summary['avg_delta_from_started_ms']}",
        f"- avg_delta_from_stopped_ms (speech_stopped -> first delta; negative = streamed before you stopped): {summary['avg_delta_from_stopped_ms']}",
        f"- avg_final_latency_ms (speech_stopped -> completed): {summary['avg_final_latency_ms']}",
        "",
        "## Per-Utterance Breakdown",
        "| # | speech_dur (ms) | delta_from_started (ms) | delta_from_stopped (ms) | final_latency (ms) | text |",
        "|---|---|---|---|---|---|",
    ]
    for idx, u in enumerate(utterances, start=1):
        md_lines.append(
            f"| {idx} | {u.get('speech_duration_ms')} | {u.get('delta_from_started_ms')} | {u.get('delta_from_stopped_ms')} | {u.get('final_latency_ms')} | {u.get('text', '')} |"
        )
    md_lines.extend([
        "",
        "## Event Types",
    ])
    for event_type, count in sorted(summary["event_types"].items()):
        md_lines.append(f"- {event_type}: {count}")
    md_lines.extend(["", "## Errors"])
    if summary["errors"]:
        for err in summary["errors"]:
            md_lines.append(f"- {err.get('code')}: {err.get('message')}")
    else:
        md_lines.append("(none)")
    md_lines.extend(["", "## Last Final Transcript", summary["last_final"] or "(none)"])
    md_lines.extend(["", "## All Final Transcripts"])
    if summary["all_finals"]:
        for idx, txt in enumerate(summary["all_finals"], start=1):
            md_lines.append(f"{idx}. {txt}")
    else:
        md_lines.append("(none)")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "md": str(md_path),
    }


async def main() -> int:
    args = parse_args()

    ws_url = build_ws_url(args.endpoint, args.deployment)
    print("=== Config ===")
    print(f"endpoint: {args.endpoint}")
    print(f"deployment: {args.deployment}")
    print(f"transcription_model: {args.transcription_model}")
    print(f"language: {args.language or '(none)'}")
    print(f"sample_rate: {args.sample_rate}")
    print(f"max_seconds: {args.max_seconds}")
    print(f"silence_ms (server_vad.silence_duration_ms): {args.silence_ms}")
    print(f"ws_url: {ws_url}")

    stop_event = asyncio.Event()
    headers = {"api-key": args.api_key}
    timing: dict[str, Any] = {
        "first_audio_sent_at": None,
        "last_audio_sent_at": None,
        "first_delta_at": None,
        "first_final_at": None,
        "audio_chunks_sent": 0,
        "pending_audio_since_commit": False,
        "speech_started_times": [],
        "speech_stopped_times": [],
        "ws_handshake_ms": None,
    }

    try:
        handshake_start = time.perf_counter()
        async with websockets.connect(ws_url, additional_headers=headers, max_size=None) as ws:
            timing["ws_handshake_ms"] = round((time.perf_counter() - handshake_start) * 1000, 2)
            print(f"[NET] ws_handshake_ms: {timing['ws_handshake_ms']}")
            await ws.send(json.dumps(build_session_update(args.transcription_model, args.language, args.silence_ms)))


            receiver = asyncio.create_task(
                receive_events(
                    ws,
                    show_deltas=not args.no_deltas,
                    stop_event=stop_event,
                    timing=timing,
                )
            )
            sender = asyncio.create_task(
                send_mic_audio(
                    ws,
                    sample_rate=args.sample_rate,
                    max_seconds=args.max_seconds,
                    stop_event=stop_event,
                    timing=timing,
                    use_server_vad=True,
                )
            )

            try:
                await sender
                # Give the server a short window to send final events after commit.
                await asyncio.sleep(2)
            except KeyboardInterrupt:
                print("\n[MIC] Stopped by user")
            finally:
                stop_event.set()
                if not receiver.done():
                    try:
                        await asyncio.wait_for(receiver, timeout=3)
                    except asyncio.TimeoutError:
                        receiver.cancel()
                        try:
                            await receiver
                        except asyncio.CancelledError:
                            pass
                result = (
                    receiver.result()
                    if receiver.done() and not receiver.cancelled()
                    else {
                        "delta_count": 0,
                        "final_count": 0,
                        "last_final": "",
                        "all_finals": [],
                        "event_types": {},
                        "utterances": [],
                        "errors": [],
                    }
                )

    except KeyboardInterrupt:
        print("\n[MIC] Stopped by user")
        return 0
    except websockets.exceptions.InvalidStatus as exc:
        print(f"\n[FAIL] WebSocket handshake rejected: HTTP {exc.response.status_code}")
        return 1
    except Exception as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        return 1

    print("\n=== Summary ===")
    print(f"delta_count: {result['delta_count']}")
    print(f"final_count: {result['final_count']}")
    print(f"last_final_length: {len(result['last_final'])}")
    utts = [u for u in result.get("utterances", []) if u.get("completed_at") is not None]
    if utts:
        def _avg(key: str):
            vals = [u[key] for u in utts if key in u]
            return round(sum(vals) / len(vals), 2) if vals else None
        print(f"utterance_count: {len(utts)}")
        print(f"avg_speech_duration_ms: {_avg('speech_duration_ms')}")
        print(f"avg_delta_from_started_ms (speech_started -> first delta): {_avg('delta_from_started_ms')}")
        print(f"avg_delta_from_stopped_ms (speech_stopped -> first delta; <0 means streamed early): {_avg('delta_from_stopped_ms')}")
        print(f"avg_final_latency_ms (speech_stopped -> completed): {_avg('final_latency_ms')}")
    print(f"ws_handshake_ms: {timing.get('ws_handshake_ms')}")
    print("event_types:")
    for event_type, count in sorted(result["event_types"].items()):
        print(f"  {event_type}: {count}")

    report_paths = write_report_files(args, result, timing)
    print("\n=== Reports ===")
    print(f"json: {report_paths['json']}")
    print(f"csv: {report_paths['csv']}")
    print(f"md: {report_paths['md']}")

    if result["delta_count"] > 0:
        print("\n[PASS] Received delta events.")
    else:
        print("\n[INFO] No delta events observed.")

    if result["final_count"] > 0:
        print("[PASS] Received completed events.")
        return 0

    print("[FAIL] No completed event observed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
