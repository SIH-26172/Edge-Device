#!/usr/bin/env python3

"""

Bumblebee ESP32 -> WebSocket -> whisper.cpp

Keeps the supplied ESP32 .ino unchanged.

ESP32 protocol:

    START                  (text)

    PCM16 mono 16 kHz     (binary frames, normally 1024 bytes)

    END                    (text)

Server behavior:

    - Receives audio continuously over ONE WebSocket connection.

    - Shows rolling Whisper "PARTIAL" text on the Mac while the ESP32 is

      still speaking.

    - Runs a final Whisper transcription after END.

    - Sends only the final JSON result back to the existing ESP32 client,

      because the supplied ESP32 code reads server text only after END.

Whisper.cpp:

    Local official whisper-server on 127.0.0.1:8080

    Model: ~/whisper.cpp/models/ggml-base.en.bin

WebSocket:

    0.0.0.0:8765

"""

import asyncio

import http.client

import io

import json

import socket

import subprocess

import sys

import time
import wave
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

try:

    from websockets.asyncio.server import serve

except ImportError:

    from websockets.server import serve

from websockets.exceptions import ConnectionClosed

# ============================================================

# FIXED CONFIGURATION

# ============================================================

WS_HOST = "0.0.0.0"

WS_PORT = 8765

WHISPER_HOST = "127.0.0.1"

WHISPER_PORT = 8080

WHISPER_CPP_DIR = Path.home() / "whisper.cpp"
WHISPER_SERVER = WHISPER_CPP_DIR / "build" / "bin" / "whisper-server"

MODEL_DIR = WHISPER_CPP_DIR / "models"
MODEL_PATH = MODEL_DIR / "ggml-base.en.bin"

# Downloaded automatically the first time the server is started.
MODEL_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/"
    "resolve/main/ggml-base.en.bin?download=true"
)

SAMPLE_RATE = 16000

CHANNELS = 1

SAMPLE_WIDTH = 2  # PCM16

# Current ESP32 sends 1024-byte frames:

# 1024 / (16000 * 2) = 0.032 s = 32 ms.

EXPECTED_ESP32_FRAME_BYTES = 1024

# Rolling preview.

PARTIAL_INTERVAL_SEC = 0.75

PARTIAL_WINDOW_SEC = 5.0

# Do not start Whisper previews until enough audio exists.

MIN_PARTIAL_AUDIO_SEC = 0.75

# One Whisper inference at a time.

INFERENCE_LOCK = asyncio.Lock()

whisper_process = None

# ============================================================

# AUDIO HELPERS

# ============================================================

def pcm_to_wav(pcm: bytes) -> bytes:

    """Wrap raw PCM16 mono 16 kHz data in a WAV container."""

    if len(pcm) % SAMPLE_WIDTH:

        pcm = pcm[:-1]

    out = io.BytesIO()

    with wave.open(out, "wb") as wf:

        wf.setnchannels(CHANNELS)

        wf.setsampwidth(SAMPLE_WIDTH)

        wf.setframerate(SAMPLE_RATE)

        wf.writeframes(pcm)

    return out.getvalue()

def make_multipart(wav_data: bytes) -> tuple[bytes, str]:

    """Build multipart/form-data for whisper.cpp /inference."""

    boundary = "----BumblebeeWhisperBoundary9a7e"

    boundary_bytes = boundary.encode("ascii")

    body = bytearray()

    def add_field(name: str, value: str) -> None:

        body.extend(b"--" + boundary_bytes + b"\r\n")

        body.extend(

            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(

                "utf-8"

            )

        )

        body.extend(value.encode("utf-8"))

        body.extend(b"\r\n")

    add_field("temperature", "0.0")

    add_field("temperature_inc", "0.0")

    add_field("language", "en")

    add_field("response_format", "json")

    add_field("no_timestamps", "true")

    add_field("no_speech_thold", "0.6")

    body.extend(b"--" + boundary_bytes + b"\r\n")

    body.extend(

        b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'

    )

    body.extend(b"Content-Type: audio/wav\r\n\r\n")

    body.extend(wav_data)

    body.extend(b"\r\n")

    body.extend(b"--" + boundary_bytes + b"--\r\n")

    return bytes(body), f"multipart/form-data; boundary={boundary}"

# ============================================================

# WHISPER SERVER

# ============================================================

def wait_for_whisper(timeout_sec: float = 30.0) -> bool:

    deadline = time.time() + timeout_sec

    while time.time() < deadline:

        try:

            with socket.create_connection(

                (WHISPER_HOST, WHISPER_PORT),

                timeout=1.0,

            ):

                return True

        except OSError:

            time.sleep(0.25)

    return False

def ensure_whisper_model() -> None:
    """Download ggml-base.en.bin if it is not already present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.is_file():
        print(f"Whisper model already exists: {MODEL_PATH}")
        return

    print("=" * 64)
    print(" Whisper model not found")
    print(" Downloading ggml-base.en.bin...")
    print("=" * 64)

    temp_path = MODEL_PATH.with_name(MODEL_PATH.name + ".part")
    request = Request(
        MODEL_URL,
        headers={"User-Agent": "Bumblebee-ASR/1.0"},
    )

    try:
        with urlopen(request, timeout=60) as response, temp_path.open("wb") as output:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            downloaded = 0

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break

                output.write(chunk)
                downloaded += len(chunk)

                if total_bytes:
                    percent = downloaded * 100 / total_bytes
                    print(
                        f"\rDownloading: {percent:5.1f}% "
                        f"({downloaded / (1024 * 1024):.1f} MiB)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\rDownloaded: {downloaded / (1024 * 1024):.1f} MiB",
                        end="",
                        flush=True,
                    )

        temp_path.replace(MODEL_PATH)
        print(f"\nModel ready: {MODEL_PATH}\n")

    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(
            f"Failed to download Whisper model from {MODEL_URL}: {exc}"
        ) from exc


def start_whisper() -> None:

    global whisper_process

    if not WHISPER_SERVER.is_file():

        raise FileNotFoundError(

            f"Whisper server executable not found:\n{WHISPER_SERVER}\n\n"

            "Build whisper.cpp first."

        )

    print("=" * 64)

    print(" Starting whisper.cpp")

    print("=" * 64)

    print(f"Executable: {WHISPER_SERVER}")

    print(f"Model     : {MODEL_PATH}")

    print(f"HTTP      : http://{WHISPER_HOST}:{WHISPER_PORT}")

    print()

    whisper_process = subprocess.Popen(

        [

            str(WHISPER_SERVER),

            "-m",

            str(MODEL_PATH),

            "--host",

            WHISPER_HOST,

            "--port",

            str(WHISPER_PORT),

            "-l",

            "en",

            "-t",

            "4",

        ]

    )

    if not wait_for_whisper():

        exit_code = whisper_process.poll()

        if exit_code is not None:

            raise RuntimeError(

                f"whisper-server exited with code {exit_code}"

            )

        whisper_process.terminate()

        raise RuntimeError(

            "whisper-server did not become ready on 127.0.0.1:8080"

        )

    print("whisper.cpp is READY.\n")

def stop_whisper() -> None:

    global whisper_process

    if whisper_process is None:

        return

    if whisper_process.poll() is None:

        print("\nStopping whisper.cpp...")

        whisper_process.terminate()

        try:

            whisper_process.wait(timeout=5)

        except subprocess.TimeoutExpired:

            whisper_process.kill()

            whisper_process.wait()

    whisper_process = None

def transcribe(pcm: bytes) -> str:

    """Run one synchronous inference against local whisper-server."""

    if len(pcm) < int(MIN_PARTIAL_AUDIO_SEC * SAMPLE_RATE * SAMPLE_WIDTH):

        return ""

    wav_data = pcm_to_wav(pcm)

    body, content_type = make_multipart(wav_data)

    conn = http.client.HTTPConnection(

        WHISPER_HOST,

        WHISPER_PORT,

        timeout=30,

    )

    try:

        conn.request(

            "POST",

            "/inference",

            body=body,

            headers={

                "Content-Type": content_type,

                "Content-Length": str(len(body)),

                "Connection": "close",

            },

        )

        response = conn.getresponse()

        raw = response.read()

        if response.status != 200:

            raise RuntimeError(

                f"HTTP {response.status}: "

                f"{raw.decode('utf-8', errors='replace')}"

            )

        payload = json.loads(raw.decode("utf-8"))

        return str(payload.get("text", "")).strip()

    finally:

        conn.close()

# ============================================================

# LIVE PARTIAL TASK

# ============================================================

async def partial_worker(audio: bytearray, state: dict) -> None:

    """

    Rolling-window Whisper preview.

    whisper-server /inference is file-based rather than a native online

    decoder, so this worker periodically transcribes the most recent

    window and prints the newest text on the Mac.

    """

    last_text = ""

    while not state["ended"]:

        await asyncio.sleep(PARTIAL_INTERVAL_SEC)

        if state["ended"]:

            return

        snapshot = bytes(audio)

        max_bytes = int(

            PARTIAL_WINDOW_SEC * SAMPLE_RATE * SAMPLE_WIDTH

        )

        if len(snapshot) > max_bytes:

            snapshot = snapshot[-max_bytes:]

        if len(snapshot) < int(

            MIN_PARTIAL_AUDIO_SEC * SAMPLE_RATE * SAMPLE_WIDTH

        ):

            continue

        try:

            async with INFERENCE_LOCK:

                text = await asyncio.to_thread(

                    transcribe,

                    snapshot,

                )

            if text and text != last_text:

                print(

                    f"\rPARTIAL: {text:<110}",

                    end="",

                    flush=True,

                )

                last_text = text

        except Exception as exc:

            print(f"\n[PARTIAL ERROR] {exc}")

# ============================================================

# ESP32 WEBSOCKET HANDLER

# ============================================================

async def handle_client(websocket):

    peer = getattr(websocket, "remote_address", None)

    print()

    print(f"[WS] ESP32 connected: {peer}")

    audio = bytearray()

    state = {

        "started": False,

        "ended": False,

    }

    partial_task = None

    try:

        async for message in websocket:

            # ----------------------------

            # TEXT CONTROL

            # ----------------------------

            if isinstance(message, str):

                command = message.strip().upper()

                if command == "START":

                    audio.clear()

                    state["started"] = True

                    state["ended"] = False

                    if partial_task is not None:

                        partial_task.cancel()

                        try:

                            await partial_task

                        except asyncio.CancelledError:

                            pass

                    partial_task = asyncio.create_task(

                        partial_worker(audio, state)

                    )

                    print("\n[WS] START")

                    print("[ASR] Live preview started.")

                    continue

                if command == "END":

                    if not state["started"]:

                        await websocket.send(

                            json.dumps({

                                "type": "final",

                                "text": "",

                            })

                        )

                        continue

                    print("\n\n[WS] END")

                    state["ended"] = True

                    if partial_task is not None:

                        partial_task.cancel()

                        try:

                            await partial_task

                        except asyncio.CancelledError:

                            pass

                        partial_task = None

                    final_audio = bytes(audio)

                    duration = len(final_audio) / (

                        SAMPLE_RATE * SAMPLE_WIDTH

                    )

                    print(

                        f"[ASR] Final audio: {len(final_audio)} bytes "

                        f"({duration:.2f}s)"

                    )

                    try:

                        async with INFERENCE_LOCK:

                            final_text = await asyncio.to_thread(

                                transcribe,

                                final_audio,

                            )

                        print(

                            f"[FINAL] "

                            f"{final_text or '[no speech]'}"

                        )

                        # The supplied ESP32 reads a text frame only after END.

                        await websocket.send(

                            json.dumps(

                                {

                                    "type": "final",

                                    "text": final_text,

                                },

                                ensure_ascii=False,

                            )

                        )

                    except Exception as exc:

                        print(f"[ASR ERROR] {exc}")

                        await websocket.send(

                            json.dumps(

                                {

                                    "type": "final",

                                    "text": "",

                                    "error": str(exc),

                                }

                            )

                        )

                    audio.clear()

                    state["started"] = False

                    print("[WS] Final result sent to ESP32.")

                    continue

                print(f"[WS] Ignored command: {message}")

                continue

            # ----------------------------

            # BINARY PCM AUDIO

            # ----------------------------

            if isinstance(message, (bytes, bytearray)):

                if not state["started"]:

                    continue

                audio.extend(message)

                if len(message) != EXPECTED_ESP32_FRAME_BYTES:

                    print(

                        f"\n[WS] Frame size: {len(message)} bytes "

                        f"(expected {EXPECTED_ESP32_FRAME_BYTES})"

                    )

    except ConnectionClosed:

        print("\n[WS] ESP32 disconnected.")

    except Exception as exc:

        print(f"\n[WS ERROR] {exc}")

    finally:

        state["ended"] = True

        if partial_task is not None:

            partial_task.cancel()

            try:

                await partial_task

            except asyncio.CancelledError:

                pass

        print("[WS] Connection closed.")

# ============================================================

# MAIN

# ============================================================

async def main() -> None:

    start_whisper()

    print("=" * 64)

    print(" Bumblebee LIVE Whisper Server")

    print("=" * 64)

    print(f"WebSocket : ws://0.0.0.0:{WS_PORT}")

    print(f"Whisper   : http://127.0.0.1:{WHISPER_PORT}/inference")

    print()

    print("Waiting for ESP32...")

    print("=" * 64)

    try:

        async with serve(

            handle_client,

            WS_HOST,

            WS_PORT,

            max_size=None,

            ping_interval=20,

            ping_timeout=20,

        ):

            await asyncio.Future()

    finally:

        stop_whisper()

if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print("\nStopped.")

        stop_whisper()

    except Exception as exc:

        print(f"\nFATAL: {exc}")

        stop_whisper()

        sys.exit(1)
