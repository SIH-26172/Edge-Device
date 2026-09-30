# Bumblebee Whisper ASR Server

This folder contains the live ASR server for the Bumblebee ESP32 system.

## Data flow

ESP32 -> WebSocket -> whisper_live_server.py -> whisper.cpp -> Whisper model

## ESP32 protocol

The server expects one WebSocket connection using:

- `START` — text control message
- PCM16 mono 16 kHz audio — binary frames, normally 1024 bytes
- `END` — text control message

The server continuously receives audio, prints rolling Whisper `PARTIAL` text on the Mac while the ESP32 is speaking, and runs a final transcription after `END`.

Only the final JSON result is sent back to the existing ESP32 client after `END`.

## Network configuration

### ESP32 WebSocket server

- Host: `0.0.0.0`
- Port: `8765`

### Local whisper.cpp server

- Host: `127.0.0.1`
- Port: `8080`

The Python server starts the local `whisper-server` executable automatically.

## whisper.cpp requirement

The Python server expects the whisper.cpp executable at:

```text
~/whisper.cpp/build/bin/whisper-server
```

Build/install whisper.cpp separately. Do **not** commit the `build/` directory to this repository.

## Whisper model

The server uses:

```text
~/whisper.cpp/models/ggml-base.en.bin
```

The model is downloaded automatically on first server start when it is not already present.

The model file should **not** be committed to Git.

## Python setup

From the repository root:

```bash
cd ASR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run:

```bash
python3 whisper_live_server.py
```

## Important

The ASR server expects `whisper-server` from whisper.cpp to be built and available at the path above. The Python file handles the ESP32 WebSocket connection, audio buffering, rolling partial transcription, final transcription, and JSON response.
