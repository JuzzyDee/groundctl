# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Ground Control (groundctl) — the Primary Motor Cortex for an EarthRover Mini+. Translates Claude's intent into physical movement. Part of the holy trinity: audio-analyzer (ears) + Memoria (memory) + groundctl (presence).

Python-based. No external linter or formatter config — use standard practices.

## Build & Run Commands

```bash
pip install -e .                          # install in dev mode
python -m groundctl.mcp_server            # run MCP server (stdio)
python test_listener.py                   # test background speech listener
python test_listener.py active            # test single-utterance listen
python test_continuous_listen.py          # test continuous transcription
```

## Architecture

### The Nervous System (Biology Mapping)

| Biology | Component | What it does |
|---------|-----------|-------------|
| Prefrontal cortex | Claude (API) | Planning, intent, decisions |
| Primary motor cortex | Intent executor | Translates intent to motor plans |
| Cerebellum | IL model (Phase 2) | Learned smooth execution |
| Spinal reflex | YOLO + Depth safety | Emergency stop, no thinking |
| Sensory cortex | Whisper + cameras + YOLO | Perception feeding all layers |
| Hippocampus | Memoria + spatial memory | Memory and place recognition |

### Three Control Speeds

- **20fps**: YOLO11n object tracking + depth safety (reflex)
- **10Hz**: Intent executor motor commands (muscle memory)
- **Every few seconds**: Claude API heartbeat (conscious thought)

### Two API Streams (Phase 2 — when rover goes API)

- **Heartbeat stream**: Haiku, every 5 seconds, tiny context. Status + speech + intent management. Pauses during conversation.
- **Conversation stream**: Sonnet/Opus, on demand, carries Memoria context. Triggered by speech or decisions needing real thinking.
- Conversation cadence replaces heartbeat while talking — each response includes a status check.
- Estimated cost: ~$0.50/hour using prompt caching.

## Modules

### `mcp_server.py` — MCP server (25 tools)
Perception, control, navigation, missions, intents, speech, spatial memory, interventions. Entry point for Claude Code usage.

### `rover_client.py` — SDK REST API wrapper
Async HTTP client for the EarthRover SDK running on localhost:8000. Cameras, movement, telemetry, TTS, missions.

### `navigation.py` — GPS waypoint navigation
Haversine distance, bearing calculation, proportional steering. Phase 1 classical control — gets replaced by IL model for execution quality.

### `intent.py` — Intent executor
Decouples Claude's decision rate from the rover's 10Hz control rate. Four intent types:
- **GO_FORWARD**: maintain speed and heading
- **FOLLOW_BEARING**: continuous heading correction
- **TURN_TO**: rotate to face a bearing
- **NAVIGATE_TO**: GPS waypoint with proportional steering

`_compute_steering()` is the method the IL model replaces in Phase 2.

### `perception.py` — YOLO + DepthAnything
Two perception layers:
- **YOLO11n**: object detection at 1.5ms/frame. Tracking data for follow/approach intents. Claude Cam overlay renderer for demo.
- **DepthAnything V2 Small**: depth-based obstacle detection. Catches what YOLO misses (barriers, walls, unknown objects). Gradient discontinuity detection distinguishes ground from obstacles. Emergency stop at >50% centre close ratio.

Note: YOLO is trained on COCO (80 classes). Kangaroos are classified as elephants. Australian wildlife requires fine-tuning or YOLO-World for accurate labels. Claude's own vision handles scene understanding — YOLO is for tracking and safety, not identification.

### `listener.py` — Speech listener
Always-on background thread: VAD (energy-based) + Whisper (small model). Writes transcriptions to `~/.groundctl/transcriptions.jsonl`. Two modes:
- **Background**: daemon thread, writes to JSONL, piggybacked on status polls
- **Active**: `listen_once()` blocks until speech detected

Apple Voice Isolation via Loopback planned for noise filtering.

### `places.py` — Spatial memory (Phase 1)
Named places with GPS coords + radius. JSON-backed at `~/.groundctl/places.json`. Mark spots, navigate back, check current location. Visit counting for future Memoria integration.

Phase 2: Memoria spatial memory type with Ebbinghaus decay (visit-based, not recall-based), Hebbian place-event linking.

## Key Design Decisions

- **Emergency stop allows reverse**: `linear <= 0` always permitted during emergency stop. The rover can self-recover by backing up. Only forward movement is blocked.
- **Claude sees raw frames**: Claude's multimodal vision is better than any detection model. YOLO provides tracking data (offsets, sizes) for the intent executor. Claude provides scene understanding.
- **Depth over size ratio**: Approach intent uses DepthAnything for proximity, not bounding box size. Depth is object-agnostic — works regardless of what you're approaching.
- **Two-threshold depth safety**: Gradient discontinuity (object breaks ground gradient) AND close ratio (too much of centre is near). Both paths can trigger emergency stop independently.
- **Rotate-then-drive pattern**: Validated on real EarthRover test drive. Natural control pattern for latency-affected operation. Turn to face target, then drive forward.
- **Named places separate from waypoints**: `navigate_to` for transient GPS coordinates. `go_to_place` for remembered named locations. Different tools, different purposes.

## Demo Plan

See `DEMO_PLAN.md` for the full storyboard, MVP intents, gear list, and distribution plan.

## Training Pipeline (Phase 2)

- **Dataset**: FrodoBots-2K (1,300 hours, CC-BY-SA). Perth/Adelaide subsets for Australian environment.
- **Architecture**: Raw camera frames → IL model → [linear, angular]. No YOLO in training — model learns directly from pixels.
- **Collection**: Every manual drive generates training pairs. Sim rig (Simagic P2000) for premium analogue input.
- **Fine-tuning**: Pre-train on FrodoBots, fine-tune on Yeppoon-specific data.
- **Hardware**: RTX 4090 24GB for training. Inference runs on rover's compute.

## Hardware

- **Rover**: EarthRover Mini+ ($349 USD). 4G-enabled, front+rear cameras, GPS, IMU, speaker, mic. 3.5 km/h max speed. ~5 hours battery. Weatherproof.
- **Arriving**: April 2nd, 2026. DHL from Hong Kong via Wuhan.
- **Test ground**: Norfolk Drive cul-de-sac, Hidden Valley QLD. Open grass, no traffic, WiFi mesh coverage.

## Dependencies

- **ultralytics** — YOLO11n object detection
- **transformers** — DepthAnything V2 depth estimation
- **openai-whisper** — speech transcription
- **pyaudio** — microphone input
- **httpx** — async HTTP for SDK communication
- **mcp** — MCP server SDK
- **numpy** — array operations for perception
- **opencv-python-headless** — frame processing and overlay rendering
