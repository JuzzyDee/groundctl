"""Background speech listener — always-on VAD + Whisper transcription.

Runs as a daemon thread alongside the MCP server. Continuously monitors
the microphone, detects speech via energy-based VAD, and transcribes
with Whisper. Transcriptions are written to a JSONL file that the MCP
server tails for new utterances.
"""

import json
import threading
import time
import tempfile
import wave
from pathlib import Path

import numpy as np
import pyaudio
import whisper

# Audio config
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # ~32ms at 16kHz
CHANNELS = 1
FORMAT = pyaudio.paInt16

# VAD config
ENERGY_THRESHOLD = 500
SILENCE_CHUNKS = 25  # ~800ms of silence = end of utterance
MIN_SPEECH_CHUNKS = 10  # ~320ms minimum to bother transcribing

# Whisper
MODEL_NAME = "small"

# Output
DEFAULT_TRANSCRIPT_PATH = Path.home() / ".groundctl" / "transcriptions.jsonl"


def _rms_energy(audio_chunk: bytes) -> float:
    data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(data ** 2))


def _frames_to_wav(frames: list[bytes], sample_rate: int) -> str:
    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    return path


class SpeechListener:
    """Always-on speech listener with VAD and Whisper transcription.

    Writes transcriptions to a JSONL file. The MCP server reads this
    file to check for new speech without any threading complexity in
    the tool handlers.
    """

    def __init__(self, transcript_path: Path | None = None, model_name: str = MODEL_NAME):
        self.transcript_path = transcript_path or DEFAULT_TRANSCRIPT_PATH
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = None
        self._running = False
        self._thread = None
        self._last_read_pos = 0

    def start(self):
        """Start the background listener thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def get_recent_speech(self) -> list[dict]:
        """Read new transcriptions since last check.

        Called by get_status to piggyback speech data on telemetry polls.
        Returns list of {timestamp, text} dicts.
        """
        if not self.transcript_path.exists():
            return []

        entries = []
        with open(self.transcript_path, "r") as f:
            f.seek(self._last_read_pos)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            self._last_read_pos = f.tell()

        return entries

    def listen_once(self, timeout: float = 30.0) -> str | None:
        """Active listen — block until one utterance is captured.

        Used by the `listen` MCP tool for explicit conversation.
        Returns transcribed text, or None on timeout.
        """
        if self._model is None:
            self._model = whisper.load_model(self.model_name)

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        try:
            # Quick calibration
            noise_frames = []
            for _ in range(int(SAMPLE_RATE / CHUNK_SIZE * 1)):
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                noise_frames.append(_rms_energy(data))

            noise_floor = np.mean(noise_frames)
            threshold = max(ENERGY_THRESHOLD, noise_floor * 3)

            speech_buffer = []
            silence_count = 0
            is_speaking = False
            speech_chunk_count = 0
            start_time = time.time()

            while time.time() - start_time < timeout:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                energy = _rms_energy(data)

                if energy > threshold:
                    if not is_speaking:
                        is_speaking = True
                        speech_chunk_count = 0
                    speech_buffer.append(data)
                    silence_count = 0
                    speech_chunk_count += 1

                elif is_speaking:
                    silence_count += 1
                    speech_buffer.append(data)

                    if silence_count > SILENCE_CHUNKS:
                        if speech_chunk_count >= MIN_SPEECH_CHUNKS:
                            return self._transcribe(speech_buffer)
                        # Too short, reset and keep listening
                        speech_buffer = []
                        silence_count = 0
                        speech_chunk_count = 0
                        is_speaking = False

            return None  # Timeout

        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def _transcribe(self, frames: list[bytes]) -> str | None:
        """Transcribe audio frames with Whisper."""
        if self._model is None:
            self._model = whisper.load_model(self.model_name)

        wav_path = _frames_to_wav(frames, SAMPLE_RATE)
        result = self._model.transcribe(wav_path, fp16=False)
        text = result["text"].strip()

        # Filter Whisper hallucinations on noise
        if not text or text in ("", ".", "...", "you", "Thank you.", "Thanks for watching!"):
            return None

        return text

    def _write_entry(self, text: str):
        """Append a transcription to the JSONL file."""
        entry = {
            "timestamp": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "text": text,
        }
        with open(self.transcript_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _listen_loop(self):
        """Main background listening loop. Runs in daemon thread."""
        import sys

        # Load model in background thread
        print("[listener] Loading Whisper model...", file=sys.stderr)
        self._model = whisper.load_model(self.model_name)
        print("[listener] Model loaded.", file=sys.stderr)

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        # Calibrate
        print("[listener] Calibrating noise floor...", file=sys.stderr)
        noise_frames = []
        for _ in range(int(SAMPLE_RATE / CHUNK_SIZE * 2)):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            noise_frames.append(_rms_energy(data))

        noise_floor = np.mean(noise_frames)
        threshold = max(ENERGY_THRESHOLD, noise_floor * 3)
        print(f"[listener] Noise floor: {noise_floor:.0f}, Threshold: {threshold:.0f}", file=sys.stderr)
        print("[listener] Listening...", file=sys.stderr)

        speech_buffer = []
        silence_count = 0
        is_speaking = False
        speech_chunk_count = 0

        while self._running:
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            except Exception:
                continue

            energy = _rms_energy(data)

            if energy > threshold:
                if not is_speaking:
                    is_speaking = True
                    speech_chunk_count = 0
                speech_buffer.append(data)
                silence_count = 0
                speech_chunk_count += 1

            elif is_speaking:
                silence_count += 1
                speech_buffer.append(data)

                if silence_count > SILENCE_CHUNKS:
                    is_speaking = False

                    if speech_chunk_count >= MIN_SPEECH_CHUNKS:
                        text = self._transcribe(speech_buffer)
                        if text:
                            self._write_entry(text)
                            print(f"[listener] Heard: {text}", file=sys.stderr)

                    speech_buffer = []
                    silence_count = 0
                    speech_chunk_count = 0

        stream.stop_stream()
        stream.close()
        pa.terminate()
