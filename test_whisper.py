"""Quick test: record from mic and transcribe with Whisper."""

import subprocess
import tempfile
import sys
import os
import whisper

DURATION = 5  # seconds
MODEL = "small"  # better accuracy for casual speech and accents


def record_audio(duration: int, output_path: str):
    """Record audio from the default mic using sox."""
    print(f"Recording for {duration} seconds... speak now!")
    result = subprocess.run(
        ["rec", "-b", "16", output_path, "rate", "16000", "channels", "1", "trim", "0", str(duration)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"rec stderr: {result.stderr}")
    print("Recording complete.")


def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else DURATION

    print(f"Loading Whisper model '{MODEL}'...")
    model = whisper.load_model(MODEL)
    print("Model loaded.")

    audio_path = "/tmp/whisper_test.wav"

    record_audio(duration, audio_path)

    # Check if file has actual audio data
    size = os.path.getsize(audio_path)
    print(f"Recorded file size: {size} bytes")

    if size < 1000:
        print("File too small — mic may not be capturing. Check System Preferences > Sound > Input")
        return

    print("Transcribing...")
    result = model.transcribe(audio_path)
    print(f"\nYou said: {result['text']}")
    print(f"Language: {result.get('language', 'unknown')}")


if __name__ == "__main__":
    main()
