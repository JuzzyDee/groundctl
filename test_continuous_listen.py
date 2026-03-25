"""Continuous listening with VAD — talk naturally, pause when done."""

import numpy as np
import pyaudio
import whisper
import torch
import sys
import io
import wave
import tempfile

# Audio config
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # samples per read (~32ms at 16kHz)
CHANNELS = 1
FORMAT = pyaudio.paInt16

# VAD config
ENERGY_THRESHOLD = 500  # RMS energy threshold for speech detection
SILENCE_CHUNKS = 25  # ~800ms of silence to consider speech ended
MIN_SPEECH_CHUNKS = 10  # minimum ~320ms of speech to bother transcribing

# Whisper
MODEL = "small"


def rms_energy(audio_chunk: bytes) -> float:
    """Calculate RMS energy of audio chunk."""
    data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(data ** 2))


def audio_to_wav_bytes(frames: list[bytes], sample_rate: int) -> str:
    """Convert raw audio frames to a wav file, return path."""
    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    return path


def main():
    print(f"Loading Whisper model '{MODEL}'...")
    model = whisper.load_model(MODEL)
    print("Model loaded.")
    print()
    print("=" * 50)
    print("CONTINUOUS LISTENING MODE")
    print("Talk naturally. Pause when you're done.")
    print("I'll transcribe when you stop speaking.")
    print("Ctrl+C to quit.")
    print("=" * 50)
    print()

    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    speech_buffer = []
    silence_count = 0
    is_speaking = False
    speech_chunk_count = 0

    # Calibrate noise floor
    print("Calibrating... stay quiet for 2 seconds...")
    noise_frames = []
    for _ in range(int(SAMPLE_RATE / CHUNK_SIZE * 2)):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        noise_frames.append(rms_energy(data))

    noise_floor = np.mean(noise_frames)
    threshold = max(ENERGY_THRESHOLD, noise_floor * 3)
    print(f"Noise floor: {noise_floor:.0f}, Threshold: {threshold:.0f}")
    print()
    print("Listening... speak whenever you're ready.")
    print()

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            energy = rms_energy(data)

            if energy > threshold:
                if not is_speaking:
                    is_speaking = True
                    speech_chunk_count = 0
                    sys.stdout.write("🎤 ")
                    sys.stdout.flush()

                speech_buffer.append(data)
                silence_count = 0
                speech_chunk_count += 1

            elif is_speaking:
                silence_count += 1
                speech_buffer.append(data)  # keep some trailing silence

                if silence_count > SILENCE_CHUNKS:
                    is_speaking = False

                    if speech_chunk_count >= MIN_SPEECH_CHUNKS:
                        sys.stdout.write("transcribing... ")
                        sys.stdout.flush()

                        wav_path = audio_to_wav_bytes(speech_buffer, SAMPLE_RATE)
                        result = model.transcribe(wav_path, fp16=False)
                        text = result['text'].strip()

                        if text and text not in ["", ".", "...", "you", "Thank you."]:
                            print(f"\n💬 You said: {text}")
                            print()
                        else:
                            print("(noise/unclear)")
                            print()
                    else:
                        pass  # too short, probably a cough or bump

                    speech_buffer = []
                    silence_count = 0
                    speech_chunk_count = 0

    except KeyboardInterrupt:
        print("\n\nStopped listening.")

    stream.stop_stream()
    stream.close()
    pa.terminate()


if __name__ == "__main__":
    main()
