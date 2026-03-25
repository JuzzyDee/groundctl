"""Test the SpeechListener — background mode and active listen."""

import sys
import time

sys.path.insert(0, ".")
from groundctl.listener import SpeechListener

listener = SpeechListener()


def test_background():
    """Test the always-on background listener."""
    print("Starting background listener...")
    print("Talk naturally. Check for transcriptions every 5 seconds.")
    print("Ctrl+C to stop.\n")

    listener.start()

    try:
        while True:
            time.sleep(5)
            entries = listener.get_recent_speech()
            if entries:
                for entry in entries:
                    print(f"  [{entry['iso']}] {entry['text']}")
            else:
                print("  (nothing new)")
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()


def test_active():
    """Test the single-utterance active listen."""
    print("Active listen mode — say something (30s timeout)...\n")

    text = listener.listen_once(timeout=30.0)
    if text:
        print(f"You said: {text}")
    else:
        print("(no speech detected)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "background"

    if mode == "active":
        test_active()
    else:
        test_background()
